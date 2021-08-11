import logging
from pathlib import Path

import numpy as np
from astropy import units as u
import astropy.wcs
from matplotlib import pyplot as plt
import spectral_cube
from astropy.visualization import quantity_support

quantity_support()
from astropy.table import QTable

from diskchef.chemistry.andes import ReadAndesData
from diskchef.lamda.line import Line
from diskchef.maps import RadMCRTSingleCall

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
        datefmt='%m.%d.%Y %H:%M:%S',
    )
    logger = logging.getLogger(__name__)
    line_list = [Line(name='CO J=2-1', transition=2, molecule='CO'),
                 # Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
                 Line(name='H2CO 3_03-2_02', transition=3, molecule='pH2CO'),
                 # Line(name='CO J=3-2', transition=3, molecule='CO'),
                 # Line(name='13CO J=3-2', transition=3, molecule='13CO'),
                 # Line(name='C18O J=3-2', transition=3, molecule='C18O'),
                 ]
    fig_main, ax_main = plt.subplots(1, len(line_list), figsize=(5 * len(line_list), 5))

    for fileindex in [5]:
        chem = ReadAndesData(folder='example_andes_input/data', index=fileindex)
        chem.table['13CO'] = chem.table['CO'] / 70
        chem.table['C18O'] = chem.table['CO'] / 550
        chem.table['pH2CO'] = chem.table['H2CO'] / 4
        folder = Path(f'example_andes/radmc_{fileindex:05d}')
        folder.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
        chem.physics.plot_density(axes=ax[0, 0])
        tempplot = chem.physics.plot_temperatures(axes=ax[1, 0])
        tempplot.contours("Gas temperature", [20, 40] * u.K)

        chem.plot_chemistry("CO", "HCO+", axes=ax[0, 1])
        chem.plot_chemistry("CN", "HCN", axes=ax[1, 1])

        coplot = chem.plot_absolute_chemistry("CO", "CO", axes=ax[0, 2], maxdepth=1e9)
        coplot.contours("Gas temperature", [20, 40] * u.K, clabel_kwargs={"fmt": "T=%d K"})
        chem.plot_absolute_chemistry("H2CO", "H2CO", axes=ax[1, 2])

        fig2, ax2 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))

        chem.physics.plot_column_densities(axes=ax2[0, 0])
        chem.plot_column_densities(axes=ax2[1, 0], species=["CO", "CS"])
        chem.plot_column_densities(axes=ax2[0, 1], species=["H2CO"])
        chem.plot_column_densities(axes=ax2[1, 1], species=["N2H+", "HCO+"])

        fig.savefig(folder / f"report_{fileindex}.pdf")
        fig2.savefig(folder / f"report_{fileindex}_coldens.pdf")

        map = RadMCRTSingleCall(
            folder=folder,
            chemistry=chem, line_list=line_list, outer_radius=200 * u.au, radii_bins=30, theta_bins=30)
        map.create_files(channels_per_line=200, window_width=8 * u.km / u.s)
        map.run(
            inclination=30 * u.deg, position_angle=0 * u.deg,
            velocity_offset=6 * u.km / u.s, threads=2, distance=700 * u.pc, npix=40
        )

        for transition_name, ax in zip([line.name for line in line_list], ax_main):
            scube = spectral_cube.SpectralCube.read(folder / f"{transition_name}_image.fits")
            scube = scube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
            pixel_area_units = u.Unit(scube.wcs.celestial.world_axis_units[0]) * u.Unit(
                scube.wcs.celestial.world_axis_units[1])
            pixel_area = astropy.wcs.utils.proj_plane_pixel_area(scube.wcs.celestial) * pixel_area_units

            spectrum = (scube * pixel_area).to(u.mJy).sum(axis=(1, 2))  # 1d spectrum in Jy
            flux = np.abs(np.trapz(spectrum, scube.spectral_axis))
            tbl = astropy.table.QTable(
                [scube.spectral_axis, u.Quantity(spectrum)],
                meta={"flux": flux},
                names=["Velocity", "Flux density"],
            )
            tbl.write(folder / f"{transition_name}_spectrum.ecsv", overwrite=True)

            ax.set_title(transition_name)
            ax.plot(scube.spectral_axis, u.Quantity(spectrum), label=fileindex)

    ax_main[0].legend()
    fig_main.savefig(folder / "../line_profiles.png")

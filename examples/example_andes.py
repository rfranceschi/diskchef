import logging
import shutil
from pathlib import Path
from astropy import units as u

from matplotlib import pyplot as plt
import spectral_cube
from astropy.visualization import quantity_support

quantity_support()
from astropy.table import QTable
from scipy.integrate import trapz

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
    fig_main, ax_main = plt.subplots()
    for fileindex in [5]:
        chem = ReadAndesData(folder='example_andes/data', index=fileindex)
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
            chemistry=chem, line_list=[
                #    Line(name='CO J=2-1', transition=2, molecule='CO'),
                # Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
                Line(name='H2CO 3_03-2_02', transition=3, molecule='pH2CO'),
                #    Line(name='CO J=3-2', transition=3, molecule='CO'),
                #    Line(name='13CO J=3-2', transition=3, molecule='13CO'),
                #    Line(name='C18O J=3-2', transition=3, molecule='C18O'),
            ], outer_radius=200 * u.au, radii_bins=30, theta_bins=30)
        map.create_files(channels_per_line=200, window_width=8 * u.km / u.s)
        map.run(
            inclination=30 * u.deg, position_angle=0 * u.deg,
            velocity_offset=6 * u.km / u.s, threads=2, distance=700 * u.pc, npix=40
        )

        cube = spectral_cube.SpectralCube.read(folder / "H2CO 3_03-2_02_image.fits")
        cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
        spectrum = cube.sum(axis=(1, 2))
        velax = cube.spectral_axis
        tbl = QTable(data=[velax, spectrum], names=["Velocity", "Flux"])
        tbl.meta["Integrated flux"] = abs(trapz(spectrum, velax))
        tbl.write(folder / "demo.txt", format="ascii.ecsv", overwrite=True)
        ax_main.plot(velax, spectrum)

    fig_main.savefig(folder / "../andes.png")

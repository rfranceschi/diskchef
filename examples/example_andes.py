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

shutil.rmtree('radmc', ignore_errors=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
bins = 10
for fileindex in [5, 15, 30]:
    chem = ReadAndesData(folder='/home/abaker/Documents/diskchef/Molyarova_etal2018/', index=fileindex)

    print(chem.table[chem.table.colnames[0:10]][:5])

    chem.table['13CO'] = chem.table['CO'] / 70
    chem.table['C18O'] = chem.table['CO'] / 550
    chem.table['pH2CO'] = chem.table['H2CO'] / 4
    folder = Path(f'example_andes/radex_{fileindex:05d}')
    map = RadMCRTSingleCall(
        folder=folder,
        chemistry=chem, line_list=[
            #    Line(name='CO J=2-1', transition=2, molecule='CO'),
            # Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
            Line(name='H2CO 3_03-2_02', transition=3, molecule='pH2CO'),
            #    Line(name='CO J=3-2', transition=3, molecule='CO'),
            #    Line(name='13CO J=3-2', transition=3, molecule='13CO'),
            #    Line(name='C18O J=3-2', transition=3, molecule='C18O'),
        ], outer_radius=200 * u.au, radii_bins=100, theta_bins=100)
    map.create_files(channels_per_line=200, window_width=8 * u.km / u.s)
    map.run(
        inclination=30 * u.deg, position_angle=0 * u.deg,
        velocity_offset=6 * u.km / u.s, threads=2, distance=700 * u.pc, npix=100
    )

    cube = spectral_cube.SpectralCube.read(folder / "H2CO 3_03-2_02_image.fits")
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    spectrum = cube.sum(axis=(1, 2))
    velax = cube.spectral_axis
    tbl = QTable(data=[velax, spectrum], names=["Velocity", "Flux"])
    tbl.meta["Integrated flux"] = abs(trapz(spectrum, velax))
    tbl.write(folder / "demo.txt", format="ascii.ecsv", overwrite=True)
    plt.plot(velax, spectrum)

    plt.savefig(folder / "demo.png")


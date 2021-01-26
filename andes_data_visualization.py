import logging
import shutil
from astropy import units as u
from matplotlib import pyplot as plt
import spectral_cube
from astropy.visualization import quantity_support
quantity_support()
from astropy.table import QTable
from scipy.integrate import trapz

from diskchef.chemistry.andes import ReadAndesData
from diskchef.lamda.line import Line
from diskchef.maps.radmcrt import RadMCRTSingleCall

shutil.rmtree('radmc', ignore_errors=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
bins = 10
for fileindex in range(4,5):
    chem = ReadAndesData(folder='/home/abaker/Dropbox/1Projects/Programs/ANDES_2.0/data_output_test/', index=fileindex)
    # chem = ReadAndesData(folder='example_data/andes2/data', index=4)

    print(chem.table[chem.table.colnames[0:10]][:5])

    chem.table['13CO'] = chem.table['CO'] / 70
    chem.table['C18O'] = chem.table['CO'] / 550

    map = RadMCRTSingleCall(
        folder=f'radmc_{fileindex:05d}',
        chemistry=chem, line_list=[
            #    Line(name='CO J=2-1', transition=1, molecule='CO'),
            Line(name='HCO+ J=3-2', transition=2, molecule='HCO+'),
            #    Line(name='CO J=3-2', transition=2, molecule='CO'),
            #    Line(name='13CO J=3-2', transition=2, molecule='13CO'),
            #    Line(name='C18O J=3-2', transition=2, molecule='C18O'),
        ], outer_radius=200 * u.au, radii_bins=100, theta_bins=100)
    map.create_files(channels_per_line=100, window_width=15*u.km/u.s)
    map.run(
        inclination=35.18 * u.deg, position_angle=79.19 * u.deg,
        velocity_offset=6 * u.km / u.s, threads=2, distance=128 * u.pc, npix=100
    )

    cube = spectral_cube.SpectralCube.read(f"radmc_{fileindex:05d}/HCO+ J=3-2_image.fits")
    cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio')
    spectrum = cube.sum(axis=(1, 2))
    velax = cube.spectral_axis
    tbl = QTable(data=[velax, spectrum], names=["Velocity", "Flux"])
    tbl.meta["Integrated flux"] = abs(trapz(spectrum, velax))
    tbl.write(f"radmc_{fileindex:05d}/demo.txt", format="ascii.ecsv", overwrite=True)
    plt.plot(velax, spectrum)

    plt.savefig(f"radmc_{fileindex:05d}/demo.png")
# chem.run_chemistry()

# physics.plot_density()
# chem.plot_chemistry()
# chem.plot_h2_coldens()

# chem.table.write_e()

plt.show()

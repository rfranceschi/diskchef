import logging
import shutil
from astropy import units as u
from matplotlib import pyplot as plt

from diskchef.chemistry.andes import ReadAndesData
from diskchef.lamda.line import Line
from diskchef.maps.radmcrt import RadMCRTSingleCall

shutil.rmtree('radmc', ignore_errors=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
bins = 10
# chem = ReadAndesData(folder='/home/abaker/Dropbox/1Projects/Programs/ANDES_2.0/data_output/')
chem = ReadAndesData(folder='example_data/andes2/data', index=4)

print(chem.table[chem.table.colnames[0:10]][:5])

chem.table['13CO'] = chem.table['CO'] / 70
chem.table['C18O'] = chem.table['CO'] / 550

map = RadMCRTSingleCall(chemistry=chem, line_list=[
    #    Line(name='CO J=2-1', transition=1, molecule='CO'),
    Line(name='HCO+ J=3-2', transition=2, molecule='HCO+'),
    #    Line(name='CO J=3-2', transition=2, molecule='CO'),
    #    Line(name='13CO J=3-2', transition=2, molecule='13CO'),
    #    Line(name='C18O J=3-2', transition=2, molecule='C18O'),

])
map.create_files(channels_per_line=50)
map.run(
    inclination=35.18 * u.deg, position_angle=79.19 * u.deg,
    velocity_offset=6 * u.km / u.s, threads=2, distance=128 * u.pc, npix=500
)

# chem.run_chemistry()

# physics.plot_density()
# chem.plot_chemistry()
# chem.plot_h2_coldens()

# chem.table.write_e()

plt.show()

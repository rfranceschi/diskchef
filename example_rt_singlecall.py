import pytest
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import logging

from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.chemistry.williams_best_co import NonzeroChemistryWB2014
from diskchef.maps.radmcrt import RadMCRTSingleCall
from diskchef.lamda.line import Line

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
bins = 100
physics = WilliamsBest2014(star_mass=0.52 * u.solMass, radial_bins=bins, vertical_bins=bins)
chem = NonzeroChemistryWB2014(physics)
chem.run_chemistry()
chem.table['13CO'] = chem.table['CO'] / 70
chem.table['C18O'] = chem.table['CO'] / 550

map = RadMCRTSingleCall(chemistry=chem, line_list=[
    Line(name='CO J=2-1', transition=1, molecule='CO'),
    Line(name='CO J=3-2', transition=2, molecule='CO'),
    Line(name='13CO J=3-2', transition=2, molecule='13CO'),
    Line(name='C18O J=3-2', transition=2, molecule='C18O'),

])
map.create_files(channels_per_line=200)
map.run(
    inclination=35.18 * u.deg, position_angle=79.19 * u.deg,
    velocity_offset=6 * u.km / u.s, threads=8, distance=128 * u.pc,
)
map.copy_for_propype()

# physics.plot_density()
# chem.plot_chemistry()
# chem.plot_h2_coldens()
#
# # chem.table.write_e()
#
# plt.show()

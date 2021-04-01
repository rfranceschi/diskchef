"""
Example to generate a Williams & Best 2014 disk model with chemistry,
assume isotopologues ratio, and run line transfer at once for all transitions
"""

import logging
from astropy import units as u
from matplotlib import pyplot as plt

from diskchef.chemistry.williams_best_co import NonzeroChemistryWB2014
from diskchef.lamda.line import Line
from diskchef.maps import RadMCRTSingleCall
from diskchef.physics.williams_best import WilliamsBest2014

# Setting logger configuration to control output.
# Can be done with different verbosity. See python `logging` documentation
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

bins = 100
physics = WilliamsBest2014(star_mass=0.52 * u.solMass, radial_bins=bins, vertical_bins=bins)
folder="example_lines"
chem = NonzeroChemistryWB2014(physics)
chem.run_chemistry()
chem.table['13CO'] = chem.table['CO'] / 70
chem.table['C18O'] = chem.table['CO'] / 550

radmc = RadMCRTSingleCall(
    chemistry=chem, line_list=[
        # Line(name='CO J=2-1', transition=2, molecule='CO'),
        Line(name='CO J=3-2', transition=3, molecule='CO'),
        Line(name='13CO J=3-2', transition=3, molecule='13CO'),
        # Line(name='C18O J=3-2', transition=3, molecule='C18O'),
    ],
    radii_bins=100, theta_bins=100,
    folder=folder
)
radmc.create_files(channels_per_line=200)
radmc.run(
    inclination=35.18 * u.deg, position_angle=79.19 * u.deg,
    velocity_offset=6 * u.km / u.s, threads=8, distance=128 * u.pc,
)

radmc.copy_for_propype()

# Plot the generated physical and chemical distributions
physics.plot_density(folder=folder)
chem.plot_chemistry(folder=folder)
chem.plot_h2_coldens(folder=folder)
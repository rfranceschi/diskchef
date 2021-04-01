import logging
from astropy import units as u
import os
from matplotlib import pyplot as plt

from divan import Divan

from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.chemistry import NonzeroChemistryWB2014
from diskchef.maps import RadMCTherm, RadMCRTSingleCall
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.physics.multidust import DustPopulation
from diskchef.dust_opacity.dust_files import dust_files
from diskchef.lamda.line import Line

from diskchef.maps.radmcrt import RadMCOutput

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

bins = 100
physics = WilliamsBest2014(star_mass=0.52 * u.solMass, radial_bins=bins, vertical_bins=bins)

dust = DustPopulation(dust_files("diana")[0], table=physics.table, name="DIANA dust")
dust.write_to_table()

# This uses experimental chemistry based on KNearestNeighbors machine learning on ANDES2 model grid
# with temperature and density are the only parameters
# (r2 ~ 0.95, good approximation except for density ~1e5 c,-3 and T~30K)
# This is not used for dust transfer anyway

# chem = SciKitChemistry(physics)
chem = NonzeroChemistryWB2014(physics)
folder = "example_lines_dust"
radmc_dust = RadMCTherm(
    chemistry=chem,
    folder=folder
)
radmc_dust.create_files()
radmc_dust.run(threads=8)
radmc_dust.read_dust_temperature()

print(chem.table[0:5])

chem.run_chemistry()
chem.table['13CO'] = chem.table['CO'] / 77
chem.table['C18O'] = chem.table['CO'] / 560

physics.plot_density(folder=folder)
chem.plot_chemistry(folder=folder)
chem.plot_h2_coldens(folder=folder)

radmc = RadMCRTSingleCall(
    chemistry=chem, line_list=[
        # Line(name='CO J=2-1', transition=1, molecule='CO'),
        Line(name='CO J=3-2', transition=3, molecule='CO'),
        # Line(name='13CO J=3-2', transition=2, molecule='13CO'),
        # Line(name='C18O J=3-2', transition=2, molecule='C18O'),
    ],
    radii_bins=100, theta_bins=100,
    folder="example_lines_dust",
    scattering_mode_max=0,
    coordinate="DQ Tau"  # That is not going to represent a DQ Tau model, just a demo how coordinate can be supplied
)
radmc.create_files(channels_per_line=40, window_width=5 * u.km / u.s)
radmc.run(
    inclination=35.18 * u.deg, position_angle=79.19 * u.deg,
    velocity_offset=6 * u.km / u.s, threads=8, distance=128 * u.pc,
)

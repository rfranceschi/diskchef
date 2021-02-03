import logging
from astropy import units as u

from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.lamda.line import Line
from diskchef.maps.radmcrt import RadMCTherm
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.physics.multidust import DustPopulation
from diskchef.dust_opacity.dust_files import dust_files

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logging.captureWarnings(True)
logger = logging.getLogger(__name__)

bins = 100
physics = WilliamsBest2014(star_mass=0.52 * u.solMass, radial_bins=bins, vertical_bins=bins)

dust = DustPopulation(dust_files("draine03", size=1e-5 * u.cm)[0], table=physics.table, name="Default dust")
dust.write_to_table()

# This uses experimental chemistry based on KNearestNeighbors machine learning on ANDES2 model grid
# with temperature and density are the only parameters
# (r2 ~ 0.95, good approximation except for density ~1e5 c,-3 and T~30K)
chem = SciKitChemistry(physics)
chem.run_chemistry()
chem.table['13CO'] = chem.table['CO'] / 70
chem.table['C18O'] = chem.table['CO'] / 550
# This is not used for dust transfer anyway

map = RadMCTherm(
    chemistry=chem,
    folder="example_dust"
)
map.create_files()
map.run(threads=8)

# import pickle
#
# with open("pkl.plk", "wb") as fff:
#     pickle.dump(map.polar_table, fff)

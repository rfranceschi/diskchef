import logging
from astropy import units as u

from matplotlib import pyplot as plt

from divan import Divan

from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.maps import RadMCTherm
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
# This is not used for dust transfer anyway

chem = SciKitChemistry(physics)

map = RadMCTherm(
    chemistry=chem,
    folder="example_dust"
)
map.create_files()
map.run(threads=8)
map.read_dust_temperature()

print(chem.table[0:5])

chem.run_chemistry()
chem.table['13CO'] = chem.table['CO'] / 77
chem.table['C18O'] = chem.table['CO'] / 560

chem.physics.plot_density()

dvn = Divan()
dvn.physical_structure = chem.table
dvn.generate_figure_volume_densities(extra_gas_to_dust=100)
dvn.generate_figure_temperatures()
dvn.generate_figure_temperatures(
    r=map.polar_table.r, z=map.polar_table.z,
    gas_temperature=map.polar_table["RadMC Dust temperature"],
    dust_temperature=map.polar_table["Dust temperature"]
)
dvn.save_figures_pdf("example_dust/figs.pdf")
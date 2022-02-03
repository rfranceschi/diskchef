import logging
from astropy import units as u
from diskchef.engine.plot import Plot2D

from matplotlib import pyplot as plt

from divan import Divan

from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.maps import RadMCTherm
from diskchef.maps.radmc_lines import RadMCRTImage
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.physics.multidust import DustPopulation
from diskchef.dust_opacity.dust_files import dust_files

logging.basicConfig(
    level=logging.INFO,
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

mctherm = RadMCTherm(
    chemistry=chem,
    folder="radmc"
)
mctherm.create_files()
mctherm.run(threads=8)
mctherm.read_dust_temperature()

print(chem.table[0:5])

chem.run_chemistry()
chem.table['13CO'] = chem.table['CO'] / 77
chem.table['C18O'] = chem.table['CO'] / 560

fig, ax = plt.subplots(2, figsize=(5, 10))
chem.physics.plot_density(axes=ax[0])
Plot2D(chem.table, axes=ax[1], data1="RadMC Dust temperature", data2="Original Dust temperature")
fig.savefig("figs.pdf")

image = RadMCRTImage(
    chemistry=chem,
    folder="radmc",
    scattering_mode_max=0
)
image.create_files()
image.run(
    inclination=10 * u.deg, position_angle=30 * u.deg,
    wav=1 * u.mm
)

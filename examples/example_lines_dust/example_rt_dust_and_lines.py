import logging
from astropy import units as u
import os
from matplotlib import pyplot as plt

from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.chemistry import NonzeroChemistryWB2014
from diskchef.maps import RadMCTherm, RadMCRTLines
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.physics.multidust import DustPopulation
from diskchef.dust_opacity.dust_files import dust_files
from diskchef.lamda.line import Line
from diskchef import logging_basic_config

logging_basic_config()

from diskchef.maps.radmcrt import RadMCOutput

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

chem = SciKitChemistry(physics)
# chem = NonzeroChemistryWB2014(physics)
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

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
physics.plot_density(axes=ax[0, 0])
physics.plot_temperatures(axes=ax[0, 1])
chem.plot_chemistry(axes=ax[1, 0], species1="CO", species2="HCO+")
chem.plot_chemistry(axes=ax[1, 1], species1="CN", species2="HCN")

radmc = RadMCRTLines(
    chemistry=chem, line_list=[
        # Line(name='CO J=2-1', transition=1, molecule='CO'),
        Line(name='CO J=3-2', transition=3, molecule='CO'),
        Line(name='13CO J=3-2', transition=2, molecule='13CO'),
        # Line(name='C18O J=3-2', transition=2, molecule='C18O'),
    ],
    radii_bins=100, theta_bins=100,
    folder="example_lines_dust",
    scattering_mode_max=0,
    coordinate="DQ Tau"  # That is not going to represent a DQ Tau model, just a demo how coordinate can be supplied
)
radmc.create_files(channels_per_line=31, window_width=5 * u.km / u.s)
radmc.run(
    inclination=35.18 * u.deg, position_angle=79.19 * u.deg,
    velocity_offset=6 * u.km / u.s, threads=8, distance=128 * u.pc,
)
radmc.channel_maps()

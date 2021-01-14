import pytest
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import logging

from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.chemistry.williams_best_co import NonzeroChemistryWB2014

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
bins = 10
physics = WilliamsBest2014(radial_bins=bins, vertical_bins=bins)
chem = NonzeroChemistryWB2014(physics)
chem.run_chemistry()

physics.plot_density()
chem.plot_chemistry()
chem.plot_h2_coldens()

# chem.table.write_e()

plt.show()

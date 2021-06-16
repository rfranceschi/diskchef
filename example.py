import pytest
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt
import logging

import diskchef.chemistry.scikit
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.chemistry.williams_best_co import NonzeroChemistryWB2014

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
bins = 100
physics = WilliamsBest2014(radial_bins=bins, vertical_bins=bins)
chem = diskchef.chemistry.scikit.SciKitChemistry(physics)
chem.run_chemistry()

logging.getLogger('matplotlib.font_manager').disabled = True

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
physics.plot_density(axes=ax[0, 0])
physics.plot_temperatures(axes=ax[1, 0])

chem.plot_chemistry("CO", "HCO+", axes=ax[0, 1])
chem.plot_chemistry("CN", "HCN", axes=ax[1, 1])

chem.plot_absolute_chemistry("CO", "HCO+", axes=ax[0, 2])
chem.plot_absolute_chemistry("CN", "HCN", axes=ax[1, 2])


plt.show()

from matplotlib import pyplot as plt
import logging

import numpy as np
from astropy import units as u

import diskchef.chemistry.scikit
from diskchef.physics.williams_best import WilliamsBest2014

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
logging.getLogger('matplotlib.ticker').disabled = True

physics.plot_density()

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
physics.plot_density(axes=ax[0, 0])
tempplot = physics.plot_temperatures(axes=ax[1, 0])
tempplot.contours("Gas temperature", [20, 40] * u.mK * 1000)

chem.plot_chemistry("CO", "HCO+", axes=ax[0, 1])
chem.plot_chemistry("CN", "HCN", axes=ax[1, 1])

coplot = chem.plot_absolute_chemistry("CO", "HCO+", axes=ax[0, 2], maxdepth=1e9)
coplot.contours("Gas temperature", [20, 40] * u.mK * 1000, clabel_kwargs={"fmt": "T=%d K"})
chem.plot_absolute_chemistry("CN", "HCN", axes=ax[1, 2])

plt.show()
print(chem)

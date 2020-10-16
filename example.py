import pytest
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

from diskchef.physics.parametrized import powerlawpiece, PieceWisePowerLaw, PowerLawPhysics
from diskchef.physics.williams_best import WilliamsBest2014

physics = WilliamsBest2014()
# physics.plot_column_density()
physics.plot_density()
plt.show()
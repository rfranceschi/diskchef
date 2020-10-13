import pytest
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

from diskchef.physics.parametrized import powerlawpiece, PieceWisePowerLaw, PowerLawPhysics


physics = PowerLawPhysics()
# physics.plot_column_density()
physics.plot_density()
plt.show()
import pytest
from astropy import units as u
import numpy as np
from matplotlib import pyplot as plt

from diskchef.physics.parametrized import powerlawpiece, PieceWisePowerLaw, PowerLawPhysics
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.chemistry.williams_best_co import NonzeroChemistryWB2014

physics = WilliamsBest2014(radial_bins=5, vertical_bins=5)
chem = NonzeroChemistryWB2014(physics)
# physics.plot_column_density()
chem.run_chemistry()
# chem.table.pprint_all()
#
# physics.plot_density()
# chem.plot_chemistry()
# plt.show()

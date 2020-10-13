import pytest
from astropy import units as u
import numpy.testing

from diskchef.chemistry.base import ChemistryBase
from diskchef.physics.parametrized import PowerLawPhysics


def test_default_chemistry():
    chem = ChemistryBase(PowerLawPhysics())
    numpy.testing.assert_almost_equal(chem.table["n(H+2H2)"][:10].value,
                                      [3.74174141e+15, 2.95087606e+15, 2.32717032e+15, 1.83529284e+15,
                                       1.44738001e+15, 1.14145756e+15, 9.00195773e+14, 7.09927777e+14,
                                       5.59875378e+14, 4.41538491e+14])
    chem.run_chemistry()
    numpy.testing.assert_almost_equal(chem.table["CO"][:10], 0.5e-4)

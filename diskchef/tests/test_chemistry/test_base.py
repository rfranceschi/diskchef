import pytest
from astropy import units as u
import numpy.testing

from diskchef.chemistry.base import ChemistryBase
from diskchef.physics.parametrized import PowerLawPhysics


def test_default_chemistry():
    chem = ChemistryBase(PowerLawPhysics())
    numpy.testing.assert_allclose(chem.table["n(H+2H2)"][:10].value,
                                  [3.7417414e+15, 2.9508761e+15, 2.3271703e+15, 1.8352928e+15,
                                   1.4473800e+15, 1.1414576e+15, 9.0019577e+14, 7.0992778e+14,
                                   5.5987538e+14, 4.4153849e+14],
                                  )
    chem.run_chemistry()
    numpy.testing.assert_allclose(chem.table["CO"][:10], 0.5e-4)

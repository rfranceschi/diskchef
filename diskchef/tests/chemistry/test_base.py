import numpy.testing

from diskchef.chemistry.base import ChemistryModel
from diskchef.physics.parametrized import PowerLawPhysics


def test_default_chemistry():
    chem = ChemistryModel(PowerLawPhysics())
    numpy.testing.assert_allclose(chem.table["n(H+2H2)"][:10].value,
                                  [3.741741e+15, 3.714278e+15, 3.633090e+15, 3.501702e+15,
                                   3.325703e+15, 3.112353e+15, 2.870090e+15, 2.607974e+15,
                                   2.335137e+15, 2.060262e+15],
                                  rtol=1e-5)
    chem.run_chemistry()
    numpy.testing.assert_allclose(chem.table["CO"][:10], 0.5e-4)

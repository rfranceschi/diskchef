import numpy.testing

from diskchef.chemistry.base import ChemistryModel
from diskchef.physics.parametrized import PowerLawPhysics


def test_default_chemistry():
    chem = ChemistryModel(PowerLawPhysics())
    numpy.testing.assert_allclose(chem.table["n(H+2H2)"][:10].value,
                                  [6.396250e+15, 6.349303e+15, 6.210518e+15, 5.985919e+15,
                                   5.685060e+15, 5.320354e+15, 4.906221e+15, 4.458153e+15,
                                   3.991756e+15, 3.521877e+15],
                                  rtol=1e-5)
    chem.run_chemistry()
    numpy.testing.assert_allclose(chem.table["CO"][:10], 0.5e-4)

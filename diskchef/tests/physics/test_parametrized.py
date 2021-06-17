import sys
import numpy as np
import pytest
from astropy import units as u
from astropy.io.ascii import read
from astropy.table import setdiff

from diskchef.physics.parametrized import powerlawpiece, PieceWisePowerLaw, PowerLawPhysics


@pytest.mark.parametrize(
    "input",
    [
        [1 * u.au, 1 * u.s, 1, 1],
        [1 * u.s, 1 * u.au, 1, 1],
        [1 * u.s, 1 * u.s, 1, 1],
        [1 * u.Hz, 1 * u.au, 1, 1],
        [1, 1 * u.au, 1, 1],
        [1 * u.au, 1, 1, 1],
    ]
)
def test_powerlawpiece_wrongunits(input):
    with pytest.raises((u.core.UnitsError, TypeError)):
        powerlawpiece(*input)


@pytest.mark.parametrize(
    "r",
    [
        np.arange(6) * u.au,
        np.arange(6) * u.cm * 1.496e13,
    ]
)
@pytest.mark.parametrize(
    "input, expected",
    [
        [[[1 * u.au, 100 * u.au, 2, 1]], np.arange(6) ** 2 * u.dimensionless_unscaled],
        [[[1 * u.au, 100 * u.au, 2, 1], [1 * u.au, 100 * u.au, 2, 3]],
         4 * np.arange(6) ** 2 * u.dimensionless_unscaled],
        [[[1 * u.au, 100 * u.au, 2, 1], [1 * u.au, 2.1 * u.au, 1, 3]],
         np.array([0, 4, 10, 9, 16, 25]) * u.dimensionless_unscaled],
    ]
)
def test_piecewisepowerlaw(input, r, expected):
    assert u.allclose(PieceWisePowerLaw(input)(r), expected, rtol=1e-4)

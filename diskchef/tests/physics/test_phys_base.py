import pytest
from astropy import units as u

from diskchef.engine.exceptions import CHEFNotImplementedError
from diskchef.physics.base import PhysicsModel


@pytest.mark.parametrize(
    "input, error",
    [
        [(1, 1), TypeError],
        [(1 * u.au, 1 * u.au), CHEFNotImplementedError],
    ]
)
@pytest.mark.parametrize(
    "method",
    [
        "gas_temperature",
        "dust_temperature",
        "gas_density",
        "dust_density"
    ]
)
def test_raises(input, method, error):
    phys = PhysicsModel()
    pytest.raises(error, getattr(phys, method), *input)

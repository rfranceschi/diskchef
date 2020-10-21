import pytest
from astropy import units as u
from diskchef.physics.base import PhysicsBase
from diskchef.engine.exceptions import CHEFNotImplementedError


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
    phys = PhysicsBase()
    pytest.raises(error, getattr(phys, method), *input)

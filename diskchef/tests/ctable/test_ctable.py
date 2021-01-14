import numpy as np
import pytest
from astropy import units as u

from diskchef import CTable


@pytest.fixture
def table():
    return CTable(
        {
            "Radius": (1, 2, 3, 1, 2, 3) * u.au,
            "Height": (0, 0, 0, 1, 1, 1) * u.au,
            "Value": (1, 2, 3, 4, 5, 6) * u.g
        }
    )


def test_name(table):
    assert table["Value"].name == "Value"


def test_interpolation_edgepoint(table):
    value = table.interpolate("Value")
    assert value(1 * u.au, 0 * u.au) == 1 * u.g


def test_interpolation_scalar(table):
    value = table.interpolate("Value")
    pytest.approx(value(1 * u.au, 0.5 * u.au), 2.5 * u.g)


def test_interpolation_wrong_unit(table):
    value = table.interpolate("Value")
    pytest.approx(value(2e13 * u.cm, 0.5 * u.cm), 1.33691742 * u.g)


def test_interpolation_array(table):
    value = table.interpolate("Value")
    pytest.approx(
        value(np.linspace(1, 2, 4) << u.au, np.linspace(0, 1, 4) << u.au),
        [1., 2.3333333333333333333, 3.666666666666666666667, 5.] * u.g
    )

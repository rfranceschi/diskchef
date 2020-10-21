import sys
import pytest
from astropy import units as u
from astropy.table import setdiff
import numpy as np
from matplotlib import pyplot as plt
from astropy.io.ascii import read
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


def test_default_setup():
    physics = PowerLawPhysics(radial_bins=3, vertical_bins=3)
    # physics.table.write(format="ascii.ecsv", filename=sys.stdout)
    expected = """
# %ECSV 0.9
# ---
# datatype:
# - {name: Radius, unit: AU, datatype: float64}
# - {name: Height, unit: AU, datatype: float64}
# - {name: Height to radius, datatype: float64}
# - {name: Gas density, unit: g / cm3, datatype: float64}
# - {name: Dust density, unit: g / cm3, datatype: float64}
# - {name: Gas temperature, unit: K, datatype: float64}
# - {name: Dust temperature, unit: K, datatype: float64}
# meta: !!omap
# - __serialized_columns__:
#     Dust density:
#       __class__: astropy.units.quantity.Quantity
#       unit: !astropy.units.Unit {unit: g / cm3}
#       value: !astropy.table.SerializedColumn {name: Dust density}
#     Dust temperature:
#       __class__: astropy.units.quantity.Quantity
#       unit: &id001 !astropy.units.Unit {unit: K}
#       value: !astropy.table.SerializedColumn {name: Dust temperature}
#     Gas density:
#       __class__: astropy.units.quantity.Quantity
#       unit: !astropy.units.Unit {unit: g / cm3}
#       value: !astropy.table.SerializedColumn {name: Gas density}
#     Gas temperature:
#       __class__: astropy.units.quantity.Quantity
#       unit: *id001
#       value: !astropy.table.SerializedColumn {name: Gas temperature}
#     Height:
#       __class__: astropy.units.quantity.Quantity
#       unit: &id002 !astropy.units.Unit {unit: AU}
#       value: !astropy.table.SerializedColumn {name: Height}
#     Radius:
#       __class__: astropy.units.quantity.Quantity
#       unit: *id002
#       value: !astropy.table.SerializedColumn {name: Radius}
# schema: astropy-2.0
Radius Height "Height to radius" "Gas density" "Dust density" "Gas temperature" "Dust temperature"
0.1 0.0 0.0 1.4477007137260397e-08 1.4477007137260398e-10 9999.999999999998 9999.999999999998
0.1 0.034999999999999996 0.35 2.095818711618038e-16 2.095818711618038e-18 9999.999999999998 9999.999999999998
0.1 0.06999999999999999 0.7 6.358848650216633e-40 6.358848650216633e-42 9999.999999999998 9999.999999999998
7.071067811865475 0.0 0.0 1.1378897425827754e-13 1.1378897425827755e-15 2.0000000000000004 2.0000000000000004
7.071067811865475 2.474873734152916 0.35 1.5847357750038788e-14 1.5847357750038788e-16 2.0000000000000004 2.0000000000000004
7.071067811865475 4.949747468305832 0.7 4.2808193485924884e-17 4.2808193485924885e-19 2.0000000000000004 2.0000000000000004
499.99999999999994 0.0 0.0 8.943789652093236e-19 8.943789652093236e-21 0.0004000000000000001 0.0004000000000000001
499.99999999999994 174.99999999999997 0.35 7.211423548391477e-19 7.211423548391478e-21 0.0004000000000000001 0.0004000000000000001
499.99999999999994 349.99999999999994 0.7 3.7802392342957524e-19 3.7802392342957524e-21 0.0004000000000000001 0.0004000000000000001"""
    tbl = read(expected)
    assert not setdiff(tbl, physics.table)

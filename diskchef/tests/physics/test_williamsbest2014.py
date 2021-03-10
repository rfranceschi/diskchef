from astropy.io.ascii import read
from astropy.table import setdiff
import sys
from diskchef.physics.williams_best import WilliamsBest2014


def test_default_setup():
    physics = WilliamsBest2014(radial_bins=3, vertical_bins=3)
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
0.1 0.0 0.0 1.153290417008343e-15 1.153290417008343e-17 709.626778467151 709.626778467151
0.1 0.034999999999999996 0.35 5.0245869970310865e-34 5.024586997031087e-36 3548.1338923357553 3548.1338923357553
0.1 0.06999999999999999 0.7 5.9212684702974e-72 5.9212684702974e-74 3548.1338923357553 3548.1338923357553
7.071067811865475 0.0 0.0 2.2851681797625917e-13 2.285168179762592e-15 68.20453375973065 68.20453375973065
7.071067811865475 2.474873734152916 0.35 2.716385841960013e-17 2.716385841960013e-19 341.0226687986533 341.0226687986533
7.071067811865475 4.949747468305832 0.7 7.189026042423895e-23 7.189026042423895e-25 341.0226687986533 341.0226687986533
499.99999999999994 0.0 0.0 2.871710463998362e-20 2.871710463998362e-22 6.55535919237802 6.55535919237802
499.99999999999994 174.99999999999997 0.35 6.929035936387884e-22 6.929035936387884e-24 26.250827491444678 26.250827491444678
499.99999999999994 349.99999999999994 0.7 8.170463977902202e-23 8.170463977902203e-25 32.7767959618901 32.7767959618901"""
    tbl = read(expected)
    assert not setdiff(tbl, physics.table)

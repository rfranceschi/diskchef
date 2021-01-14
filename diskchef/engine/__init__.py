"""
Package with basic objects used in the whole project

Important:

`diskchef.engine.CTable` -- `astropy.table.QTable`-like object, with extra features for handling `diskchef` data

`diskchef.engine.CHEFNotImplementedError` -- exception raised by base classes methods which only define interface for child classes

`diskchef.engine.CHEFSlowDownWarning` -- warning issued by pieces of code which are known to be slow, if a possible alternative exists
"""
from diskchef.engine.ctable import CTable
from diskchef.engine.exceptions import CHEFSlowDownWarning, CHEFNotImplementedError

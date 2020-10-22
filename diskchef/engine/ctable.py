"""Class CTable(astropy.table.QTable) with additional features for CheF"""
import sys
from typing import Callable
from functools import cached_property
from collections import UserDict

import numpy as np
from astropy.table import QTable
from astropy import units as u
from astropy.io.ascii import write
from named_constants import Constants
from scipy.interpolate import griddata


from diskchef.engine.exceptions import CHEFNotImplementedError

class TableColumns(Constants):
    radius = 'Radius'
    height = 'Height'

class DictLikeDefault(UserDict):
    def __init__(self, default=None):
        super(DictLikeDefault, self).__init__()
        self.default = default

    def __getitem__(self, item):
        try:
            super().__getitem__(item)
        except KeyError:
            return self.default


class CTable(QTable):
    """
    Subclass of astropy.table.Qtable for DiskCheF

    Features:
        puts `name` attribute to the __getitem__ output
        returns appropriate columns with `r` and `z` properties
        provides `interpolate` method that returns a `Callable(r,z)`

    """

    def __getitem__(self, item):
        column_quantity = super().__getitem__(item)
        column_quantity.name = item
        return column_quantity

    @property
    def r(self):
        """Column with radius coordinate"""
        return self['Radius']

    @property
    def z(self):
        """Column with height coordinate"""
        return self['Height']

    @property
    def zr(self):
        return self['Height to radius']

    def interpolate(self, column: str) -> Callable[[u.Quantity, u.Quantity], u.Quantity]:
        """
        Interpolate the selected quantity
        Args:
            column: str -- column name of the table to interpolate

        Returns: callable(r, z) with interpolated value of column
        """

        def _interpolation(r: u.au, z: u.au):
            interpolated = griddata(
                points=(self.r, self.z),
                values=self[column],
                xi=(r.to(u.au).value, z.to(u.au).value),
                fill_value=0
            ) << self[column].unit
            return interpolated

        return _interpolation

    @cached_property
    def is_in_zr_regular_grid(self) -> bool:
        """
        Returns: whether the table has same number of relative heights
        """
        zr_set = set(self.zr)
        lengths = [len(np.argwhere(self.zr == zr_set))]
        return len(set(lengths)) == 1

    def add_row(self, vals=None, mask=None):
        raise CHEFNotImplementedError("Adding rows (grid points) is not possible in CTable")

    def write_e(self, file=sys.stdout, format="fixed_width", **kwargs):
        write(self, file, formats={column: "%e" for column in self.colnames}, format=format, **kwargs)


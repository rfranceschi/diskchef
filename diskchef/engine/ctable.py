"""Class CTable(astropy.table.QTable) with additional features for CheF"""
import sys
from typing import Callable
from functools import cached_property
import io
from contextlib import redirect_stdout

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


class CTable(QTable):
    """
    Subclass of astropy.table.Qtable for DiskCheF

    Features:
        puts `name` attribute to the __getitem__ output
        returns appropriate columns with `r` and `z` properties
        provides `interpolate` method that returns a `Callable(r,z)`


    Usage:
    >>> #  __repr__() call sets formats to "e"
    >>> tbl = CTable()
    >>> tbl['Radius'] = [1, 2] * u.m; tbl['b'] = [3e-4, 4e3]
    >>> tbl # doctest: +NORMALIZE_WHITESPACE
       Radius         b
          m
    ------------ ------------
    1.000000e+00 3.000000e-04
    2.000000e+00 4.000000e+03
    >>> tbl.r
    <Quantity [1., 2.] m>
    >>> # .name attribute is properly set for the returned Quantity
    >>> tbl['b'].name
    'b'
    >>> tbl.r.name
    'Radius'
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

    # def write_e(self, file=None, format="fixed_width", print_in_the_end: bool = True, **kwargs):
    #     """
    #     Write CTable with exponential format for each column
    #
    #     Args:
    #         file: file for the output
    #         format: astropy.io format
    #         print_in_the_end: if True and file is None, print the output
    #         **kwargs: to be passed to astropy.io.ascii.write()
    #
    #     Returns: if not print_in_the_end, returns formatted table as string, otherwise returns None
    #
    #     Usage:
    #     >>> tbl = CTable()
    #     >>> tbl['a'] = [1, 2]; tbl['b'] = [3e-4, 4e3] * u.m
    #     >>> tbl # doctest: +NORMALIZE_WHITESPACE
    #          a            b
    #                       m
    #     ------------ ------------
    #     1.000000e+00 3.000000e-04
    #     2.000000e+00 4.000000e+03
    #     """
    #     print_in_the_end = False
    #     if file is None:
    #         file = io.StringIO()
    #         print_in_the_end = True
    #     write(self, file, formats={column: "%e" for column in self.colnames}, format=format, **kwargs)
    #     if print_in_the_end:
    #         print(file.getvalue(), end='')

    def __repr__(self):
        for column in self.colnames:
            self[column].info.format = "e"
        with io.StringIO() as buf, redirect_stdout(buf):
            self.pprint_all()
            output = buf.getvalue()
        return output

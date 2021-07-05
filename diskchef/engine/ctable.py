"""Class CTable(astropy.table.QTable) with additional features for CheF"""
import io
from contextlib import redirect_stdout
from functools import cached_property
from typing import Callable

import numpy as np
from astropy import units as u
from astropy.table import QTable
from scipy.interpolate import griddata

from diskchef.engine.exceptions import CHEFNotImplementedError, CHEFRuntimeError


class CTable(QTable):
    """
    Subclass of astropy.table.Qtable for DiskCheF

    Features:

        puts `name` attribute to the `__getitem__` output

        returns appropriate columns with `r` and `z` properties

        provides `interpolate` method that returns a `Callable(r,z)`

        __repr__() call sets formats to "e"

    Usage:

    >>> tbl = CTable()
    >>> tbl['Radius'] = [1, 2] * u.m; tbl['Data'] = [3e-4, 4e3]
    >>> tbl # doctest: +NORMALIZE_WHITESPACE
       Radius        Data
          m
    ------------ ------------
    1.000000e+00 3.000000e-04
    2.000000e+00 4.000000e+03
    >>> # Radius, Height, and some other keywords are achievable with r, z, and other respective properties
    >>> tbl.r
    <Quantity [1., 2.] m>

    >>> # .name attribute is properly set for the returned Quantity
    >>> tbl['Data'].name
    'Data'
    >>> tbl.r.name
    'Radius'

    >>> # Adding rows is not possible
    >>> tbl.add_row([1*u.cm, 10])
    Traceback (most recent call last):
       ...
    diskchef.engine.exceptions.CHEFNotImplementedError: Adding rows (grid points) is not possible in CTable

    >>> # Interpolation
    >>> tbl = CTable()
    >>> tbl["Radius"] = [1, 2, 3, 1, 2, 3] * u.au
    >>> tbl["Height"] = [0, 0, 0, 1, 1, 1] * u.au
    >>> tbl["Data"] = [2, 4, 6, 3, 5, 7] * u.K
    >>> tbl  # doctest: +NORMALIZE_WHITESPACE
       Radius       Height        Data
         AU           AU           K
    ------------ ------------ ------------
    1.000000e+00 0.000000e+00 2.000000e+00
    2.000000e+00 0.000000e+00 4.000000e+00
    3.000000e+00 0.000000e+00 6.000000e+00
    1.000000e+00 1.000000e+00 3.000000e+00
    2.000000e+00 1.000000e+00 5.000000e+00
    3.000000e+00 1.000000e+00 7.000000e+00
    >>> tbl.interpolate("Data")(1.5 * u.au, 0 * u.au)
    <Quantity 3. K>
    >>> tbl.interpolate("Data")([1, 2.5] * u.au, [0.2, 0.8] * u.au)
    <Quantity [2.2, 5.8] K>
    >>> # Compatible units are allowed
    >>> tbl.interpolate("Data")(1.5e8 * u.km, [0.2, 0.8] * u.au)
    <Quantity [2.20537614, 2.80537614] K>
    """

    def __getitem__(self, item):
        try:
            column_quantity = super().__getitem__(item)
        except KeyError as e:
            if " number density" in item:
                column_quantity = self[item[:-len(" number density")]] * self["n(H+2H2)"]
            else:
                raise e
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

        # TODO non-linear interpolation
        def _interpolation(r: u.au, z: u.au):
            interpolated = griddata(
                points=(self.r.to(u.au).value, self.z.to(u.au).value),
                values=self[column],
                xi=(r.to(u.au).value, z.to(u.au).value),
                fill_value=0
            )
            if self[column].unit:
                interpolated = interpolated << self[column].unit
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
        """
        Adding rows (grid points) is not possible in CTable

        Raises: CHEFNotImplementedError
        """
        raise CHEFNotImplementedError("Adding rows (grid points) is not possible in CTable")

    def __repr__(self):
        for column in self.colnames:
            self[column].info.format = "e"
        with io.StringIO() as buf, redirect_stdout(buf):
            self.pprint_all()
            output = buf.getvalue()
        return output

    def __str__(self):
        return self.__repr__()

    @property
    def _dust_pop_sum(self):
        return sum([self[f"{dust.name} mass fraction"] for dust in self.meta["Dust list"]])

    @property
    def dust_population_fully_set(self, atol=1e-5):
        """
        Check whether sum of mass fractions of dust populations is equal to 1
        """
        return np.all(abs(1 - self._dust_pop_sum) < atol)

    def normalize_dust(self):
        """
        Normalize the dust fractions so that the sum is 1
        """
        pop_sum = self._dust_pop_sum
        for dust in self.meta["Dust list"]:
            dust.mass_fraction /= pop_sum
            dust.write_to_table()
        if not self.dust_population_fully_set:
            raise CHEFRuntimeError

    def column_density(self, colname: str, r: u.au = None) -> u.Quantity:
        """
        Calculate column density of `colname` on `r` grid
        Args:
            colname: name of the column to collapse
            r: grid (centers) for the output. If not specified, defaults to `sorted(set(self.r))`

        Returns:
            column density of `self[colname]` on `r` grid
        """
        if r is None:
            r = sorted(set(self.r))

        if self.is_in_zr_regular_grid:
            r_grid = sorted(set(self.r))
            coldenses = []
            for _r in r_grid:
                indices = self.r == _r
                value = self[colname][indices]
                z = self.z[indices]
                coldenses.append(np.trapz(value, z))
            coldenses = u.Quantity(coldenses)
            return np.interp(r, r_grid, coldenses)
        else:
            raise NotImplementedError("Column density is currently only implemented for zr grids")


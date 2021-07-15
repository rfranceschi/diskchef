import os
from dataclasses import dataclass
from functools import cached_property
import logging
import warnings
from typing import Union

import diskchef.physics.ionization
import scipy.integrate
from astropy import units as u
from astropy import constants
from astropy.visualization import quantity_support

import matplotlib.axes
import matplotlib.colors
import numpy as np

from diskchef import CTable
from diskchef.engine.exceptions import CHEFNotImplementedError, CHEFSlowDownWarning
from diskchef.engine.plot import Plot2D, Plot1D

quantity_support()


@dataclass
class PhysicsBase:
    """Base class for disk physics models"""
    star_mass: u.solMass = 1 * u.solMass
    xray_plasma_temperature: u.K = 1e7 * u.K
    xray_luminosity: u.erg / u.s = 1e31 * u.erg / u.s

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

    @property
    def table(self) -> CTable:
        """Storage for all the position-dependent data of the disk"""
        return self._table

    @table.setter
    def table(self, value: CTable):
        self._table = value

    def plot_density(
            self,
            axes: matplotlib.axes.Axes = None,
            table: CTable = None, folder=".",
            cmap: Union[matplotlib.colors.Colormap, str] = 'PuBuGn',
            **kwargs
    ) -> Plot2D:
        """Plot 2D Gas and Dust density figures"""
        if table is None:
            table = self.table
        return Plot2D(table, axes=axes, data1="Gas density", data2="Dust density", cmap=cmap, **kwargs)

    def plot_temperatures(
            self,
            axes: matplotlib.axes.Axes = None,
            table: CTable = None, folder=".",
            cmap: Union[matplotlib.colors.Colormap, str] = 'afmhot',
            **kwargs
    ) -> Plot2D:
        """Plot 2D Gas and Dust temperature figures"""
        if table is None:
            self.check_temperatures()
            table = self.table
        return Plot2D(table, axes=axes, data1="Gas temperature", data2="Dust temperature", cmap=cmap, **kwargs)

    def plot_column_densities(
            self,
            axes: matplotlib.axes.Axes = None,
            table: CTable = None, folder=".",
            **kwargs
    ) -> Plot1D:
        """Plot 1D column plots of Gas and Dust density. Use the code of this method as an example."""
        if table is None:
            table = self.table
        return Plot1D(table, axes=axes, data=["Gas density", "Dust density"], **kwargs)

    def check_temperatures(self):
        self.table.check_zeros("Gas temperature")
        self.table.check_zeros("Dust temperature")

    @u.quantity_input
    def column_density_to(
            self, r: u.au, z: u.au,
            colname,
            r0: u.au = 0 * u.au, z0: u.au = 0 * u.au,
            steps=500, steps_for_log_fraction=0.9,
            only_gridpoint=False,
            point_by_point=False
    ):
        """
        Calculates column density of the given CTable quantity towards the `(r0, z0)`

        Args:
            r: radial coordinate of the desired point(s)
            z: vertical coordinate of the desired point(s)
            colname: name of self.table column to integrate
            r0: coordinate of the initial point (0 by default)
            z0: coordinate of the initial point (0 by default)
            steps: number of steps
            steps_for_log_fraction: fraction of steps to be taken in log scale
            only_gridpoint: when calculating column density towards star, whether to interpolate, or only take grid points
            point_by_point: force separate integration for each point if True
        Returns:
            column density, in self.table[colname].unit * u.cm
        """
        steps_for_log = int(steps_for_log_fraction * steps)
        steps_for_lin = steps - steps_for_log
        klog = np.geomspace(1 / steps_for_log, 1, steps_for_log)
        klin = np.linspace(0, 1 / steps_for_log, steps_for_lin, endpoint=False)
        k = np.concatenate([klin, klog])
        integrals = np.empty_like(self.table.r.value) << (u.cm * self.table[colname].unit)

        if (self.table.is_in_zr_regular_grid and r0 == z0 == 0 * u.au) and not point_by_point:
            zr_set = set(self.table.zr)
            for _zr in zr_set:
                this_zr_rows = np.where(self.table.zr == _zr)[0]
                int_r = r0 + (np.max(self.table.r[this_zr_rows]) - r0) * k
                int_z = z0 + (np.max(self.table.z[this_zr_rows]) - z0) * k
                if not only_gridpoint:
                    # Mix the grid elements into the int_r array
                    int_r = u.Quantity([*int_r, *self.table.r[this_zr_rows]])
                    int_z = u.Quantity([*int_z, *self.table.z[this_zr_rows]])
                    # The position of the grid elemenets in new int_r array
                    idx = np.where(int_r.argsort() - len(k) >= 0)[0]
                    int_r.sort()
                    int_z.sort()
                else:
                    idx = np.argsort(self.table.r[this_zr_rows])
                    int_r = self.table.r[this_zr_rows][idx]
                    int_z = self.table.z[this_zr_rows][idx]
                k = (int_r - r0) / (np.max(self.table.r[this_zr_rows]) - r0)
                k_length_cm = (
                        ((int_r[-1] - int_r[0]) ** 2 + (int_z[-1] - int_z[0]) ** 2) ** 0.5
                ).to(u.cm)
                value = self.table.interpolate(colname)(int_r, int_z)
                integrals_at_zr = scipy.integrate.cumtrapz(value.value, k, initial=0) * value.unit * k_length_cm
                integrals[this_zr_rows] = integrals_at_zr[idx]
        elif (self.table.is_in_zr_regular_grid and z0 == np.inf * u.au) and not point_by_point:
            radii = set(self.table.r)
            for _r in radii:
                indices = self.table.r == _r
                _z = self.table.z[indices]
                _data = self.table[colname][indices]
                order = np.argsort(_z)[::-1]
                integrals[indices] = np.abs(scipy.integrate.cumtrapz(
                    _data.value[order],
                    _z.value[order],
                    initial=0
                ))[np.argsort(order)] * _data.unit * _z.unit
        else:
            warnings.warn(CHEFSlowDownWarning(
                "Column density calculations are expensive, unless the grid is regular in z/r AND r0 == z0 == 0"
            ))
            for i, (_r, _z) in enumerate(zip(self.table.r, self.table.z)):
                int_r = r0 + (_r - r0) * k
                int_z = z0 + (_z - z0) * k
                k_length_cm = (
                        ((int_r[-1] - int_r[0]) ** 2 + (int_z[-1] - int_z[0]) ** 2) ** 0.5
                ).to(u.cm)
                value = self.table.interpolate(colname)(int_r, int_z)
                integral = scipy.integrate.trapz(value, k) * k_length_cm
                integrals[i] = integral
        return u.Quantity(integrals)

    def xray_bruderer(self):
        """Calculate X-ray ionization rates using Bruderer+09 Table 3

        Requires `xray_plasma_temperature` and `xray_luminosity` to be set
        """
        if "Total density" not in self.table.colnames:
            self.table["Total density"] = self.table["Gas density"] + self.table["Dust density"]
        self.table["Nucleon column density towards star"] = (self.column_density_to(
            self.table.r, self.table.z,
            f"Total density",
            only_gridpoint=True,
        ) / u.u).to(u.cm ** -2)

        self.table["X ray ionization rate"] = (
                diskchef.physics.ionization.bruderer09(
                    self.table["Nucleon column density towards star"].to_value(u.cm ** -2),
                    self.xray_plasma_temperature.to_value(u.K)
                ) / u.s * (
                        self.xray_luminosity / (4 * np.pi * self.table.r ** 2)
                ).to_value(u.erg / u.s / u.cm ** 2)
        ).to(1 / u.s)

    def cosmic_ray_padovani18(self):
        """Calculate CR ionization rate according to App. F model L of Padovani+2018

        https://www.aanda.org/articles/aa/pdf/2018/06/aa32202-17.pdf
        """
        if "Total density" not in self.table.colnames:
            self.table["Total density"] = self.table["Gas density"] + self.table["Dust density"]

        if "Nucleon column density upwards" not in self.table.colnames:
            self.table["Nucleon column density upwards"] = (self.column_density_to(
                np.nan * u.au, np.nan * u.au,
                f"Total density",
                r0=np.nan * u.au, z0=np.inf * u.au,
                only_gridpoint=True,
            ) / u.u).to(u.cm ** -2)

        density_upwards = self.table["Nucleon column density upwards"].to_value(u.cm ** -2)

        midplane_i = self.table.zr == 0
        midplane_coldens = self.table["Nucleon column density upwards"][midplane_i]
        midplane_r = self.table.r[midplane_i]
        midplane_dict = dict(zip(midplane_r, midplane_coldens))
        coldens = u.Quantity([midplane_dict[r] for r in self.table.r]).to_value(u.cm ** -2)

        self.table["CR ionization rate"] = 0.5 * (
                diskchef.physics.ionization.padovani18l(np.log10(density_upwards)) +
                diskchef.physics.ionization.padovani18l(
                    np.log10(2 * coldens - density_upwards))
        ) / u.s


@dataclass
class PhysicsModel(PhysicsBase):
    """
    The base class describing the most basic parameters of the disk

    Can not be used directly, rather subclasses should be used. See `WilliamsBest2014` documentation for more details
    """
    r_min: u.au = 0.1 * u.au
    r_max: u.au = 500 * u.au
    zr_max: float = 0.7
    radial_bins: int = 100
    vertical_bins: int = 100
    dust_to_gas: float = 0.01

    @u.quantity_input
    def gas_density(self, r: u.au, z: u.au) -> u.g / u.cm ** 3:
        """Calculates gas density at given r, z"""
        raise CHEFNotImplementedError

    @u.quantity_input
    def dust_temperature(self, r: u.au, z: u.au) -> u.K:
        """Calculates dust temperature at given r, z"""
        raise CHEFNotImplementedError

    @u.quantity_input
    def dust_density(self, r: u.au, z: u.au) -> u.g / u.cm ** 3:
        """Calculates dust density at given r, z"""
        return self.gas_density(r, z) * self.dust_to_gas

    @u.quantity_input
    def gas_temperature(self, r: u.au, z: u.au) -> u.K:
        """Calculates gas temperature at given r, z

        Returns:
            dust temperature
        """
        return self.dust_temperature(r, z)

    @cached_property
    def table(self) -> CTable:
        """Return the associated diskchef.CTable with r, z, and dust and gas properties"""
        z2r, r = np.meshgrid(
            np.linspace(0, self.zr_max, self.vertical_bins),
            np.geomspace(self.r_min.to(u.au).value, self.r_max.to(u.au).value, self.radial_bins),

        )
        table = CTable(
            [
                (r * u.au).flatten(),
                (r * z2r * u.au).flatten()
            ],
            names=["Radius", "Height"]
        )
        table["Height to radius"] = u.Quantity(z2r.flatten())
        table["Gas density"] = self.gas_density(table.r, table.z)
        table["Dust density"] = self.dust_density(table.r, table.z)
        table["Gas temperature"] = self.gas_temperature(table.r, table.z)
        table["Dust temperature"] = self.dust_temperature(table.r, table.z)
        return table

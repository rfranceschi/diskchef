from dataclasses import dataclass
from functools import cached_property
import logging
import warnings

import scipy.integrate
from astropy import units as u
from astropy.visualization import quantity_support

import matplotlib.axes
import numpy as np

from divan import Divan

from diskchef import CTable
from diskchef.engine.exceptions import CHEFNotImplementedError, CHEFSlowDownWarning

quantity_support()


@dataclass
class PhysicsBase:
    star_mass: u.solMass = 1 * u.solMass

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

    @property
    def table(self) -> CTable:
        return self._table

    @table.setter
    def table(self, value: CTable):
        self._table = value


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

    @u.quantity_input(r=u.au, z=u.au)
    def dust_temperature(self, r, z) -> u.K:
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
        table["Height to radius"] = z2r.flatten()
        table["Gas density"] = self.gas_density(table.r, table.z)
        table["Dust density"] = self.dust_density(table.r, table.z)
        table["Gas temperature"] = self.gas_temperature(table.r, table.z)
        table["Dust temperature"] = self.dust_temperature(table.r, table.z)
        return table

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
        integrals = np.empty_like(r.value) << (u.cm * self.table[colname].unit)
        if not (self.table.is_in_zr_regular_grid and r0 == z0 == 0 * u.au) or point_by_point:
            warnings.warn(CHEFSlowDownWarning(
                "Column density calculations are expensive, unless the grid is regular in z/r AND r0 == z0 == 0"
            ))
            for i, (_r, _z) in enumerate(zip(r, z)):
                int_r = r0 + (_r - r0) * k
                int_z = z0 + (_z - z0) * k
                k_length_cm = (
                        ((int_r[-1] - int_r[0]) ** 2 + (int_z[-1] - int_z[0]) ** 2) ** 0.5
                ).to(u.cm)
                value = self.table.interpolate(colname)(int_r, int_z)
                integral = scipy.integrate.trapz(value, k) * k_length_cm
                integrals[i] = integral
        else:
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

        return u.Quantity(integrals)

    def plot_column_density(self, axes=None, table=None):
        raise CHEFNotImplementedError

    def plot_density(self, axes: matplotlib.axes.Axes = None, table: CTable = None):
        if table is None:
            table = self.table
        # if axes is None:
        #     fig, axes = plt.subplots()
        dvn = Divan()
        dvn.physical_structure = table
        dvn.generate_figure_volume_densities(extra_gas_to_dust=100)
        dvn.generate_figure_temperatures()

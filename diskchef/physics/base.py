from dataclasses import dataclass
from functools import cached_property

import scipy.integrate
from astropy import units as u
from astropy.visualization import quantity_support

quantity_support()
import matplotlib.axes
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
import numpy as np

from divan import Divan

from diskchef import CTable
from diskchef.engine.exceptions import CHEFNotImplementedError


@dataclass
class PhysicsBase:
    """The base class describing the most basic parameters of the disk"""
    star_mass: u.solMass = 1 * u.solMass
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
            steps=500, steps_for_log_fraction=0.9
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

        Returns:
            column density, in self.table[colname].unit * u.cm
        """
        steps_for_log = int(steps_for_log_fraction * steps)
        steps_for_lin = steps - steps_for_log
        klog = np.geomspace(1 / steps_for_log, 1, steps_for_log)
        klin = np.linspace(0, 1 / steps_for_log, steps_for_lin, endpoint=False)
        k = np.concatenate([klin, klog])

        integrals = []
        for _r, _z in zip(r, z):
            int_r = r0 + (_r - r0) * k
            int_z = z0 + (_z - z0) * k
            k_length_cm = (
                    ((int_r[-1] - int_r[0]) ** 2 + (int_z[-1] - int_z[0]) ** 2) ** 0.5
            ).to(u.cm)
            value = self.table.interpolate(colname)(int_r, int_z)
            integral = scipy.integrate.trapz(value, k) * k_length_cm
            integrals.append(integral)

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


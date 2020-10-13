from dataclasses import dataclass
from functools import cached_property

from astropy import units as u
from astropy.visualization import quantity_support

quantity_support()
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
import numpy as np

from divan import Divan

from diskchef import CTable
from diskchef.engine.exceptions import CHEFNotImplemented


@dataclass
class PhysicsBase:
    """The base class describing the most basic parameters of the disk"""
    star_mass: u.solMass = 1 * u.solMass
    r_min: u.au = 0.1 * u.au
    r_max: u.au = 500 * u.au
    zr_max: float = 0.7
    radial_bins = 100
    vertical_bins = 100
    dust_to_gas: float = 0.01

    @u.quantity_input(r=u.au, z=u.au)
    def gas_density(self, r, z) -> u.g / u.cm ** 3:
        """Calculates gas density at given r, z"""
        raise CHEFNotImplemented

    @u.quantity_input(r=u.au, z=u.au)
    def dust_temperature(self, r, z) -> u.K:
        """Calculates dust temperature at given r, z"""
        raise CHEFNotImplemented

    @u.quantity_input(r=u.au, z=u.au)
    def dust_density(self, r, z) -> u.g / u.cm ** 3:
        """Calculates dust density at given r, z"""
        return self.gas_density(r, z) * self.dust_to_gas

    @u.quantity_input(r=u.au, z=u.au)
    def gas_temperature(self, r, z) -> u.K:
        """Calculates gas temperature at given r, z

        Returns:
            dust temperature
        """
        return self.dust_temperature(r, z)

    @cached_property
    def table(self) -> CTable:
        """Return the associated diskchef.CTable with r, z, and dust and gas properties"""
        r, z2r = np.meshgrid(
            np.geomspace(self.r_min.to(u.au).value, self.r_max.to(u.au).value, self.radial_bins),
            np.linspace(0, self.zr_max, self.vertical_bins)
        )
        table = CTable(
            [
                (r * u.au).flatten(),
                (r * z2r * u.au).flatten()
            ],
            names=["Radius", "Height"]
        )
        table["Height to radius"] = z2r.flatten()
        table["Gas density"] = self.gas_density(table["Radius"], table["Height"])
        table["Dust density"] = self.dust_density(table["Radius"], table["Height"])
        table["Gas temperature"] = self.gas_temperature(table["Radius"], table["Height"])
        table["Dust temperature"] = self.dust_temperature(table["Radius"], table["Height"])
        return table

    def plot_column_density(self, axes=None, table=None):
        raise CHEFNotImplemented
        # if axes is None:
        #     fig, axes = plt.subplots()
        # if table is None:
        #     table = self.table
        # axes.semilogx(table["Radius"], table["Height to radius"], "r*")

    def plot_density(self, axes=None, table=None):
        if table is None:
            table = self.table
        # if axes is None:
        #     fig, axes = plt.subplots()
        dvn = Divan()
        dvn.physical_structure = table
        dvn.generate_figure_volume_densities(extra_gas_to_dust=100)
        dvn.generate_figure_temperatures()

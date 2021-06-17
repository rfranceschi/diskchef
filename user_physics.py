"""Example of a user-written Physics subclass"""

from diskchef.physics import PhysicsModel
from dataclasses import dataclass
from astropy import units as u
import numpy as np

from astropy import constants as c


@dataclass
class MyIsothermalPhysics(PhysicsModel):
    """Example of a user-written Physics subclass

    In PhysicsModel, if `dust_density` is not defined,
    it is assumed equal to `gas_density * self.gas_to_dust`,
    and `gas_temperature` is assumed to be equal to
    `dust_temperature`
    TODO There is a unit mismatch somewhere here

    Usage:
    Initialize the model, but do not calculate anything yet
    >>>  mydisk = MyIsothermalPhysics()

    Calcualtions happen when `table` property is first accessed
    >>> print(mydisk.table)
    """
    temperature: u.K = 100 * u.K
    density_1au: u.g / u.cm ** 2 = 1 * u.g / u.cm ** 2
    slope: float = -2

    @u.quantity_input
    def dust_temperature(self, r: u.au, z: u.au) -> u.K:
        return self.temperature

    @u.quantity_input
    def scale_height(self, r: u.au) -> u.au:
        return np.sqrt(5 / 3 * c.k_B * self.gas_temperature(r, 0 * u.au) /
                       (c.G * self.star_mass / r ** 3))

    @u.quantity_input
    def gas_density(self, r: u.au, z: u.au) -> u.g / u.cm ** 3:
        return self.density_1au / (
                2 * np.pi * self.scale_height(r)
        ) * np.exp(self.slope * r / (1 * u.au)) * np.exp(
            -z ** 2 / 2 / self.scale_height(r) ** 2)




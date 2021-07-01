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

    Usage:
    Initialize the model, but do not calculate anything yet
    >>> mydisk = MyIsothermalPhysics()

    Calcualtions happen when `table` property is first accessed
    >>> print(mydisk.table[:5])  # doctest: +NORMALIZE_WHITESPACE
           Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature
         AU           AU                         g / cm3      g / cm3           K               K
    ------------ ------------ ---------------- ------------ ------------ --------------- ----------------
    1.000000e-01 0.000000e+00     0.000000e+00 2.666585e-11 2.666585e-13    1.000000e+02     1.000000e+02
    1.000000e-01 7.070707e-04     7.070707e-03 1.836627e-11 1.836627e-13    1.000000e+02     1.000000e+02
    1.000000e-01 1.414141e-03     1.414141e-02 6.000917e-12 6.000917e-14    1.000000e+02     1.000000e+02
    1.000000e-01 2.121212e-03     2.121212e-02 9.301338e-13 9.301338e-15    1.000000e+02     1.000000e+02
    1.000000e-01 2.828283e-03     2.828283e-02 6.839185e-14 6.839185e-16    1.000000e+02     1.000000e+02
    """
    temperature: u.K = 100 * u.K
    density_1au: u.g / u.cm ** 2 = 1 * u.g / u.cm ** 2
    slope: float = -2
    gamma: float = 5. / 3.
    mean_molecular_mass: u.g / u.mol = 2.33 * u.g / u.mol

    @u.quantity_input
    def dust_temperature(self, r: u.au, z: u.au) -> u.K:
        return self.temperature

    @u.quantity_input
    def scale_height(self, r: u.au) -> u.au:
        return (np.sqrt(self.gamma * c.R * self.gas_temperature(r, 0 * u.au) / self.mean_molecular_mass /
                        (c.G * self.star_mass / r ** 3)).to(u.au)).to(u.au)

    @u.quantity_input
    def gas_density(self, r: u.au, z: u.au) -> u.g / u.cm ** 3:
        return (self.density_1au / (
                np.sqrt(2 * np.pi) * self.scale_height(r)
        ) * np.exp(self.slope * r / (1 * u.au)) * np.exp(
            -z ** 2 / 2 / self.scale_height(r) ** 2)).to(u.g / u.cm ** 3)

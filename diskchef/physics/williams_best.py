from collections import namedtuple
from typing import List, NamedTuple, Union
from functools import cached_property
from dataclasses import dataclass

import numpy as np
import scipy.integrate
import scipy.interpolate
from astropy import units as u, constants as const

from diskchef.physics.parametrized import ParametrizedPhysics


@dataclass
class WilliamsBest2014(ParametrizedPhysics):
    """
    Class that defines physical model as in Williams & Best 2014
    https://iopscience.iop.org/article/10.1088/0004-637X/788/1/59/pdf
    """
    gas_mass: u.solMass = 1e-3 * u.solMass
    tapering_radius: u.au = 100 * u.au
    tapering_gamma: float = 0.75
    midplane_temperature_1au: u.K = 200 * u.K
    atmosphere_temperature_1au: u.K = 1000 * u.K
    temperature_slope: float = 0.55
    mean_molecular_mass: u.kg / u.mol = 2.33 * u.g / u.mol
    inner_radius: u.au = 1 * u.au

    @u.quantity_input
    def gas_temperature(self, r: u.au, z: u.au) -> u.K:
        """Function that returns gas temperature

        at given r,z using the parametrization from
        Williams & Best 2014, Eq. 5-7
        https://iopscience.iop.org/article/10.1088/0004-637X/788/1/59/pdf

        Args:
            r: u.au -- radial distance
            z: u.au -- height
        Return:
            temperature: u.K
        Raises:
            astropy.unit.UnitConversionError if units are not consistent
        """
        temp_midplane = self.midplane_temperature_1au * (r.to(u.au) / u.au) ** (-self.temperature_slope)
        temp_atmosphere = self.atmosphere_temperature_1au * (r.to(u.au) / u.au) ** (-self.temperature_slope)
        pressure_scalehight = (
                (
                        const.R * temp_midplane * r ** 3 /
                        (const.G * self.star_mass * self.mean_molecular_mass)
                ) ** 0.5
        ).to(u.au)
        temperature = np.zeros_like(z).value << u.K
        indices_atmosphere = z >= 4 * pressure_scalehight
        indices_midplane = ~ indices_atmosphere
        temperature[indices_atmosphere] = temp_atmosphere[indices_atmosphere]
        temperature[indices_midplane] = (
                temp_midplane[indices_midplane]
                + (temp_atmosphere[indices_midplane] - temp_midplane[indices_midplane])
                * np.sin((np.pi * z[indices_midplane] / (8 * pressure_scalehight[indices_midplane]))
                         .to(u.rad, equivalencies=u.dimensionless_angles())
                         ) ** 4
        )

        return temperature

    @u.quantity_input
    def gas_density(self, r: u.au, z: u.au) -> u.g / u.cm ** 3:
        """
        Calculates gas density according to Eq. 1-3
        Args:
            r: u.au -- radial distance
            z: u.au -- height

        Returns:
            gas density, in u.g / u.cm ** 3
        """
        density = np.zeros_like(z.value) << u.g / u.cm ** 3
        for _r in set(r):
            indices_this_radius = r == _r
            density[indices_this_radius] = self._vertical_density(_r)(z[indices_this_radius])
        return density

    @u.quantity_input
    def _vertical_density(self, _r: u.au, zsteps: int = 100, maxzr: float = 1):
        """
        Calculates unnormalized vertical density according to Eq. 1

        Args:
            _r: u.au -- a single radius to calculate vertical scale

        Returns:
            callable to calculate vertical density
        """
        z = np.linspace(0, _r * maxzr, zsteps).to(u.au)
        r = np.ones_like(z).value * _r
        local_temperature = self.gas_temperature(r, z)
        unscaled_log_density = scipy.integrate.cumtrapz(
            self._vertical_density_integrator(_r, z, local_temperature), z, initial=0
        )
        unscaled_density = np.exp(unscaled_log_density)
        total_density = np.trapz(unscaled_density, z)
        scaled_density = (unscaled_density / total_density * self.column_density(_r)).to(u.g / u.cm ** 3)
        callable = scipy.interpolate.interp1d(z, scaled_density)

        def quantity_callable(z: u.au) -> u.g / u.cm ** 3:
            return callable(z.to(u.au)) << (u.g / u.cm ** 3)

        return quantity_callable

    def _vertical_density_integrator(self, r, z, t):
        dlntdz = np.zeros_like(t).value << (1 / u.au)
        dlntdz[:-1] = (t[1:] - t[:-1]) / (z[1:] - z[:-1]) / (t[:-1])
        return -(
                const.G * self.star_mass * z / (r ** 2 + z ** 2) ** 1.5
                * self.mean_molecular_mass / const.R / t
                + dlntdz
        ).to(1 / u.au)

    @u.quantity_input
    def dust_density(self, r: u.au, z: u.au) -> u.g / u.cm ** 3:
        return self.gas_density(r, z) * self.dust_to_gas

    dust_temperature = gas_temperature

    @cached_property
    def column_density_1au(self) -> u.g / u.cm ** 2:
        return (2 - self.tapering_gamma) * self.star_mass / (2 * np.pi * self.tapering_radius ** 2) \
               * np.exp((self.inner_radius / self.tapering_radius) ** (2 - self.tapering_gamma))

    @u.quantity_input
    def column_density(self, r: u.au) -> u.g / u.cm ** 2:
        return self.column_density_1au * (r / self.tapering_radius) ** (-self.tapering_gamma) \
               * np.exp(-(r / self.tapering_radius) ** (2 - self.tapering_gamma))

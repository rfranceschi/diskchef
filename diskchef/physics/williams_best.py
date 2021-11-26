from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pytest
import scipy.integrate
import scipy.interpolate
from astropy import units as u, constants as const

from diskchef.physics.parametrized import ParametrizedPhysics


@dataclass
class WilliamsBest2014(ParametrizedPhysics):
    """
    Class that defines physical model as in Williams & Best 2014
    https://iopscience.iop.org/article/10.1088/0004-637X/788/1/59/pdf

    Usage:

    >>> # Initializing physics class with default parameters
    >>> physics = WilliamsBest2014()
    >>> physics  # doctest: +NORMALIZE_WHITESPACE
    WilliamsBest2014(star_mass=<Quantity 1. solMass>, xray_plasma_temperature=<Quantity 10000000. K>,
                     xray_luminosity=<Quantity 1.e+31 erg / s>, r_min=<Quantity 0.1 AU>, r_max=<Quantity 500. AU>,
                     zr_max=0.7, radial_bins=100, vertical_bins=100, dust_to_gas=0.01,
                     gas_mass=<Quantity 0.001 solMass>, tapering_radius=<Quantity 100. AU>,
                     tapering_gamma=0.75, midplane_temperature_1au=<Quantity 200. K>,
                      atmosphere_temperature_1au=<Quantity 1000. K>, temperature_slope=0.55,
                      molar_mass=<Quantity 2.33 g / mol>, inner_radius=<Quantity 1. AU>, inner_depletion=1e-06)
    >>> # Defaults can be overridden
    >>> WilliamsBest2014(star_mass=2 * u.solMass, r_max=200 * u.au)  # doctest: +NORMALIZE_WHITESPACE
    WilliamsBest2014(star_mass=<Quantity 2. solMass>, xray_plasma_temperature=<Quantity 10000000. K>,
                     xray_luminosity=<Quantity 1.e+31 erg / s>, r_min=<Quantity 0.1 AU>,
                     r_max=<Quantity 200. AU>, zr_max=0.7, radial_bins=100, vertical_bins=100, dust_to_gas=0.01,
                     gas_mass=<Quantity 0.001 solMass>, tapering_radius=<Quantity 100. AU>, tapering_gamma=0.75,
                     midplane_temperature_1au=<Quantity 200. K>, atmosphere_temperature_1au=<Quantity 1000. K>,
                     temperature_slope=0.55, molar_mass=<Quantity 2.33 g / mol>, inner_radius=<Quantity 1. AU>,
                     inner_depletion=1e-06)

    >>> # Generate physics on 3x3 grid
    >>> physics_small_grid = WilliamsBest2014(vertical_bins=3, radial_bins=3)
    >>> # table attribute stores the table with the model. Called first time, calculates the table
    >>> table = physics_small_grid.table
    >>> # print table with floats in exponential format
    >>> table  # doctest: +NORMALIZE_WHITESPACE
    <CTable length=9>
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature
         AU           AU                         g / cm3      g / cm3           K               K
      float64      float64        float64        float64      float64        float64         float64
    ------------ ------------ ---------------- ------------ ------------ --------------- ----------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-15 1.153290e-17    7.096268e+02     7.096268e+02
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-34 5.024587e-36    3.548134e+03     3.548134e+03
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-72 5.921268e-74    3.548134e+03     3.548134e+03
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-13 2.285168e-15    6.820453e+01     6.820453e+01
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-17 2.716386e-19    3.410227e+02     3.410227e+02
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-23 7.189026e-25    3.410227e+02     3.410227e+02
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-20 2.871710e-22    6.555359e+00     6.555359e+00
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-22 6.929036e-24    2.625083e+01     2.625083e+01
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-23 8.170464e-25    3.277680e+01     3.277680e+01
    >>> physics_wrong_unit = WilliamsBest2014(star_mass=2)  # Does NOT raise an exception
    >>> physics_wrong_unit.table  # but will cause unit conversion errors later
    Traceback (most recent call last):
       ...
    astropy.units.core.UnitConversionError: 'AU(3/2) J(1/2) kg(1/2) s / (g(1/2) m(3/2))' and 'AU' (length) are not convertible
    """
    gas_mass: u.solMass = 1e-3 * u.solMass
    tapering_radius: u.au = 100 * u.au
    tapering_gamma: float = 0.75
    midplane_temperature_1au: u.K = 200 * u.K
    atmosphere_temperature_1au: u.K = 1000 * u.K
    temperature_slope: float = 0.55
    molar_mass: u.kg / u.mol = 2.33 * u.g / u.mol
    inner_radius: u.au = 1 * u.au
    inner_depletion: float = 1e-6

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
            astropy.unit.UnitConversion

            Error if units are not consistent
        """
        temp_midplane = self.midplane_temperature_1au * (r.to(u.au) / u.au) ** (-self.temperature_slope)
        temp_atmosphere = self.atmosphere_temperature_1au * (r.to(u.au) / u.au) ** (-self.temperature_slope)
        pressure_scalehight = (
                (
                        const.R * temp_midplane * r ** 3 /
                        (const.G * self.star_mass * self.molar_mass)
                ) ** 0.5
        ).to(u.au)
        temperature = u.Quantity(np.zeros_like(z)).value << u.K
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
                * self.molar_mass / const.R / t
                + dlntdz
        ).to(1 / u.au)

    @u.quantity_input
    def dust_density(self, r: u.au, z: u.au) -> u.g / u.cm ** 3:
        return self.gas_density(r, z) * self.dust_to_gas

    dust_temperature = gas_temperature

    @cached_property
    def column_density_norm(self) -> u.g / u.cm ** 2:
        return (2 - self.tapering_gamma) * self.gas_mass / (2 * np.pi * self.tapering_radius ** 2) \
               * np.exp((self.inner_radius / self.tapering_radius) ** (2 - self.tapering_gamma))

    @u.quantity_input
    def column_density(self, r: u.au) -> u.g / u.cm ** 2:
        """Gas column density at given radius"""
        drop = np.where(r < self.inner_radius, self.inner_depletion, 1)
        return drop * self.column_density_norm * (r / self.tapering_radius) ** (-self.tapering_gamma) \
               * np.exp(-(r / self.tapering_radius) ** (2 - self.tapering_gamma))


@dataclass
class WilliamsBest100au(WilliamsBest2014):
    """
    A subclass of WilliamsBest2014 which uses temperatures at 100 au rather than at 1 au for a more robust fitting

    These defintions are equivalent:
    >>> t_a, t_m, sl = 100, 1000, 0.55
    >>> original = WilliamsBest2014(vertical_bins=3, radial_bins=3,
    ...     midplane_temperature_1au=t_a * u.K, atmosphere_temperature_1au=t_m * u.K,
    ...     temperature_slope=sl)
    >>> at_100_au = WilliamsBest100au(vertical_bins=3, radial_bins=3,
    ...     midplane_temperature_100au=t_a * u.K / 100**sl, atmosphere_temperature_100au = t_m * u.K / 100**sl,
    ...     temperature_slope=sl)
    >>> original.table     # doctest: +NORMALIZE_WHITESPACE
    <CTable length=9>
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature
         AU           AU                         g / cm3      g / cm3           K               K
      float64      float64        float64        float64      float64        float64         float64
    ------------ ------------ ---------------- ------------ ------------ --------------- ----------------
    1.000000e-01 0.000000e+00     0.000000e+00 2.102690e-15 2.102690e-17    3.548134e+02     3.548134e+02
    1.000000e-01 3.500000e-02     3.500000e-01 3.080145e-34 3.080145e-36    3.548134e+03     3.548134e+03
    1.000000e-01 7.000000e-02     7.000000e-01 3.629823e-72 3.629823e-74    3.548134e+03     3.548134e+03
    7.071068e+00 0.000000e+00     0.000000e+00 3.593866e-13 3.593866e-15    3.410227e+01     3.410227e+01
    7.071068e+00 2.474874e+00     3.500000e-01 2.138733e-17 2.138733e-19    3.410227e+02     3.410227e+02
    7.071068e+00 4.949747e+00     7.000000e-01 5.660244e-23 5.660244e-25    3.410227e+02     3.410227e+02
    5.000000e+02 0.000000e+00     0.000000e+00 4.346110e-20 4.346110e-22    3.277680e+00     3.277680e+00
    5.000000e+02 1.750000e+02     3.500000e-01 4.525949e-22 4.525949e-24    3.277680e+01     3.277680e+01
    5.000000e+02 3.500000e+02     7.000000e-01 6.835933e-23 6.835933e-25    3.277680e+01     3.277680e+01
    >>> str(at_100_au.table) == str(original.table)
    True
    """
    midplane_temperature_100au: u.K = 15.89 * u.K
    atmosphere_temperature_100au: u.K = 281.8 * u.K
    midplane_temperature_1au: u.K = None
    atmosphere_temperature_1au: u.K = None

    def __post_init__(self):
        super().__post_init__()
        if self.midplane_temperature_1au is not None:
            self.logger.warning("'midplane_temperature_1au' is ignored, use 'midplane_temperature_100au' instead")
        if self.atmosphere_temperature_1au is not None:
            self.logger.warning("'atmosphere_temperature_1au' is ignored, use 'atmosphere_temperature_100au' instead")

        self.midplane_temperature_1au = self.midplane_temperature_100au * 100 ** self.temperature_slope
        self.atmosphere_temperature_1au = self.atmosphere_temperature_100au * 100 ** self.temperature_slope

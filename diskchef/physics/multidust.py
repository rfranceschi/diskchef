"""Module with a definition of multiple dust populations"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from astropy import units as u

from diskchef.engine.other import PathLike
from diskchef.engine.ctable import CTable
from diskchef.engine.exceptions import CHEFRuntimeError


@dataclass
class DustPopulation:
    """
    Class with the data on the dust population

    Args:
        total_dust_density:  total density of all dust components, `astropy.units.Quantity` in density units
        mass_fraction:  multiplier to get the density of this dust population
        average_size_for_chemistry:  average grain size weighted by surface area, see Vasyunin et al. 2011 Eq. 9
        (https://iopscience.iop.org/article/10.1088/0004-637X/727/2/76/pdf)


    Usage:

    >>> # If the name is not set, use index of the created dust (unsafe!)
    >>> dust0 = DustPopulation("some_opacity.inp", total_dust_density=1e-15 * u.g / u.cm ** 3)
    >>> dust0.name
    'Dust_0'
    >>> dust1 = DustPopulation("some_opacity.inp", total_dust_density=1e-15 * u.g / u.cm ** 3)
    >>> dust1.name
    'Dust_1'

    >>> table = CTable()
    >>> table["Radius"] = [1, 2, 3] * u.au
    >>> table["Dust density"] = [100, 10, 1] * u.g / u.cm ** 3 * 1e-15
    >>> table["Dust temperature"] = [100, 50, 25] * u.K
    >>> dust = DustPopulation("some_opacity.inp", table=table, name="Default dust")
    >>> dust.write_to_table()
    >>> table  # doctest: +NORMALIZE_WHITESPACE
    <CTable length=3>
       Radius    Dust density Dust temperature Default dust mass fraction Default dust number density Default dust temperature Default dust surface area per volume
         AU        g / cm3           K                                              1 / cm3                      K                            1 / cm
      float64      float64        float64               float64                     float64                   float64                        float64
    ------------ ------------ ---------------- -------------------------- --------------------------- ------------------------ ------------------------------------
    1.000000e+00 1.000000e-13     1.000000e+02               1.000000e+00                7.957747e+00             1.000000e+02                         1.000000e-08
    2.000000e+00 1.000000e-14     5.000000e+01               1.000000e+00                7.957747e-01             5.000000e+01                         1.000000e-09
    3.000000e+00 1.000000e-15     2.500000e+01               1.000000e+00                7.957747e-02             2.500000e+01                         1.000000e-10
    >>> table.dust_list[0].name
    'Default dust'

    >>> dust_large = DustPopulation("large_opacity.inp", table=table, name="Large dust", mass_fraction=0.3, average_size_for_chemistry=1*u.cm)
    >>> dust_large.write_to_table()
    >>> table  # doctest: +NORMALIZE_WHITESPACE
    <CTable length=3>
       Radius    Dust density Dust temperature Default dust mass fraction Default dust number density Default dust temperature Default dust surface area per volume Large dust mass fraction Large dust number density Large dust temperature Large dust surface area per volume
         AU        g / cm3           K                                              1 / cm3                      K                            1 / cm                                                  1 / cm3                    K                          1 / cm
      float64      float64        float64               float64                     float64                   float64                        float64                        float64                   float64                 float64                      float64
    ------------ ------------ ---------------- -------------------------- --------------------------- ------------------------ ------------------------------------ ------------------------ ------------------------- ---------------------- ----------------------------------
    1.000000e+00 1.000000e-13     1.000000e+02               1.000000e+00                7.957747e+00             1.000000e+02                         1.000000e-08             3.000000e-01              2.387324e-15           1.000000e+02                       3.000000e-14
    2.000000e+00 1.000000e-14     5.000000e+01               1.000000e+00                7.957747e-01             5.000000e+01                         1.000000e-09             3.000000e-01              2.387324e-16           5.000000e+01                       3.000000e-15
    3.000000e+00 1.000000e-15     2.500000e+01               1.000000e+00                7.957747e-02             2.500000e+01                         1.000000e-10             3.000000e-01              2.387324e-17           2.500000e+01                       3.000000e-16
    >>> [dust.name for dust in table.dust_list]
    ['Default dust', 'Large dust']

    >>> # Now, the total dust population exceeds the total dust mass (sum of mass_fraction != 1)
    >>> table.dust_population_fully_set
    False
    >>> table.normalize_dust()
    >>> table.dust_population_fully_set
    True
    >>> table  # doctest: +NORMALIZE_WHITESPACE
    <CTable length=3>
       Radius    Dust density Dust temperature Default dust mass fraction Default dust number density Default dust temperature Default dust surface area per volume Large dust mass fraction Large dust number density Large dust temperature Large dust surface area per volume
         AU        g / cm3           K                                              1 / cm3                      K                            1 / cm                                                  1 / cm3                    K                          1 / cm
      float64      float64        float64               float64                     float64                   float64                        float64                        float64                   float64                 float64                      float64
    ------------ ------------ ---------------- -------------------------- --------------------------- ------------------------ ------------------------------------ ------------------------ ------------------------- ---------------------- ----------------------------------
    1.000000e+00 1.000000e-13     1.000000e+02               7.692308e-01                6.121344e+00             1.000000e+02                         7.692308e-09             2.307692e-01              1.836403e-15           1.000000e+02                       2.307692e-14
    2.000000e+00 1.000000e-14     5.000000e+01               7.692308e-01                6.121344e-01             5.000000e+01                         7.692308e-10             2.307692e-01              1.836403e-16           5.000000e+01                       2.307692e-15
    3.000000e+00 1.000000e-15     2.500000e+01               7.692308e-01                6.121344e-02             2.500000e+01                         7.692308e-11             2.307692e-01              1.836403e-17           2.500000e+01                       2.307692e-16

    >>> # Different ratio at different locations, also note that if Dust temperature was not set, it will be set as np.nan
    >>> table = CTable()
    >>> table["Radius"] = [1, 2, 3] * u.au
    >>> table["Dust density"] = [100, 10, 1] * u.g / u.cm ** 3 * 1e-15
    >>> dust = DustPopulation("some_opacity.inp", table=table, name="Default dust", mass_fraction=[0.8, 0.7, 0.5])
    >>> dust.write_to_table()
    >>> dust_large = DustPopulation("large_opacity.inp", table=table, name="Large dust", mass_fraction=[0.2, 0.3, 0.5], average_size_for_chemistry=1*u.cm)
    >>> dust_large.write_to_table()
    >>> table  # doctest: +NORMALIZE_WHITESPACE
    <CTable length=3>
       Radius    Dust density Default dust mass fraction Default dust number density Default dust temperature Default dust surface area per volume Large dust mass fraction Large dust number density Large dust temperature Large dust surface area per volume
         AU        g / cm3                                         1 / cm3                      K                            1 / cm                                                  1 / cm3                    K                          1 / cm
      float64      float64             float64                     float64                   float64                        float64                        float64                   float64                 float64                      float64
    ------------ ------------ -------------------------- --------------------------- ------------------------ ------------------------------------ ------------------------ ------------------------- ---------------------- ----------------------------------
    1.000000e+00 1.000000e-13               8.000000e-01                6.366198e+00                      nan                         8.000000e-09             2.000000e-01              1.591549e-15                    nan                       2.000000e-14
    2.000000e+00 1.000000e-14               7.000000e-01                5.570423e-01                      nan                         7.000000e-10             3.000000e-01              2.387324e-16                    nan                       3.000000e-15
    3.000000e+00 1.000000e-15               5.000000e-01                3.978874e-02                      nan                         5.000000e-11             5.000000e-01              3.978874e-17                    nan                       5.000000e-16
    >>> table.dust_population_fully_set
    True

    >>> # Rescaling also works if mass fractions are given independently
    >>> dust_asteroid = DustPopulation("asteroid_opacity.inp", table=table, name="Asteroids", mass_fraction=[0.5, 0.3, 0.5], average_size_for_chemistry=1*u.km)
    >>> dust_asteroid.write_to_table()
    >>> table.dust_population_fully_set
    False
    >>> table.normalize_dust()
    >>> table.dust_population_fully_set
    True
    >>> table  # doctest: +NORMALIZE_WHITESPACE
    <CTable length=3>
       Radius    Dust density Default dust mass fraction Default dust number density Default dust temperature Default dust surface area per volume Large dust mass fraction Large dust number density Large dust temperature Large dust surface area per volume Asteroids mass fraction Asteroids number density Asteroids temperature Asteroids surface area per volume
         AU        g / cm3                                         1 / cm3                      K                            1 / cm                                                  1 / cm3                    K                          1 / cm                                               1 / cm3                    K                         1 / cm
      float64      float64             float64                     float64                   float64                        float64                        float64                   float64                 float64                      float64                       float64                 float64                 float64                     float64
     ------------ ------------ -------------------------- --------------------------- ------------------------ ------------------------------------ ------------------------ ------------------------- ---------------------- ---------------------------------- ----------------------- ------------------------ --------------------- ---------------------------------
    1.000000e+00 1.000000e-13               5.333333e-01                4.244132e+00                      nan                         5.333333e-09             1.333333e-01              1.061033e-15                    nan                       1.333333e-14            3.333333e-01             2.652582e-30                   nan                      3.333333e-19
    2.000000e+00 1.000000e-14               5.384615e-01                4.284941e-01                      nan                         5.384615e-10             2.307692e-01              1.836403e-16                    nan                       2.307692e-15            2.307692e-01             1.836403e-31                   nan                      2.307692e-20
    3.000000e+00 1.000000e-15               3.333333e-01                2.652582e-02                      nan                         3.333333e-11             3.333333e-01              2.652582e-17                    nan                       3.333333e-16            3.333333e-01             2.652582e-32                   nan                      3.333333e-21

    """
    opacity_file: PathLike
    table: CTable = None
    total_dust_density: u.g / u.cm ** 3 = None
    dust_temperature: u.K = None
    mass_fraction: u.dimensionless_unscaled = 1.
    name: str = None
    mean_mass_per_grain: u.g = None
    grain_particle_density: u.g / u.cm ** 3 = 3 * u.g / u.cm ** 3
    opacity_file_format: Literal['radmc'] = 'radmc'
    average_size_for_chemistry: u.cm = 1e-5 * u.cm
    _idx = 0

    def __post_init__(self):
        if self.total_dust_density is None:
            if self.table is None:
                raise CHEFRuntimeError("table OR total_dust_density must be set for DustPopulation")
            self.total_dust_density = self.table["Dust density"]
        if self.mean_mass_per_grain is None:
            self.mean_mass_per_grain = 4. / 3. * np.pi * self.average_size_for_chemistry ** 3 * self.grain_particle_density

        if self.name is None: self.name = f"Dust_{self.__class__._idx}"
        self.__class__._idx += 1

    @property
    @u.quantity_input
    def number_density(self) -> u.cm ** (-3):
        """
        Number density column of given dust species
        """
        return self.total_dust_density * self.mass_fraction / self.mean_mass_per_grain

    @property
    @u.quantity_input
    def surface_area_per_volume(self) -> u.cm ** (-1):
        """
        Surface area per volume unit, important for chemistry

        Assumes spherical grains in default setup
        """
        return 4 * np.pi * self.average_size_for_chemistry ** 2 * self.number_density

    @property
    @u.quantity_input
    def temperature(self) -> u.K:
        if self.dust_temperature is not None:
            return self.dust_temperature
        if self.table is not None:
            if "Dust temperature" in self.table.colnames:
                return self.table["Dust temperature"]
            else:
                return np.nan * u.K

    def write_to_table(self, table: CTable = None):
        """
        Write the dust population in the table

        Args:
            `table`: if given, writes the data in this table. Else, writes in `self.table`

        Adds columns
            "{self.name} mass fraction"
            "{self.name} number density"
            "{self.name} temperature"
            "{self.name} surface area per volume"

        Also adds `self` to `table.dust_list`
        """
        if table is None:
            table = self.table
        table[f"{self.name} mass fraction"] = self.mass_fraction
        table[f"{self.name} number density"] = self.number_density
        table[f"{self.name} temperature"] = self.temperature
        table[f"{self.name} surface area per volume"] = self.surface_area_per_volume
        try:
            if self not in table.dust_list:
                table.dust_list.append(self)
        except KeyError:
            table.dust_list = [self]

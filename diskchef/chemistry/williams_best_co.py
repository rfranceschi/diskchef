from dataclasses import dataclass

import numpy as np
from astropy import units as u

import diskchef.engine
from diskchef.chemistry.base import ChemistryModel
from diskchef.physics.williams_best import WilliamsBest2014


@dataclass
class ChemistryWB2014(ChemistryModel):
    """Calculates chemistry according to Williams & Best 2014

    Eq. 5-7
    https://iopscience.iop.org/article/10.1088/0004-637X/788/1/59/pdf


    Usage:

    >>> # Define physics first
    >>> physics = WilliamsBest2014(radial_bins=3, vertical_bins=3)
    >>> # Define chemistry class using the physics instance
    >>> chemistry = ChemistryWB2014(physics)
    >>> # chemistry.table is pointing to the physics.table
    >>> chemistry.table  # doctest: +NORMALIZE_WHITESPACE
    <CTable length=9>
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature   n(H+2H2)
         AU           AU                         g / cm3      g / cm3           K               K           1 / cm3
      float64      float64        float64        float64      float64        float64         float64        float64
    ------------ ------------ ---------------- ------------ ------------ --------------- ---------------- ------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-15 1.153290e-17    7.096268e+02     7.096268e+02 5.095483e+08
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-34 5.024587e-36    3.548134e+03     3.548134e+03 2.219970e-10
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-72 5.921268e-74    3.548134e+03     3.548134e+03 2.616143e-48
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-13 2.285168e-15    6.820453e+01     6.820453e+01 1.009636e+11
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-17 2.716386e-19    3.410227e+02     3.410227e+02 1.200157e+07
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-23 7.189026e-25    3.410227e+02     3.410227e+02 3.176265e+01
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-20 2.871710e-22    6.555359e+00     6.555359e+00 1.268783e+04
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-22 6.929036e-24    2.625083e+01     2.625083e+01 3.061396e+02
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-23 8.170464e-25    3.277680e+01     3.277680e+01 3.609885e+01

    >>> # Now run chemistry calculations
    >>> chemistry.run_chemistry()
    >>> # This updates chemistry.table with new columns
    >>> chemistry.table.colnames  # doctest: +NORMALIZE_WHITESPACE
    ['Radius', 'Height', 'Height to radius', 'Gas density', 'Dust density', 'Gas temperature', 'Dust temperature',
    'n(H+2H2)', 'H2', 'H2 number density', 'H2 column density towards star', 'H2 column density upwards', 'CO']
    >>> chemistry.table['H2']
    <Column name='H2' dtype='float64' format='e' length=9>
    5.000000e-01
    5.000000e-01
    5.000000e-01
    5.000000e-01
    5.000000e-01
    5.000000e-01
    5.000000e-01
    5.000000e-01
    5.000000e-01

    >>> # Remember that chemistry.table is just a pointer to chemistry.physics.table:
    >>> chemistry.table is chemistry.physics.table
    True
    """
    midplane_co_abundance: float = 0
    molecular_layer_co_abundance: float = 1e-4
    co_freezeout_temperature: u.Quantity = 20 * u.K
    atmosphere_co_abundance: float = 0
    h2_column_denisty_that_shields_co: u.Quantity = 1.3e21 / u.cm ** 2

    def run_chemistry(self):
        self.table["H2"] = 0.5
        self.calculate_column_density_towards_star("H2")
        self.calculate_column_density_upwards("H2")

        self.table["CO"] = self.midplane_co_abundance
        self.table["CO"][
            self.table["Gas temperature"] > self.co_freezeout_temperature
            ] = self.molecular_layer_co_abundance

        self.table["CO"][
            (self.table["H2 column density towards star"] < self.h2_column_denisty_that_shields_co)
            | (self.table["H2 column density upwards"] < self.h2_column_denisty_that_shields_co)
            ] = self.atmosphere_co_abundance


@dataclass
class NonzeroChemistryWB2014(ChemistryWB2014):
    """
    Subclass of ChemistryWB2014 that has non-zero default midplane and atmosphere abundance

    Usage:

    >>> physics = WilliamsBest2014(radial_bins=3, vertical_bins=3)
    >>> chemistry = NonzeroChemistryWB2014(physics)
    >>> chemistry.run_chemistry()
    >>> chemistry.table["CO"].info.format = "e"
    >>> chemistry.table["CO"]
    <Column name='CO' dtype='float64' format='e' length=9>
    1.000000e-10
    1.000000e-10
    1.000000e-10
    1.000000e-04
    1.000000e-10
    1.000000e-10
    1.000000e-10
    1.000000e-10
    1.000000e-10
    """
    midplane_co_abundance: float = 1e-6
    atmosphere_co_abundance: float = 1e-10

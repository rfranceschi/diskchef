from dataclasses import dataclass

from astropy import units as u

from diskchef.chemistry.base import ChemistryBase
from diskchef.physics.williams_best import WilliamsBest2014

@dataclass
class ChemistryWB2014(ChemistryBase):
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
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature   n(H+2H2)
         AU           AU                         g / cm3      g / cm3           K               K           1 / cm3
    ------------ ------------ ---------------- ------------ ------------ --------------- ---------------- ------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-06 1.153290e-08    7.096268e+02     7.096268e+02 2.980806e+17
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-25 5.024587e-27    3.548134e+03     3.548134e+03 1.298660e-01
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-63 5.921268e-65    3.548134e+03     3.548134e+03 1.530417e-39
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-10 2.285168e-12    6.820453e+01     6.820453e+01 5.906268e+13
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-14 2.716386e-16    3.410227e+02     3.410227e+02 7.020797e+09
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-20 7.189026e-22    3.410227e+02     3.410227e+02 1.858083e+04
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-17 2.871710e-19    6.555359e+00     6.555359e+00 7.422251e+06
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-19 6.929036e-21    2.625083e+01     2.625083e+01 1.790885e+05
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-20 8.170464e-22    3.277680e+01     3.277680e+01 2.111746e+04

    >>> # Now run chemistry calculations
    >>> chemistry.run_chemistry()
    >>> # This updates chemistry.table with new columns
    >>> chemistry.table.colnames  # doctest: +NORMALIZE_WHITESPACE
    ['Radius', 'Height', 'Height to radius', 'Gas density', 'Dust density', 'Gas temperature', 'Dust temperature',
     'n(H+2H2)', 'H2', 'H2 number density', 'H2 column density towards star', 'CO']
    >>> chemistry.table['H2']
    <Column name='H2' dtype='float64' length=9>
    0.5
    0.5
    0.5
    0.5
    0.5
    0.5
    0.5
    0.5
    0.5
    >>> # Remember that chemistry.table is just a pointer to chemistry.physics.table:
    >>> chemistry.table is chemistry.physics.table
    True
    """
    midplane_co_abundance = 0
    molecular_layer_co_abundance = 1e-4
    co_freezeout_temperature = 20 * u.K
    atmosphere_co_abundance = 0
    h2_column_denisty_that_shields_co = 1.3e21 / u.cm ** 2

    def run_chemistry(self):
        self.table["H2"] = 0.5
        self.calculate_column_density_towards_star("H2")

        self.table["CO"] = self.midplane_co_abundance
        self.table["CO"][
            self.table["Gas temperature"] > self.co_freezeout_temperature
            ] = self.molecular_layer_co_abundance

        self.table["CO"][
            self.table["H2 column density towards star"] < self.h2_column_denisty_that_shields_co
            ] = self.atmosphere_co_abundance

    def calculate_column_density_towards_star(self, species: str):
        """
        Calculates the column density of a given species towards star for each self.table row

        Adds two columns to the table: f"{species} number density" and f"{species} column density towards star"

        Args:
            species: species to integrate

        Returns: self.table[f"{species} column density towards star"]
        """
        self.table[f"{species} number density"] = self.table[species] * self.table["n(H+2H2)"]
        self.table[f"{species} column density towards star"] = self.physics.column_density_to(
            self.table.r, self.table.z,
            f"{species} number density",
            only_gridpoint=True,
        )

        return self.table[f"{species} column density towards star"]


@dataclass
class NonzeroChemistryWB2014(ChemistryWB2014):
    """Subclass of ChemistryWB2014 that has non-zero default midplane and atmosphere abundance"""
    midplane_co_abundance = 1e-6
    atmosphere_co_abundance = 1e-10

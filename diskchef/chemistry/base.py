from functools import cached_property
from dataclasses import dataclass
import logging

from matplotlib import colors
from astropy import units as u, constants as const

from divan import Divan

from diskchef.physics.base import PhysicsBase
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.chemistry.abundances import Abundances


@dataclass
class ChemistryBase:
    """
    Base class for chemistry with fixed abundances

    Args:
        physics: instance of `PhysicsBase`-like class
        initial_abundances: instance of `Abundances` class with initial abundances
        mean_molecular_mass: in u.g / u.mol units

    Usage:

    >>> # Define physics first
    >>> physics = WilliamsBest2014(radial_bins=3, vertical_bins=3)
    >>> # Define chemistry class using the physics instance
    >>> chemistry = ChemistryBase(physics)
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
    >>> chemistry.initial_abundances == {'H2': 5e-01, 'CO': 5e-05}
    True
    >>> chemistry.run_chemistry()
    >>> # The base class just sets abundance to self.initial_abundances
    >>> chemistry.table  # doctest: +NORMALIZE_WHITESPACE
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature   n(H+2H2)        H2           CO
         AU           AU                         g / cm3      g / cm3           K               K           1 / cm3
    ------------ ------------ ---------------- ------------ ------------ --------------- ---------------- ------------ ------------ ------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-06 1.153290e-08    7.096268e+02     7.096268e+02 2.980806e+17 5.000000e-01 5.000000e-05
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-25 5.024587e-27    3.548134e+03     3.548134e+03 1.298660e-01 5.000000e-01 5.000000e-05
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-63 5.921268e-65    3.548134e+03     3.548134e+03 1.530417e-39 5.000000e-01 5.000000e-05
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-10 2.285168e-12    6.820453e+01     6.820453e+01 5.906268e+13 5.000000e-01 5.000000e-05
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-14 2.716386e-16    3.410227e+02     3.410227e+02 7.020797e+09 5.000000e-01 5.000000e-05
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-20 7.189026e-22    3.410227e+02     3.410227e+02 1.858083e+04 5.000000e-01 5.000000e-05
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-17 2.871710e-19    6.555359e+00     6.555359e+00 7.422251e+06 5.000000e-01 5.000000e-05
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-19 6.929036e-21    2.625083e+01     2.625083e+01 1.790885e+05 5.000000e-01 5.000000e-05
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-20 8.170464e-22    3.277680e+01     3.277680e+01 2.111746e+04 5.000000e-01 5.000000e-05

    """
    physics: PhysicsBase = None
    initial_abundances: Abundances = Abundances()
    mean_molecular_mass: u.g / u.mol = 2.33 * u.g / u.mol

    @property
    def table(self):
        """Shortcut for self.physics.table"""
        return self.physics.table

    def __post_init__(self):
        self.update_hydrogen_atom_number_density()
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", __class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

    def update_hydrogen_atom_number_density(self):
        """Calculates the hydrogen atom density used to scale all other atoms to"""
        self.table["n(H+2H2)"] = self.table["Gas density"] / self.mean_molecular_mass * const.N_A

    def run_chemistry(self):
        """Writes the output of the chemical model into self.table

        In the default class, just adopts the values from self.initial_abundances
        """
        for species, abundance in self.initial_abundances.items():
            self.table[species] = abundance

    def plot_chemistry(self, table=None):
        if table is None:
            table = self.table
        dvn = Divan()
        dvn.chemical_structure = table
        dvn.generate_figure_chemistry(spec1="CO", spec2="CO", normalizer=colors.LogNorm())

    def plot_h2_coldens(self):
        dvn = Divan()
        dvn.chemical_structure = self.table
        dvn.generate_figure(
            r=self.table.r, z=self.table.z,
            data1=self.table["H2 column density towards star"],
            normalizer=colors.LogNorm(1e10, 1e30)
        )

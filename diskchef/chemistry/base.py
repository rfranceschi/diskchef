from dataclasses import dataclass
import logging
from typing import Union

from astropy import units as u, constants as const
from matplotlib import colors
import matplotlib.axes

import diskchef
from diskchef.engine.exceptions import CHEFNotImplementedError
from diskchef.chemistry.abundances import Abundances
from diskchef.physics.base import PhysicsBase
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.engine.plot import Plot2D

@dataclass
class ChemistryBase:
    physics: PhysicsBase = None
    mean_molecular_mass: u.g / u.mol = 2.33 * u.g / u.mol

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)
        try:
            self.update_hydrogen_atom_number_density()
        except KeyError as e:
            self.logger.warning("'Gas density' is not defined in physics.table of %s: %s", self.physics, e)

    @property
    def table(self):
        """Shortcut for self.physics.table"""
        return self.physics.table

    def run_chemistry(self):
        """Writes the output of the chemical model into self.table"""
        raise CHEFNotImplementedError

    def update_hydrogen_atom_number_density(self):
        """Calculates the hydrogen atom density used to scale all other atoms to"""
        self.table["n(H+2H2)"] = self.table["Gas density"] / self.mean_molecular_mass * const.N_A

    def plot_chemistry(
            self,
            species1, species2=None,
            axes: matplotlib.axes.Axes = None,
            table: diskchef.CTable = None, folder=".",
            cmap: Union[matplotlib.colors.Colormap, str] = 'YlGnBu',
            **kwargs
    ):
        if table is None:
            table = self.table
        Plot2D(table, axes=axes, data1=species1, data2=species2, cmap=cmap, **kwargs)

    def plot_absolute_chemistry(self, *args, cmap="RdPu", **kwargs):
        self.plot_chemistry(*args, multiply_by="n(H+2H2)", cmap=cmap, **kwargs)

@dataclass
class ChemistryModel(ChemistryBase):
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
    >>> chemistry = ChemistryModel(physics)
    >>> # chemistry.table is pointing to the physics.table
    >>> chemistry.table  # doctest: +NORMALIZE_WHITESPACE
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature   n(H+2H2)
         AU           AU                         g / cm3      g / cm3           K               K           1 / cm3
    ------------ ------------ ---------------- ------------ ------------ --------------- ---------------- ------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-15 1.153290e-17    7.096268e+02     7.096268e+02 2.980806e+08
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-34 5.024587e-36    3.548134e+03     3.548134e+03 1.298660e-10
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-72 5.921268e-74    3.548134e+03     3.548134e+03 1.530417e-48
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-13 2.285168e-15    6.820453e+01     6.820453e+01 5.906268e+10
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-17 2.716386e-19    3.410227e+02     3.410227e+02 7.020797e+06
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-23 7.189026e-25    3.410227e+02     3.410227e+02 1.858083e+01
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-20 2.871710e-22    6.555359e+00     6.555359e+00 7.422251e+03
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-22 6.929036e-24    2.625083e+01     2.625083e+01 1.790885e+02
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-23 8.170464e-25    3.277680e+01     3.277680e+01 2.111746e+01
    >>> chemistry.initial_abundances == {'H2': 5e-01, 'CO': 5e-05}
    True
    >>> chemistry.run_chemistry()
    >>> # The base class just sets abundance to self.initial_abundances
    >>> chemistry.table  # doctest: +NORMALIZE_WHITESPACE
       Radius       Height    Height to radius Gas density  Dust density Gas temperature Dust temperature   n(H+2H2)        H2           CO
         AU           AU                         g / cm3      g / cm3           K               K           1 / cm3
    ------------ ------------ ---------------- ------------ ------------ --------------- ---------------- ------------ ------------ ------------
    1.000000e-01 0.000000e+00     0.000000e+00 1.153290e-15 1.153290e-17    7.096268e+02     7.096268e+02 2.980806e+08 5.000000e-01 5.000000e-05
    1.000000e-01 3.500000e-02     3.500000e-01 5.024587e-34 5.024587e-36    3.548134e+03     3.548134e+03 1.298660e-10 5.000000e-01 5.000000e-05
    1.000000e-01 7.000000e-02     7.000000e-01 5.921268e-72 5.921268e-74    3.548134e+03     3.548134e+03 1.530417e-48 5.000000e-01 5.000000e-05
    7.071068e+00 0.000000e+00     0.000000e+00 2.285168e-13 2.285168e-15    6.820453e+01     6.820453e+01 5.906268e+10 5.000000e-01 5.000000e-05
    7.071068e+00 2.474874e+00     3.500000e-01 2.716386e-17 2.716386e-19    3.410227e+02     3.410227e+02 7.020797e+06 5.000000e-01 5.000000e-05
    7.071068e+00 4.949747e+00     7.000000e-01 7.189026e-23 7.189026e-25    3.410227e+02     3.410227e+02 1.858083e+01 5.000000e-01 5.000000e-05
    5.000000e+02 0.000000e+00     0.000000e+00 2.871710e-20 2.871710e-22    6.555359e+00     6.555359e+00 7.422251e+03 5.000000e-01 5.000000e-05
    5.000000e+02 1.750000e+02     3.500000e-01 6.929036e-22 6.929036e-24    2.625083e+01     2.625083e+01 1.790885e+02 5.000000e-01 5.000000e-05
    5.000000e+02 3.500000e+02     7.000000e-01 8.170464e-23 8.170464e-25    3.277680e+01     3.277680e+01 2.111746e+01 5.000000e-01 5.000000e-05

    """
    initial_abundances: Abundances = Abundances()

    def run_chemistry(self):
        """Writes the output of the chemical model into self.table

        In the default class, just adopts the values from self.initial_abundances
        """
        for species, abundance in self.initial_abundances.items():
            self.table[species] = abundance

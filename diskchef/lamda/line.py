from dataclasses import dataclass
from typing import Union
import logging

from astropy import units as u
from astropy.table import QTable

import diskchef.lamda
from diskchef import CTable
from diskchef.engine.other import PathLike


@dataclass
class Line:
    """
    A class that reads and stores the information of the emission line

    `Line` instances is hashable, `name` is the key used to hash

    Usage:
    >>> line = Line(name="HCO+ J=2-1", transition=1, molecule="HCO+")
    >>> line.lamda_molecule, line.molweight, line.number_levels
    ('HCO+', 29.0, 31)

    Use `line.levels.loc` to index by J. Slices and lists, but not tuples, are allowed.
    >>> line.levels.loc[[1,2]]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <CTable length=2>
       Level        Energy       Weight         J
                    1 / cm
       int...      float...     float...      int...
    ------------ ------------ ------------ ------------
    2.000000e+00 2.975000e+00 3.000000e+00 1.000000e+00
    3.000000e+00 8.925000e+00 5.000000e+00 2.000000e+00

    >>> line.transitions.loc[1]  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <Row index=0>
    Transition   Up   Low  Einstein A Frequency Energy upper
                             1 / s       GHz         K
      int...  int... int... float...   float...   float...
    ---------- ----- ----- ---------- --------- ------------
             1     2     1 4.2512e-05 89.188523         4.28
    """
    name: str
    transition: int
    molecule: str
    collision_partner: tuple = ('H2',)
    lamda_file: Union[None, PathLike] = None

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__ + f'({self.name})')
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)
        self.parse_lamda()

    def parse_lamda(self):
        """Parses LAMDA database file to identify collision partner and frequency"""
        if self.lamda_file is None:
            self.lamda_file = diskchef.lamda.lamda_files(self.molecule)[0]
            self.logger.info("Found LAMDA file: %s", self.lamda_file)
        with open(self.lamda_file) as lamda:
            lamda.readline()
            self.lamda_molecule = lamda.readline().strip()
            lamda.readline()
            self.molweight = float(lamda.readline().strip())
            lamda.readline()
            self.number_levels = int(lamda.readline().strip())

            lamda.readline()
            levels = [next(lamda) for i in range(self.number_levels)]
            self.levels = CTable.read(
                levels, format='ascii', names=["Level", "Energy", "Weight", "J"],
            )
            try:
                self.levels["Energy"].unit = 1. / u.cm
            except AttributeError:
                self.levels["Energy"] *= (1. / u.cm)
            self.levels.add_index("J")

            lamda.readline()
            self.number_transitions = int(lamda.readline().strip())
            lamda.readline()
            transitions = [next(lamda) for i in range(self.number_transitions)]
            self.transitions = QTable.read(
                transitions, format='ascii',
                names=["Transition", "Up", "Low", "Einstein A", "Frequency", "Energy upper"],
            )
            self.transitions["Einstein A"].unit = 1. / u.s
            self.transitions["Frequency"].unit = u.GHz
            self.transitions["Energy upper"].unit = u.K
            self.transitions.add_index("Transition")

            # TODO collision partners are not yet parsed

    @property
    def frequency(self):
        return self.transitions.loc[self.transition]["Frequency"]

    def __hash__(self):
        return hash(self.name)

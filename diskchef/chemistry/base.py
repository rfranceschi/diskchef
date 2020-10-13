from functools import cached_property
from dataclasses import dataclass

from astropy import units as u, constants as const

from diskchef.physics.base import PhysicsBase
from diskchef.chemistry.abundances import Abundances


@dataclass
class ChemistryBase:
    physics: PhysicsBase = None
    initial_abundances: Abundances = Abundances()
    mean_molecular_mass: u.kg / u.mol = 2.33 * u.g / u.mol

    @property
    def table(self):
        return self.physics.table

    def __post_init__(self):
        self.update_hydrogen_atom_number_density()

    def update_hydrogen_atom_number_density(self):
        self.table["n(H+2H2)"] = self.table["Gas density"] / self.mean_molecular_mass * const.N_A

    def run_chemistry(self):
        """Writes the output of the chemical model into self.table

        In the default class, just adopts the values from self.initial_abundances
        """
        for species, abundance in self.initial_abundances.items():
            self.table[species] = abundance

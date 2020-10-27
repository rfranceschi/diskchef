from functools import cached_property
from dataclasses import dataclass
import logging

from matplotlib import colors
from astropy import units as u, constants as const

from divan import Divan

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
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", __class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

    def update_hydrogen_atom_number_density(self):
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

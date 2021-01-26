from dataclasses import dataclass
import os

import astropy.io.ascii
from astropy import units as u
from astropy.visualization import quantity_support

import diskchef.physics.base

quantity_support()

from diskchef import CTable

from diskchef.chemistry.base import ChemistryBase
from diskchef.engine.other import PathLike


@dataclass
class ReadAndesData(ChemistryBase):
    """
    blah
    """
    folder: PathLike = None
    index: int = 0

    @property
    def table(self) -> CTable:
        return self._table

    def read(self, index: int = None) -> CTable:
        """Return the associated diskchef.CTable with r, z, and dust and gas properties"""
        if index is None:
            index = self.index
        reader = astropy.io.ascii.get_reader(Reader=astropy.io.ascii.CommentedHeader)
        reader.header.splitter.delimiter = '|'
        chemistry = reader.read(os.path.join(self.folder, f"Chemistry_{index:05d}"))
        physics = reader.read(os.path.join(self.folder, f"physical_structure_{index:05d}"))
        config = self._config_read(os.path.join(self.folder, "../config.ini"))
        table = CTable(
            [
                physics["Radius"] << u.au,
                physics["Height"] << u.au,
            ],
            names=["Radius", "Height"]
        )
        if "Relative Height" in chemistry.colnames:
            table["Height to radius"] = chemistry["Relative Height"]
        else:
            table["Height to radius"] = chemistry["Height"] / chemistry["Radius"]
        table["Gas density"] = physics["Gas density"] << (u.g / u.cm ** 3)
        table["Dust density"] = physics["Dust density"] << (u.g / u.cm ** 3)
        table["Gas temperature"] = physics["Gas temperature"] << (u.K)
        table["Dust temperature"] = physics["Dust temperature"] << (u.K)
        table["n(H+2H2)"] = (chemistry["H+"] + chemistry["H"] + 2 * chemistry["H2"]) << (u.cm ** (-3))
        for species in chemistry.colnames[3:]:
            table[species] = (chemistry[species] << (u.cm ** (-3))) / table["n(H+2H2)"]

        self.physics = diskchef.physics.base.PhysicsBase(
            star_mass=config["stellar mass [MSun]"] * u.solMass,
        )
        self.physics.table = physics

        return table

    def __post_init__(self):
        super().__post_init__()
        self._table = self.read()

    def _config_read(self, path: PathLike) -> dict:
        out = {}
        with open(path, 'r') as configfile:
            for line in configfile.readlines():
                value, key = [entry.strip() for entry in line.split('::')]
                out[key] = value
        return out

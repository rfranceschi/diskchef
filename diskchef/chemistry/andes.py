import logging
from dataclasses import dataclass
import os

import astropy.table
import diskchef.engine.exceptions
import diskchef.maps.radiation_fields
import numpy as np
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
    Class to read ANDES2 output for following usage in `diskchef.maps`
    """
    folder: PathLike = None
    index: int = 0
    read_uv: bool = False
    read_ionization: bool = False

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
        # if physics[["ir", "iz"]] != chemistry[["ir", "iz"]]:
        #     raise diskchef.engine.exceptions.CHEFRuntimeError("Physics and chemistry tables do not match!")
        config = self._config_read(os.path.join(self.folder, "../config.ini"))
        table = CTable(
            [
                physics["Radius"] << u.au,
                physics["Height"] << u.au,
            ],
            names=["Radius", "Height"]
        )
        table["Height to radius"] = np.nan * u.dimensionless_unscaled
        for izr in physics["iz"]:
            indices = physics["iz"] == izr
            table["Height to radius"][indices] = np.nanmedian(physics["Height"][indices] / physics["Radius"][indices])

        table["Height"] = table.zr * table.r
        table["Gas density"] = physics["Gas density"] << (u.g / u.cm ** 3)
        table["Dust density"] = physics["Dust density"] << (u.g / u.cm ** 3)
        table["Gas temperature"] = physics["Gas temperature"] << (u.K)
        table["Dust temperature"] = physics["Dust temperature"] << (u.K)
        table["n(H+2H2)"] = (chemistry["H+"] + chemistry["H"] + 2 * chemistry["H2"]) << (u.cm ** (-3))
        for species in chemistry.colnames[3:]:
            table[species] = (chemistry[species] << (u.cm ** (-3))) / table["n(H+2H2)"]

        if self.read_uv:
            try:
                radiation = reader.read(os.path.join(self.folder, f"RadInt_StrengthTotal_{index:05d}"))
            except FileNotFoundError:
                self.logger.info("Radiation strength file %:05d not found! Trying 00000 instead", index)
                radiation = reader.read(os.path.join(self.folder, f"RadInt_StrengthTotal_00000"))
            if np.any(radiation[["ir", "iz"]] != physics[["ir", "iz"]]):
                raise diskchef.engine.exceptions.CHEFRuntimeError("Physics and radiation tables do not match!")
            table["G_UV"] = radiation["G_UV"] << diskchef.maps.radiation_fields.ANDES2_G0

        if self.read_ionization:
            try:
                ionization = astropy.table.Table.read(os.path.join(self.folder, f"Ionization_Rate_{index:05d}"))
            except FileNotFoundError:
                self.logger.info("Ionization rate file %:05d not found! Trying 00000 instead", index)
                ionization = astropy.table.Table.read(os.path.join(self.folder, f"Ionization_Rate_00000"),
                                                      format="ascii.basic")
            if np.any(ionization[["ir", "iz"]] != physics[["ir", "iz"]]):
                raise diskchef.engine.exceptions.CHEFRuntimeError("Physics and ionization tables do not match!")
            table["X ray ionization rate"] = ionization["IR_XR[1/s]"] << u.s ** (-1)
            table["CR ionization rate"] = ionization["IR_CR[1/s]"] << u.s ** (-1)
            table["Ionization rate"] = ionization["IR_Total[1/s]"] << u.s ** (-1)

        self.physics = diskchef.physics.base.PhysicsBase(
            star_mass=float(config['stellar mass [MSun]']) * u.solMass,
        )
        self.physics.table = table

        return table

    def __post_init__(self):
        super().__post_init__()
        self._table = self.read()
        self.update_hydrogen_atom_number_density()

    def _config_read(self, path: PathLike) -> dict:
        out = {}
        with open(path, 'r') as configfile:
            for line in configfile.readlines():
                value, key = [entry.strip() for entry in line.split('::')]
                out[key] = value
        return out

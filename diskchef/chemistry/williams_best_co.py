from dataclasses import dataclass

from astropy import units as u

from diskchef.chemistry.base import ChemistryBase


@dataclass
class ChemistryWB2014(ChemistryBase):
    """Calculates chemistry according to Williams & Best 2014

    Eq. 5-7
    https://iopscience.iop.org/article/10.1088/0004-637X/788/1/59/pdf
    """
    midplane_co_abundance = 0
    molecular_layer_co_abundance = 1e-4
    co_freezeout_temperature = 20 * u.K
    atmosphere_co_abundance = 0
    h2_column_denisty_that_shields_co = 1.3e21 / u.cm ** 2

    def run_chemistry(self):
        self.table["H2"] = 0.5
        self.table["H2 number density"] = self.table["H2"] * self.table["n(H+2H2)"]
        self.table["H2 column density towards star"] = self.physics.column_density_to(
            self.table.r, self.table.z,
            "H2 number density"
        )

        self.table["CO"] = self.midplane_co_abundance
        self.table["CO"][
            self.table["Gas temperature"] > self.co_freezeout_temperature
            ] = self.molecular_layer_co_abundance

        self.table["CO"][
            self.table["H2 column density towards star"] < self.h2_column_denisty_that_shields_co
            ] = self.atmosphere_co_abundance

@dataclass
class NonzeroChemistryWB2014(ChemistryWB2014):
    """Subclass of ChemistryWB2014 that has non-zero default midplane and atmosphere abundance"""
    midplane_co_abundance = 1e-6
    atmosphere_co_abundance = 1e-10
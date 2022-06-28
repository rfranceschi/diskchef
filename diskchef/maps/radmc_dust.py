from datetime import timedelta
import subprocess
import time
import shutil
import re
import os
from dataclasses import dataclass, field
from typing import Union

import diskchef.engine.exceptions
import numpy as np
from astropy import units as u
from astropy import constants as c

import diskchef.physics.yorke_bodenheimer
from diskchef.engine.other import PathLike
from diskchef.maps.radmcrt import RadMCBase


@dataclass
class RadMCTherm(RadMCBase):
    star_radius: Union[None, u.Quantity] = None
    star_effective_temperature: Union[None, u.Quantity] = None
    accretion_luminosity: Union[None, u.Quantity] = None
    accretion_temperature: Union[None, u.Quantity] = 2e4 * u.K

    def __post_init__(self):
        super().__post_init__()
        self.dust_species = self.chemistry.table.dust_list
        yb08 = diskchef.physics.yorke_bodenheimer.YorkeBodenheimer2008()
        if (self.star_radius is None) and (self.star_effective_temperature is not None):
            self.star_radius = yb08.radius(self.chemistry.physics.star_mass)
            self.logger.warn(
                "Only star effective temperature was given, not star radius! Taking radius from trktime0210")
        elif (self.star_radius is not None) and (self.star_effective_temperature is None):
            self.star_effective_temperature = yb08.effective_temperature(self.chemistry.physics.star_mass)
            self.logger.warn(
                "Only  star radius was given, not effective temperature! Taking effective temperature from trktime0210")
        elif (self.star_radius is None) and (self.star_effective_temperature is None):
            self.star_radius = yb08.radius(self.chemistry.physics.star_mass)
            self.star_effective_temperature = yb08.effective_temperature(self.chemistry.physics.star_mass)
            self.logger.info("Taking effective temperature and radius from trktime0210")

    @property
    def mode(self) -> str:
        """Mode of RadMC: `mctherm`, `image`, `spectrum`, `sed`"""
        return "mctherm"

    def create_files(self) -> None:
        super().create_files()
        self.dustopac()
        self.dust_density()
        self.stars()
        self.external_source()

    def stars(self, out_file: PathLike = None) -> None:
        """Writes the `stars.inp` file"""
        if out_file is None:
            out_file = os.path.join(self.folder, 'stars.inp')
        with open(out_file, 'w') as file:
            if self.accretion_luminosity is None:
                print('2', file=file)  # Typically 2 at present
                print(f'1 {len(self.wavelengths)}', file=file)  # number of stars, wavelengths
                print(f'{self.star_radius.to(u.cm).value} '
                      f'{self.chemistry.physics.star_mass.to(u.g).value} '
                      f'0 0 0',
                      file=file)
                print('\n'.join(f"{entry:.7e}" for entry in self.wavelengths.to_value(u.um)), file=file)
                print(-self.star_effective_temperature.to(u.K).value, file=file)
            else:
                print('2', file=file)  # Typically 2 at present
                print(f'2 {len(self.wavelengths)}', file=file)
                print(f'{self.star_radius.to(u.cm).value} '
                      f'{self.chemistry.physics.star_mass.to(u.g).value} '
                      f'0 0 0',
                      file=file)
                print(f'{self.accretion_effective_radius.to_value(u.cm)} '
                      f'0 '
                      f'0 0 0',
                      file=file)
                print('\n'.join(f"{entry:.7e}" for entry in self.wavelengths.to_value(u.um)), file=file)
                print(-self.star_effective_temperature.to(u.K).value, file=file)
                print(-self.accretion_temperature.to(u.K).value, file=file)

    @property
    def accretion_effective_radius(self):
        return np.sqrt(self.accretion_luminosity / (
                4 * np.pi * c.sigma_sb * self.accretion_temperature ** 4)).cgs

    def dustopac(self, out_file: PathLike = None) -> None:
        """Writes the `dustopac.inp` file"""
        if out_file is None:
            out_file = os.path.join(self.folder, 'dustopac.inp')
        with open(out_file, 'w') as file:
            print('2', file=file)  # Typically 2 at present
            print(len(self.dust_species), file=file)  # Number of species
            for dust_spice in self.dust_species:
                print('-----------------------------', file=file)
                type = "dustkapscatmat_" if os.path.basename(dust_spice.opacity_file).startswith(
                    "dustkapscatmat_") else "dustkappa_"
                print('10' if type == "dustkapscatmat_" else '1', file=file)
                print('0', file=file)
                name_without_whitespaces = re.sub(r"\s*", "", dust_spice.name)
                print(name_without_whitespaces, file=file)
                shutil.copyfile(dust_spice.opacity_file, os.path.join(self.folder, f"{type}{name_without_whitespaces}.inp"))
            print('-----------------------------', file=file)

    def dust_density(self, out_file: PathLike = None) -> None:
        """Writes the dust density file"""
        if out_file is None:
            out_file = os.path.join(self.folder, 'dust_density.inp')
        self.interpolate("Dust density")
        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print(self.nrcells, file=file)
            print(len(self.dust_species), file=file)  # Number of species
            for dust_spice in self.dust_species:
                self.interpolate(f"{dust_spice.name} mass fraction")
                print(
                    '\n'.join(
                        f"{entry:.7e}" for entry
                        in (
                                self.polar_table["Dust density"]
                                * self.polar_table[f"{dust_spice.name} mass fraction"]
                        ).to(u.g * u.cm ** (-3)).value),
                    file=file
                )

    @u.quantity_input
    def run(
            self,
            inclination: u.deg = 0 * u.deg, position_angle: u.deg = 0 * u.deg,
            distance: u.pc = 140 * u.pc, velocity_offset: u.km / u.s = 0 * u.km / u.s,
            threads: int = 1, npix: int = 100,
    ) -> None:
        self.logger.info("Running radmc3d")
        start = time.time()
        command = (f"{self.executable} {self.mode} "
                   f"setthreads {threads} "
                   )
        self.logger.info("Running radmc3d for dust temperature calculation: %s", command)
        proc = subprocess.run(
            command,
            cwd=self.folder,
            text=True,
            capture_output=True,
            shell=True
        )
        self.logger.info("radmc3d finished after %s", timedelta(seconds=time.time() - start))
        self.catch_radmc_messages(proc)

    def read_dust_temperature(self) -> None:
        """Read RadMC3D dust temperature output into `table`

        Current limitations:

        * Only one dust population
        """
        self.polar_table["RadMC Dust temperature"] = np.loadtxt(
            os.path.join(self.folder, "dust_temperature.dat")
        )[3:] << u.K
        number_of_zeroes = np.sum(self.polar_table["RadMC Dust temperature"] == 0)
        if number_of_zeroes != 0:
            self.logger.error("RadMC never achieved %d points. "
                              "Recalculate with higher nphot_therm "
                              "and/or remove high-density regions "
                              "out of the model grid "
                              "(highly likely, you can forfeit radii ~< 1 au)."
                              "Values are set to 2.7 K", number_of_zeroes)
            self.polar_table["RadMC Dust temperature"][self.polar_table["RadMC Dust temperature"] == 0] = 2.7 * u.K
        self.interpolate_back("RadMC Dust temperature")
        self.table["Original Dust temperature"] = self.table["Dust temperature"]
        self.table["Dust temperature"] = self.table["RadMC Dust temperature"]
        self.table.check_zeros("Dust temperature")


@dataclass
class RadMCThermMono(RadMCTherm):
    """Class to run `radmc3d mctherm` and `radmc3d mcmono` one after another"""
    wavelengths: u.Quantity = field(default=np.geomspace(0.01, 1000, 125) * u.um)
    mcmono_wavelengths: u.Quantity = field(default=np.geomspace(0.0912, 0.2, 20) * u.um)

    @u.quantity_input
    def run(
            self,
            inclination: u.deg = 0 * u.deg, position_angle: u.deg = 0 * u.deg,
            distance: u.pc = 140 * u.pc, velocity_offset: u.km / u.s = 0 * u.km / u.s,
            threads: int = 1, npix: int = 100,
    ) -> None:
        super().run(
            inclination=inclination, position_angle=position_angle,
            distance=distance, velocity_offset=velocity_offset,
            threads=threads, npix=npix
        )

        self.logger.info("Running radmc3d")
        start = time.time()
        command = (f"{self.executable} mcmono "
                   f"setthreads {threads} "
                   )
        self.logger.info("Running radmc3d for radiation field: %s", command)
        proc = subprocess.run(
            command,
            cwd=self.folder,
            text=True,
            capture_output=True,
            shell=True
        )
        self.logger.info("radmc3d finished after %s", timedelta(seconds=time.time() - start))
        self.catch_radmc_messages(proc)

    def mcmono_wavelength_micron(self, out_file: PathLike = None) -> None:
        """Creates a `mcmono_wavelength_micron.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'mcmono_wavelength_micron.inp')

        with open(out_file, 'w') as file:
            print(len(self.mcmono_wavelengths), file=file)
            print('\n'.join(f"{entry.to(u.um).value:.7e}" for entry in self.mcmono_wavelengths), file=file)

    def create_files(self) -> None:
        super().create_files()
        self.mcmono_wavelength_micron()

    def read_radiation_strength(self) -> None:
        """Read RadMC3D mcmono radiation strength output into `table`"""
        radiation = np.loadtxt(
            os.path.join(self.folder, "mean_intensity.out"), skiprows=4
        ).reshape(-1, len(self.mcmono_wavelengths), order="F") << (u.erg / u.s / u.cm ** 2 / u.Hz / u.sr)

        self.polar_table["Radiation strength"] = np.abs(
            4 * np.pi * u.sr * np.trapz(
                radiation, self.mcmono_wavelengths.to(u.Hz, equivalencies=u.spectral()),
                axis=1)
        ).to(u.erg / u.s / u.cm ** 2)

        number_of_zeroes = np.sum(radiation == 0)
        if number_of_zeroes != 0:
            self.logger.warning("RadMC never achieved %d points. "
                                "Recalculate with higher nphot_therm "
                                "and/or remove high-density regions "
                                "out of the model grid "
                                "(highly likely, you can forfeit radii ~< 1 au). ", number_of_zeroes)

        self.interpolate_back("Radiation strength")
        self.table["Radiation strength"][self.table["Radiation strength"] < 0] = 0
        self.table.check_zeros("Radiation strength")

import glob
from dataclasses import dataclass, field

import numpy as np
import os
import radmc3dPy
import re
import shutil
import subprocess
import time
import typing
from astropy import constants as c
from astropy import units as u
from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from typing import Union

import diskchef.physics
import diskchef.physics.yorke_bodenheimer
from diskchef.engine.ctable import CTable
from diskchef.engine.exceptions import CHEFNotImplementedError, CHEFTypeError
from diskchef.engine.other import PathLike
from diskchef.lamda.line import Line
from diskchef.maps.base import MapBase


@dataclass
class RadMCOutput:
    """Class to store information about the RadMCRT module output"""
    line: Line
    file_radmc: PathLike = None
    file_fits: PathLike = None


@dataclass
class RadMCBase(MapBase):
    executable: PathLike = 'radmc3d'
    folder: PathLike = 'radmc'
    radii_bins: Union[None, int] = None
    theta_bins: Union[None, int] = None
    outer_radius: Union[None, u.Quantity] = None
    verbosity: int = 0
    wavelengths: u.Quantity = field(default=np.geomspace(0.1, 1000, 100) * u.um)
    modified_random_walk: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.table.is_in_zr_regular_grid:
            raise CHEFNotImplementedError

        try:
            os.mkdir(self.folder)
        except FileExistsError:
            self.logger.warn("Directory %s already exists! The results can be biased.", self.folder)

        if self.radii_bins is None:
            radii = np.sort(np.unique(self.table.r)).to(u.cm)
            if self.outer_radius is not None:
                radii = radii[radii <= self.outer_radius]
                self._check_outer_radius()
        elif isinstance(self.radii_bins, int):
            if self.outer_radius is None:
                outer_radius = self.table.r.max()
            else:
                outer_radius = self.outer_radius
                self._check_outer_radius()
            radii = np.geomspace(self.table.r.min(), outer_radius, self.radii_bins).to(u.cm)
        else:
            raise CHEFTypeError("radii_bins should be None or int, not %s (%s)", type(self.radii_bins), self.radii_bins)

        if self.theta_bins is None:
            zr = np.sort(np.unique(self.table.zr))
        elif isinstance(self.theta_bins, int):
            zr = np.linspace(self.table.zr.min(), self.table.zr.max(), self.theta_bins)
        else:
            raise CHEFTypeError("theta_bins should be None or int, not %s (%s)", type(self.theta_bins), self.theta_bins)

        theta = np.pi / 2 - np.arctan(zr)
        self.radii_edges = u.Quantity([radii[0], *np.sqrt(radii[1:] * radii[:-1]), radii[-1]]).value
        self.zr_edges = np.array([zr[0], *(0.5 * (zr[1:] + zr[:-1])), zr[-1]])
        self.theta_edges = np.sort(np.pi / 2 - np.arctan(self.zr_edges))

        R, THETA = np.meshgrid(radii, theta)
        self.polar_table = CTable()
        self.polar_table['Distance to star'] = R.flatten()
        self.polar_table['Theta'] = THETA.flatten() << u.rad
        self.polar_table['Altitude'] = (np.pi / 2 * u.rad - self.polar_table['Theta']) << u.rad
        self.polar_table['Height'] = self.polar_table['Distance to star'] * np.sin(self.polar_table['Altitude'])
        self.polar_table['Radius'] = self.polar_table['Distance to star'] * np.cos(self.polar_table['Altitude'])
        self.polar_table.sort(['Theta', 'Distance to star'])
        self.interpolate('n(H+2H2)')
        self.polar_table['Velocity R'] = 0 * u.cm / u.s
        self.polar_table['Velocity Theta'] = 0 * u.cm / u.s
        self.polar_table['Velocity Phi'] = \
            np.sqrt(c.G * self.chemistry.physics.star_mass / self.polar_table['Distance to star']).to(u.cm / u.s)

        self.nrcells = (len(self.radii_edges) - 1) * (len(self.theta_edges) - 1)
        self.fitsfiles = []
        """List of created FITS files"""
        self.outputs = {}

    def _check_outer_radius(self):
        pass
        # total_mass = np.trapz(self.table.r * np.trapz(self.table["Gas density"], self.table.z))
        # total_mass_outside_outer_radius = np.trapz(self.table.r * )
        # self.logger.warning("The outer radius %s contains only %s per cent of the total disk mass", self.outer_radius)

    @property
    def mode(self) -> str:
        """Mode of RadMC: `mctherm`, `image`, `spectrum`, `sed`"""
        raise NotImplementedError

    def catch_radmc_messages(self, proc: subprocess.CompletedProcess) -> None:
        """Raises RadMC warnings and errors in `self.logger`"""
        if proc.stderr: self.logger.error(proc.stderr)
        self.logger.debug(proc.stdout)
        for match in re.finditer(r"WARNING:(.*\n(?:  .*\n){2,})", proc.stdout):
            self.logger.warn(match.group(1))

    def interpolate(self, column: str) -> None:
        """Adds a new `column` to `self.polar_table` with the data iterpolated from `self.table`"""
        self.polar_table[column] = self.table.interpolate(column)(self.polar_table.r, self.polar_table.z)

    def create_files(self) -> None:
        """Creates all the files necessary to run RadMC3D"""
        self.radmc3d()
        self.wavelength_micron()
        self.amr_grid()
        self.logger.info("Files written to %s", self.folder)

    def radmc3d(self, out_file: PathLike = None) -> None:
        """Creates an empty `radmc3d.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'radmc3d.inp')

        with open(out_file, 'w') as file:
            if self.modified_random_walk:
                print("modified_random_walk = 1", file=file)

    def wavelength_micron(self, out_file: PathLike = None) -> None:
        """Creates a `wavelength_micron.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'wavelength_micron.inp')

        with open(out_file, 'w') as file:
            print(len(self.wavelengths), file=file)
            print('\n'.join(f"{entry.to(u.um).value:.7e}" for entry in self.wavelengths), file=file)

    def amr_grid(self, out_file: PathLike = None) -> None:
        """Creates a `amr_grid.inp` file"""

        if not self.table.is_in_zr_regular_grid:
            raise CHEFNotImplementedError

        if out_file is None:
            out_file = os.path.join(self.folder, 'amr_grid.inp')
        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print('0', file=file)  # Grid style (regular = 0)
            print('100', file=file)  # Spherical coordinate system
            print('1' if self.verbosity else '0', file=file)  # Grid info
            print('1 1 0', file=file)  # Included coordinates
            print(len(self.radii_edges) - 1, len(self.theta_edges) - 1, 1, file=file)
            print(' '.join(f"{entry:.7e}" for entry in self.radii_edges), file=file)
            print(' '.join(f"{entry:.7e}" for entry in self.theta_edges), file=file)
            print(0, 2 * np.pi, file=file)

    @u.quantity_input
    def run(
            self,
            inclination: u.deg = 0 * u.deg, position_angle: u.deg = 0 * u.deg,
            distance: u.pc = 140 * u.pc, velocity_offset: u.km / u.s = 0 * u.km / u.s,
            threads: int = 1, npix: int = 100,
    ) -> None:
        """Run RadMC3D after files were created with `create_files()`"""
        raise CHEFNotImplementedError


@dataclass
class RadMCTherm(RadMCBase):
    star_radius: Union[None, u.Quantity] = None
    star_effective_temperature: Union[None, u.Quantity] = None

    def __post_init__(self):
        super().__post_init__()
        self.dust_species = self.chemistry.table.meta["Dust list"]
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

    def stars(self, out_file: PathLike = None) -> None:
        """Writes the `stars.inp` file"""
        if out_file is None:
            out_file = os.path.join(self.folder, 'stars.inp')
        with open(out_file, 'w') as file:
            print('2', file=file)  # Typically 2 at present
            print(f'1 {len(self.wavelengths)}', file=file)  # number of stars, wavelengths
            print(f'{self.star_radius.to(u.cm).value} '
                  f'{self.chemistry.physics.star_mass.to(u.g).value} '
                  f'0 0 0',
                  file=file)
            print('\n'.join(f"{entry.to(u.um).value:.7e}" for entry in self.wavelengths), file=file)
            print(-self.star_effective_temperature.to(u.K).value, file=file)

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
                shutil.copy(dust_spice.opacity_file, os.path.join(self.folder, f"{type}{name_without_whitespaces}.inp"))
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
        self.polar_table["RadMC Dust temperature"] = np.loadtxt(
            os.path.join(self.folder, "dust_temperature.dat")
        )[3:] << u.K
        self.logger.info("radmc3d finished after %s", timedelta(seconds=time.time() - start))
        self.catch_radmc_messages(proc)


@dataclass
class RadMCRT(RadMCBase):
    """
    Class with interface to run RadMC3D by Cornelis Dullemond

    By initialization creates the basic table (`self.polar_table`) with the grid in polar RadMC3D coordinates.
    """

    @property
    def mode(self) -> str:
        """Mode of RadMC: `mctherm`, `image`, `spectrum`, `sed`"""
        return 'image'

    def create_files(self) -> None:
        """Creates all the files necessary to run RadMC3D"""
        super().create_files()
        self.lines()
        self.gas_velocity()
        self.gas_temperature()

        for molecule in self.molecules_list:
            self.numberdens(species=molecule)
            self.molecule(species=molecule, file=self.lamda_files_dict[molecule])
        self.logger.info("Line files written to %s", self.folder)

    def gas_temperature(self, out_file: PathLike = None) -> None:
        """Writes the gas temperature files"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'gas_temperature.inp')

        self.interpolate('Gas temperature')

        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print(self.nrcells, file=file)
            print('\n'.join(f"{entry:.7e}" for entry in self.polar_table['Gas temperature'].to(u.K).value), file=file)

    def numberdens(self, species: str, out_file: PathLike = None) -> None:
        """Writes the gas number density files"""

        if out_file is None:
            out_file = os.path.join(self.folder, f'numberdens_{species}.inp')

        self.interpolate(species)

        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print(self.nrcells, file=file)
            print(
                '\n'.join(
                    f"{entry:.7e}" for entry
                    in self.polar_table['n(H+2H2)'].to(u.cm ** (-3)).value * self.polar_table[species]),
                file=file
            )

    def molecule(self, species: str, lamda_file: PathLike,  out_file: PathLike = None) -> None:
        """
        Copies the molecule transition files into working directory

        species:    str     name of the molecule
        """

        if out_file is None:
            out_file = os.path.join(self.folder, f'molecule_{species}.inp')

        shutil.copy(lamda_file, out_file)

    def lines(self, out_file: PathLike = None) -> None:
        self.molecules_list = sorted(list(set([line.molecule for line in self.line_list])))
        self.lamda_files_dict = {
            line.molecule: line.lamda_file for line in self.line_list
        }

        if out_file is None:
            out_file = os.path.join(self.folder, 'lines.inp')
        with open(out_file, 'w') as file:
            print('2', file=file)  # 2 for molecules
            print(len(self.molecules_list), file=file)  # unique list of molecules

            for molecule in self.molecules_list:
                coll_partners = next(line for line in self.line_list if line.molecule == molecule).collision_partner
                print(f'{molecule} leiden 0 0 {len(coll_partners)}', file=file)
                print('\n'.join(coll_partners), file=file)

    def gas_velocity(self, out_file: PathLike = None) -> None:
        if out_file is None:
            out_file = os.path.join(self.folder, 'gas_velocity.inp')
        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print(self.nrcells, file=file)
            for vr, vtheta, vphi in zip(self.polar_table['Velocity R'].to(u.cm / u.s).value,
                                        self.polar_table['Velocity Theta'].to(u.cm / u.s).value,
                                        self.polar_table['Velocity Phi'].to(u.cm / u.s).value):
                print(f"{vr:.7e}", f"{vtheta:.7e}", f"{vphi:.7e}", file=file)

    @u.quantity_input
    def run(
            self,
            inclination: u.deg = 0 * u.deg, position_angle: u.deg = 0 * u.deg,
            distance: u.pc = 140 * u.pc, velocity_offset: u.km / u.s = 0 * u.km / u.s,
            threads: int = 1, npix: int = 100,
    ) -> None:
        """Run RadMC3D after files were created with `create_files()`"""
        self.logger.info("Running radmc3d")
        for line in self.line_list:
            self._run_single(
                molecule=self.molecules_list.index(line.molecule) + 1,
                line=line.transition,
                inclination=inclination.to(u.deg).value, position_angle=position_angle.to(u.deg).value,
                velocity_offset=velocity_offset.to(u.km / u.s).value,
                name=f"{line.name}_image.out",
                distance=distance.to(u.pc).value,
                threads=threads,
                lineobj=line,
                npix=npix,
            )

    def _run_single(
            self, molecule: int, line: int, inclination: float = 0, position_angle: float = 0,
            name: PathLike = None, distance: float = 140, velocity_offset: float = 0,
            n_channels: int = 100, threads: int = 1, lineobj: Line = None, npix: int = 100
    ) -> None:
        start = time.time()
        command = (f"{self.executable} {self.mode} "
                   f"imolspec {molecule} "
                   f"iline {line} "
                   f"widthkms 10 "
                   f"incl {inclination} "
                   f"posang {position_angle} "
                   f"vkms {velocity_offset} "
                   f"linenlam {n_channels} "
                   f"setthreads {threads} "
                   f"npix {npix} "
                   )
        self.logger.info("Running radmc3d for molecule %d and transition %d: %s", molecule, line, command)
        proc = subprocess.run(
            command,
            cwd=self.folder,
            text=True,
            capture_output=True,
            shell=True
        )
        self.logger.info("radmc3d finished after %s", timedelta(seconds=time.time() - start))
        self.catch_radmc_messages(proc)
        output = RadMCOutput(lineobj)
        if name is not None:
            newname = os.path.join(self.folder, name)
            shutil.move(os.path.join(self.folder, "image.out"), newname)
            output.file_radmc = newname
            output.file_fits = self.radmc_to_fits(newname, self.line_list[line], distance)
        self.outputs[lineobj] = output

    def radmc_to_fits(self, name: PathLike, line: Line, distance: float) -> PathLike:
        """Saves RadMC3D `image.out` files as FITS files

        Returns:
            name of a newly created fits files
        """
        im = radmc3dPy.image.readImage(fname=name)
        fitsname = name.replace(".out", ".fits")
        restfreq = line.frequency.to(u.Hz).value
        if os.path.exists(fitsname):
            os.remove(fitsname)
        im.writeFits(fname=fitsname, nu0=restfreq, dpc=distance,
                     fitsheadkeys={"CUNIT3": "Hz", "BUNIT": "Jy pix**(-1)", "RESTFREQ": restfreq})
        self.fitsfiles.append(fitsname)
        self.logger.info("Saved as %s and %s", name, fitsname)
        self.logger.debug("Modified FITS header unit: HZ to Hz, JY/PX to Jy pix**(-1), set RESTFREQ %s Hz",
                          restfreq)
        return fitsname

    def copy_for_propype(self, folder: PathLike = None) -> None:
        """Creates a copy of `self.fitsfiles` in a format which is ready to run them through PRODIGE pipeline"""
        if folder is None:
            folder = os.path.join(self.folder, "propype")
        try:
            os.mkdir(folder)
        except FileExistsError as e:
            self.logger.debug("%s", e)
        for line, output in self.outputs.items():
            molecule = line.molecule
            fitsfile = output.file_fits
            transition = output.line.transition

            shutil.copy(fitsfile, os.path.join(folder, f"m-Line-00-{molecule}_{transition}+D.fits"))


@dataclass
class RadMCVisualize:
    folder: PathLike = '.'
    line: Line = None

    def channel_map(self) -> None:
        for file in sorted(glob.glob(os.path.join(self.folder, "*image.out"))):
            im = radmc3dPy.image.readImage(fname=file)
            normalizer = Normalize(vmin=im.image.min(), vmax=im.image.max())
            nrow = 5
            ncol = 8
            fig, axes = plt.subplots(
                nrow, ncol,
                sharey='row', sharex='col',
                gridspec_kw=dict(wspace=0.0, hspace=0.0,
                                 top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                 left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1)),
                figsize=(ncol + 1, nrow + 1)
            )
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(im.image[:, :, i], norm=normalizer)

            axes_f = axes.flatten()
            cbaxes = fig.add_axes(
                [axes_f[-1].figbox.x1, axes_f[-1].figbox.y0,
                 0.1 * (axes_f[-1].figbox.x1 - axes_f[-1].figbox.x0),
                 axes_f[-1].figbox.y1 - axes_f[-1].figbox.y0]
            )
            plt.subplots_adjust(hspace=0.01, wspace=0.01)
            cbim = axes_f[-1].images[0]
            clbr = fig.colorbar(cbim, cax=cbaxes)
            fig.suptitle(file)
            fig.savefig(file + ".png")


@dataclass
class RadMCRTSingleCall(RadMCRT):
    """
    Subclass of RadMCRT to run RadMC only once for all the required frequencies

    Usage:

    >>> physics = diskchef.physics.WilliamsBest2014(star_mass=0.52 * u.solMass, radial_bins=10, vertical_bins=10)
    >>> chem = diskchef.chemistry.NonzeroChemistryWB2014(physics)
    >>> chem.run_chemistry()
    >>> mapping = RadMCRTSingleCall(chemistry=chem, line_list=[
    ...     Line(name='CO J=2-1', transition=1, molecule='CO'),
    ...     Line(name='CO J=3-2', transition=2, molecule='CO'),
    ... ])
    >>> mapping.wavelength_multiple_lines(channels_per_line=5, window_width=2 * u.km/u.s)  # doctest: +NORMALIZE_WHITESPACE
    array([2600.76630869, 2600.76197107, 2600.75763346, 2600.75329588,
           2600.7489583 , 1300.40799349, 1300.40582464, 1300.4036558 ,
           1300.40148696, 1300.39931813])
    """

    def frequency_centers(self) -> typing.List[u.Quantity]:
        """Fetches frequencies of lines from `self.line_list`"""
        for line in self.line_list:
            line.parse_lamda()
        self.ordered_line_list = sorted(self.line_list, key=lambda line: line.frequency)
        frequency_centers = [line.frequency for line in self.ordered_line_list]
        return frequency_centers

    def wavelength_multiple_lines(self, channels_per_line: int, window_width: u.km / u.s) -> np.ndarray:
        """
        Return a sorted array of wavelength for all the lines, in um (unitless)
        """
        frequencies_list = []
        self.channels_per_line = []
        for frequency_center in self.frequency_centers():
            width = np.abs(window_width.to(u.GHz, equivalencies=u.doppler_radio(frequency_center)) - frequency_center)
            frequencies_list.append(
                np.linspace(frequency_center - width / 2, frequency_center + width / 2, channels_per_line)
            )
            self.channels_per_line.append(channels_per_line)
        frequencies = np.hstack(frequencies_list)
        wavelengths = frequencies.to(u.um, equivalencies=u.spectral()).value
        return wavelengths

    @u.quantity_input
    def camera_wavelength_micron(
            self,
            out_file: PathLike = None,
            window_width: u.km / u.s = 6 * u.km / u.s,
            channels_per_line: int = 200
    ) -> None:
        """Creates a `camera_wavelength_micron.inp` file with all the frequencies for all the lines"""
        wavelengths = self.wavelength_multiple_lines(channels_per_line, window_width)
        if out_file is None:
            out_file = os.path.join(self.folder, 'camera_wavelength_micron.inp')
        with open(out_file, 'w') as file:
            print(len(wavelengths), file=file)
            print('\n'.join(f"{entry:.10e}" for entry in wavelengths), file=file)

    def create_files(self,
                     window_width: u.km / u.s = 6 * u.km / u.s,
                     channels_per_line: int = 200
                     ) -> None:
        super().create_files()
        self.camera_wavelength_micron(window_width=window_width, channels_per_line=channels_per_line)

    @u.quantity_input
    def run(
            self,
            inclination: u.deg = 0 * u.deg, position_angle: u.deg = 0 * u.deg,
            distance: u.pc = 140 * u.pc, velocity_offset: u.km / u.s = 0 * u.km / u.s,
            threads: int = 1, npix: int = 100
    ) -> None:
        self.logger.info("Running radmc3d")
        start = time.time()
        command = (f"{self.executable} {self.mode} "
                   f"incl {inclination.to(u.deg).value} "
                   f"posang {position_angle.to(u.deg).value} "
                   f"setthreads {threads} "
                   "loadlambda "
                   f"npix {npix} "
                   )
        self.logger.info("Running radmc3d for all transition at once: %s", command)
        proc = subprocess.run(
            command,
            cwd=self.folder,
            text=True,
            capture_output=True,
            shell=True
        )
        self.logger.info("radmc3d finished after %s", timedelta(seconds=time.time() - start))
        self.catch_radmc_messages(proc)
        names = [os.path.join(self.folder, f"{line.name}_image.out") for line in self.ordered_line_list]
        self.split(names=names)
        for line, name in zip(self.ordered_line_list, names):
            self.outputs[line] = RadMCOutput(line, file_radmc=name)
            self.outputs[line].file_fits = self.radmc_to_fits(name, line, distance.to(u.pc).value)

    def split(
            self, filename=None,
            names: typing.List[PathLike] = None
    ):
        """Split the `image.out` file with all lines into multiple single-frequency files"""
        lengths = self.channels_per_line
        if filename is None:
            filename = os.path.join(self.folder, 'image.out')

        with open(filename, 'r') as bigfile:
            files = [open(name, 'w') for name in names]
            header = [next(bigfile) for _ in range(4)]
            px_number = [int(entry) for entry in header[1].split()]
            self.logger.info("Splitting %s in %s", filename, names)
            for file, length in zip(files, lengths):
                file.write(header[0])
                file.write(header[1])
                file.write(str(length) + "\n")
                file.write(header[3])
                for line in range(length):
                    file.write(next(bigfile))

            for file, length in zip(files, lengths):
                for line in range(length * (px_number[0] * px_number[1] + 1)):
                    file.write(next(bigfile))

            for file in files:
                file.close()

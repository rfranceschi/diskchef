import typing
from datetime import timedelta
import subprocess
import time
import shutil
import os
from dataclasses import dataclass

import astropy.coordinates
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from spectral_cube import SpectralCube

import radmc3dPy

import diskchef
from diskchef.engine.other import PathLike
from diskchef.lamda.line import Line
from diskchef.maps.radmcrt import RadMCBase, RadMCOutput


@dataclass
class RadMCRT(RadMCBase):
    """
    Class with interface to run RadMC3D by Cornelis Dullemond

    By initialization creates the basic table (`self.polar_table`) with the grid in polar RadMC3D coordinates.
    """

    def __post_init__(self):
        super().__post_init__()
        self.interpolate("n(H+2H2)")

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
            self.molecule(species=molecule, lamda_file=self.lamda_files_dict[molecule])
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

    def molecule(self, species: str, lamda_file: PathLike, out_file: PathLike = None) -> None:
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
                coordinate=coordinate,
            )

    def _run_single(
            self, molecule: int, line: int, inclination: float = 0, position_angle: float = 0,
            name: PathLike = None, distance: float = 140, velocity_offset: float = 0,
            n_channels: int = 100, threads: int = 1, lineobj: Line = None, npix: int = 100,
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
            output.file_fits = self.radmc_to_fits(newname, self.line_list[line], distance * u.pc)
        self.outputs[lineobj] = output

    def radmc_to_fits(
            self, name: PathLike, line: Line, distance,
    ) -> PathLike:
        """Saves RadMC3D `image.out` files as FITS files

        Returns:
            name of a newly created fits files
        """
        im = radmc3dPy.image.readImage(fname=name)
        restfreq = line.frequency.to(u.Hz).value

        x_deg = ((im.x << u.cm) / distance).to(u.deg, equivalencies=u.dimensionless_angles())
        y_deg = ((im.y << u.cm) / distance).to(u.deg, equivalencies=u.dimensionless_angles())
        freq_hz = (im.freq << u.Hz)

        x_len = len(x_deg)
        midx_i = x_len // 2
        midx_val = x_deg[midx_i]
        mean_x_width = (x_deg[1:] - x_deg[:-1]).mean()

        y_len = len(y_deg)
        midy_i = y_len // 2
        midy_val = y_deg[midy_i]
        mean_y_width = (y_deg[1:] - y_deg[:-1]).mean()

        freq_len = len(freq_hz)
        midfreq_i = freq_len // 2
        midfreq_hz = freq_hz[midfreq_i]
        mean_channel_width = (freq_hz[1:] - freq_hz[:-1]).mean()

        coordinate = self.coordinate

        if coordinate is not None:
            try:
                coordinate = astropy.coordinates.SkyCoord(coordinate)
            except ValueError:
                coordinate = astropy.coordinates.SkyCoord.from_name(coordinate)
        else:
            coordinate = astropy.coordinates.SkyCoord("0h 0d")

        wcs_dict = {
            'CTYPE3': 'RA---TAN', 'CUNIT3': 'deg',
            'CDELT3': -mean_x_width.value, 'CRPIX3': midx_i + 1,
            'CRVAL3': midx_val.value + coordinate.ra.to_value(u.deg), 'NAXIS3': x_len,
            'CTYPE2': 'DEC--TAN', 'CUNIT2': 'deg',
            'CDELT2': mean_y_width.value, 'CRPIX2': midy_i + 1,
            'CRVAL2': midy_val.value + coordinate.dec.to_value(u.deg), 'NAXIS2': y_len,
            'CTYPE1': 'FREQ    ', 'CUNIT1': 'Hz',
            'CDELT1': mean_channel_width.value, 'CRPIX1': midfreq_i + 1, 'CRVAL1': midfreq_hz.value, 'NAXIS1': freq_len,

        }

        header = WCS(wcs_dict).to_header()
        header["HISTORY"] = "Created with DiskCheF package"
        header["HISTORY"] = "(G.V. Smirnov-Pinchukov, https://gitlab.com/SmirnGreg/diskchef)"
        header["HISTORY"] = "using RadMC3D (C. Dullemond, https://github.com/dullemond/radmc3d-2.0)"
        header["RESTFREQ"] = restfreq

        cube = SpectralCube(
            data=np.rot90(im.image * 1e23, axes=[0, 1]) << (u.Jy / u.sr),
            wcs=WCS(wcs_dict), header=header
        )

        fitsname = name.replace(".out", ".fits")

        cube.write(fitsname, overwrite=True)

        self.fitsfiles.append(fitsname)
        self.logger.info("Saved as %s and %s", name, fitsname)
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
            self.outputs[line].file_fits = self.radmc_to_fits(name, line, distance)

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

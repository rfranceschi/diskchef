import glob
from dataclasses import dataclass, field
import os
import re
from typing import Union, Literal
import subprocess
import platform

import diskchef.maps.radiation_fields
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import radmc3dPy

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


if platform.system() == "Windows":
    RADMC_DEFAULT_EXEC = "wsl radmc3d"
else:
    RADMC_DEFAULT_EXEC = "radmc3d"


@dataclass
class RadMCBase(MapBase):
    executable: PathLike = RADMC_DEFAULT_EXEC
    folder: PathLike = 'radmc'
    radii_bins: Union[None, int] = None
    theta_bins: Union[None, int] = None
    outer_radius: Union[None, u.Quantity] = None
    verbosity: int = 0
    wavelengths: u.Quantity = field(default=np.geomspace(0.1, 1000, 100) * u.um)
    modified_random_walk: bool = True
    scattering_mode_max: int = None
    nphot_therm: int = None
    coordinate: Union[str, SkyCoord] = None
    external_source_type: Literal[None, "Draine1978"] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.table.is_in_zr_regular_grid:
            raise CHEFNotImplementedError

        # TODO change to Path.mkdir(parents=True, exists_ok=False)
        try:
            os.makedirs(self.folder)
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
            zr = u.Quantity(np.sort(np.unique(self.table.zr)))
        elif isinstance(self.theta_bins, int):
            zr = u.Quantity(np.linspace(self.table.zr.min(), self.table.zr.max(), self.theta_bins))
        else:
            raise CHEFTypeError("theta_bins should be None or int, not %s (%s)", type(self.theta_bins), self.theta_bins)

        theta = np.pi / 2 * u.rad - np.arctan(zr)
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

    def interpolate_back(self, column: str) -> None:
        """Adds a new `column` to `self.table` with the data iterpolated from `self.polar_table`"""
        self.table[column] = self.polar_table.interpolate(column)(self.table.r, self.table.z)

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
            if self.scattering_mode_max is not None:
                print(f"scattering_mode_max = {self.scattering_mode_max}", file=file)
            if self.modified_random_walk:
                print("modified_random_walk = 1", file=file)
            if self.nphot_therm is not None:
                print(f"nphot_therm = {int(self.nphot_therm)}", file=file)

    def wavelength_micron(self, out_file: PathLike = None) -> None:
        """Creates a `wavelength_micron.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'wavelength_micron.inp')

        with open(out_file, 'w') as file:
            print(len(self.wavelengths), file=file)
            print('\n'.join(f"{entry.to(u.um).value:.7e}" for entry in self.wavelengths), file=file)

    def external_source(self, out_file: PathLike = None) -> None:
        """Creates an `external_source.inp` file"""
        if self.external_source_type is None:
            return
        elif self.external_source_type == "Draine1978":
            isrf = diskchef.maps.radiation_fields.draine1978
        else:
            raise CHEFNotImplementedError("Unsupported ISRF")

        if out_file is None:
            out_file = os.path.join(self.folder, 'external_source.inp')

        with open(out_file, 'w') as file:
            print("2", file=file)
            print(len(self.wavelengths), file=file)
            print('\n'.join(f"{entry.to(u.um).value:.7e}" for entry in self.wavelengths), file=file)
            print(
                '\n'.join(
                    f"{entry.to(u.erg / u.cm ** 2 / u.s / u.Hz / u.sr).value:.7e}"
                    for entry in isrf(self.wavelengths)),
                file=file)

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

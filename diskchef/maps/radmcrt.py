import glob
import logging
from dataclasses import dataclass, field
import os
import re
from functools import cached_property
from pathlib import Path
from typing import Union, Literal, NamedTuple
import subprocess
import platform

import chemical_names
from astropy.coordinates.matrix_utilities import rotation_matrix
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from radio_beam import Beam
from regions import CircleSkyRegion
from scipy.optimize import curve_fit
from spectral_cube import SpectralCube

import diskchef.maps.radiation_fields
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches
import spectral_cube.utils
import radmc3dPy

from diskchef.engine.ctable import CTable
from diskchef.engine.exceptions import CHEFNotImplementedError, CHEFTypeError, CHEFValueError
from diskchef.engine.other import PathLike
from diskchef.lamda.line import Line
from diskchef.maps.base import MapBase


class Cloud(NamedTuple):
    """Tuple to store the foreground and background to mark on channel maps. Used in RadMCOutput"""
    velocity_min: u.Quantity
    velocity_max: u.Quantity


@dataclass
class RadMCOutput:
    """Class to store information and visualize the RadMCRT module output"""
    line: Line
    file_radmc: PathLike = None
    file_fits: PathLike = None
    object_name: str = ""
    distance: u.pc = None
    cloud: Cloud = None
    mode: Literal["image", "image tracetau", "tausurf"] = "image"

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

    @staticmethod
    def gauss(x, H, A, x0, sigma):
        """Gaussian function with a background"""
        return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    @classmethod
    def gauss_fit(cls, x, y):
        """Fit a gaussian into data"""
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(cls.gauss, x, y, p0=[min(y), max(y), mean, sigma])
        return popt

    @classmethod
    def sigma_from_data(cls, data: np.ndarray) -> float:
        """Find rms of the data by fitting a gaussian into the histogram
        (assuming most of the pixels are noise)"""
        values, edges = np.histogram(data, bins=100, density=True)
        means = 0.5 * (edges[1:] + edges[:-1])
        params = cls.gauss_fit(means, values)
        # plt.step(means, values, where="mid")
        # print(params)
        # plt.plot(means, cls.gauss(means, *params))
        return params[3]

    def finalize_plot(self, axes, cube, window, ctr_data=None, ellcolor="black", **kwargs):
        """Finalize the channel or moment map figure."""
        dist_pc = self.distance.to_value(u.pc)
        pxsize = np.sqrt(np.abs(np.linalg.det(cube.wcs.celestial.pixel_scale_matrix)))
        try:
            axes.coords[0].set_major_formatter('hh:mm:ss')
            axes.coords[1].set_major_formatter('dd:mm:ss')
            axes.coords[0].set_ticks(spacing=5 * u.arcsec)  # , format="hh:mm:ss")
            axes.coords[1].set_ticks(spacing=5 * u.arcsec)  # , format="dd:mm:ss")
            axes.coords[0].set_ticklabel(exclude_overlapping=True)
            axes.coords[1].set_ticklabel(exclude_overlapping=True)
        except AttributeError:
            pass
        axes.scatter(
            0, 0,
            color='white', marker='x', zorder=100
        )
        if ctr_data is not None:
            noise = self.sigma_from_data(ctr_data)
            noise_value = getattr(noise, "value", noise)
            ctr_3sigma = axes.contour(
                ctr_data,
                levels=[3 * noise_value],
                colors='black',
                linewidths=0.5,
            )
        else:
            ctr_3sigma = None
        try:
            ell = matplotlib.patches.Ellipse(
                (-0.8 * window, -0.8 * window),
                cube.beam.major.to_value(u.arcsec) * dist_pc,
                cube.beam.minor.to_value(u.arcsec) * dist_pc,
                cube.beam.pa.value + 90,
                hatch='///',
                color=ellcolor
            )
            axes.add_patch(ell)
        except spectral_cube.utils.NoBeamError as e:
            pass
        return {"ctr_3sigma": ctr_3sigma}

    def _check_distance(self):
        if self.distance is None:
            try:
                self.logger.warning(
                    "Distance is not set. Querying from Simbad, but it is not free. "
                    "Provide `distance` argument to RadMCOutput instead")
                from astroquery.simbad import Simbad
                simquery = Simbad()
                simquery.add_votable_fields('distance', 'velocity')
                self.distance = simquery.query_object(self.object_name)[0]["Distance_distance"] * u.pc
            except ImportError:
                self.logger.warning(
                    "Distance to the object was not set. astroquery is not installed. Cannot query Simbad."
                )
            except Exception as e:
                self.logger.warning("Something went wrong querying Simbad: %s", e)

    @cached_property
    def unit_for_channel_map(self) -> u.Unit:
        if self.mode == "image":
            return u.K
        elif self.mode == "image tracetau":
            return u.dimensionless_unscaled
        elif self.mode == "tausurf":
            return u.au
        else:
            self.logger.error("Unknown mode %s")
            return u.dimensionless_unscaled

    def plot_channel_map(
            self,
            window=500 * u.au,
            cmap: Union[matplotlib.colors.Colormap, str] = None,
            velocity_offset: u.km / u.s = 0 * u.km / u.s,
            nx=4, ny=4,
    ) -> Figure:
        symnorm = False
        if cmap is None:
            if self.mode == "tausurf":
                cmap = "coolwarm"
                symnorm = True
            else:
                cmap = "Blues"

        cube = SpectralCube.read(self.file_fits)
        center = cube.wcs.celestial.pixel_to_world(cube.shape[1] / 2, cube.shape[2] / 2)
        self._check_distance()
        radius = (
                window / self.distance
        ).to(
            u.arcsec, equivalencies=u.dimensionless_angles()
        )
        region = CircleSkyRegion(center, radius)
        cube.allow_huge_operations = True
        cube: SpectralCube = (
            cube.subcube_from_regions([region])
            .with_spectral_unit(u.km / u.s, velocity_convention="radio")
        )
        try:
            cube = cube.to(self.unit_for_channel_map)
        except ValueError as e:
            if cube._beam is None:
                cube = cube.with_beam(Beam(1e-10 * u.arcsec)).to(self.unit_for_channel_map)
            else:
                raise ValueError(e)

        central_channel = cube.closest_spectral_channel(velocity_offset)

        # downsampled: SpectralCube = cube.downsample_axis(4, axis=0).with_spectral_unit(u.km / u.s)
        downsampled = cube[central_channel - nx * ny // 2: central_channel + nx * ny // 2 + 1]

        maxval = round(0.9 * downsampled.max().value, 1)
        if symnorm:
            norm = Normalize(-maxval, maxval)
        else:
            norm = Normalize(0, maxval)
        aspect_ratio = downsampled.shape[2] / float(downsampled.shape[1])
        fig_smallest_dim_inches = 5
        gridratio = ny / float(nx) * aspect_ratio
        if gridratio > 1:
            ysize = fig_smallest_dim_inches * gridratio
            xsize = fig_smallest_dim_inches
        else:
            xsize = fig_smallest_dim_inches * gridratio
            ysize = fig_smallest_dim_inches

        fig: Figure = plt.figure(figsize=(xsize, ysize))
        gs = GridSpec(nrows=ny, ncols=nx, hspace=0.0, wspace=0.0, figure=fig)
        ax = None
        window, window_unit = window.value, window.unit
        window_half = round(window * 0.6, ndigits=-2)
        for i in range(ny):
            for j in range(nx):
                ax = fig.add_subplot(
                    gs[i, j],
                    sharex=ax, sharey=ax
                )
                im = ax.imshow(
                    downsampled[i * nx + j].value,
                    cmap=cmap,
                    norm=norm,
                    origin="lower",
                    extent=[-window, window, -window, window],
                )
                self.finalize_plot(ax, downsampled, window)
                this_spectral_coordinate = downsampled.spectral_axis[i * nx + j]
                ax.text(
                    0.5,
                    0.9,
                    f"{this_spectral_coordinate.value: .1f} {this_spectral_coordinate.unit.to_string('latex_inline')}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize="small"
                )
                if self.cloud is not None:
                    velocity = downsampled.spectral_axis[i * nx + j]
                    if self.cloud.velocity_min < velocity < self.cloud.velocity_max:
                        ax.scatter(0.5, 0.5, marker='x', color='red', alpha=0.2, transform=ax.transAxes, s=2000)
                if i != ny - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    ax.set_xticks([-window_half, 0, window_half])
                if j != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                else:
                    ax.set_yticks([-window_half, 0, window_half])
                if i == ny - 1 and j == 0:
                    ax.set_xlabel(f"[{window_unit.to_string('latex_inline')}]")
                    ax.set_ylabel(f"[{window_unit.to_string('latex_inline')}]", labelpad=-18)
        else:
            cax = fig.add_axes([0.90, 0.1, 0.03, 0.8])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.set_xlabel(f"\n[{downsampled.unit.to_string('latex')}]")
        gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig.suptitle(f"{self.object_name} {chemical_names.from_string(self.line.molecule)}")
        return fig

    def extract_surface(self):
        """
        Extract surface for tausurf mode

        Raises:
            CHEFValueError if used for other mode outputs
        """
        if self.mode != "tausurf":
            raise CHEFValueError(f"Mode is '{self.mode}', 'tausurf' expected!")

        cube = SpectralCube.read(self.file_fits)
        try:
            cube = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio")
        except:
            self.logger.warning("Could not convert cube to km/s")
        pixscale = (cube.wcs.celestial.proj_plane_pixel_area() ** 0.5 * self.distance).to(u.au,
                                                                                          u.dimensionless_angles())
        vscale = cube.wcs.pixel_scale_matrix[-1, -1] * u.Unit(cube.wcs.world_axis_units[-1])
        x_grid = (np.arange(cube.wcs.celestial.pixel_shape[0]) - cube.wcs.celestial.pixel_shape[0] / 2 + 0.5) * pixscale
        y_grid = (np.arange(cube.wcs.celestial.pixel_shape[1]) - cube.wcs.celestial.pixel_shape[1] / 2 + 0.5) * pixscale
        v_grid = (np.arange(cube.wcs.pixel_shape[-1]) - cube.wcs.celestial.pixel_shape[-1] / 2 + 0.5) * vscale
        x, y, v = np.meshgrid(x_grid, y_grid, v_grid)
        z = np.swapaxes(cube.filled_data[:].to(u.au), 0, -1)

        mask = (np.abs(z) < x.max())
        xpl = x[mask].flatten()
        ypl = y[mask].flatten()
        zpl = z[mask].flatten()
        vpl = v[mask].flatten()



        rotation_posang = rotation_matrix(-position_angle, axis='z')
        rotation_incl = rotation_matrix(inclination, axis='x')
        coords = CartesianRepresentation(x=xpl, y=ypl, z=zpl)
        coords_posang_incl = coords.transform(rotation_posang).transform(rotation_incl)

        r = (coords_posang_incl.x ** 2 + coords_posang_incl.y ** 2) ** 0.5
        zr = coords_posang_incl.z / r

        self.tausurf_coords = dict(
            r=r,
            zr=zr,
            x=xpl,
            y=ypl,
            z=zpl,
            v=vpl
        )
        return self.tausurf_coords


if platform.system() == "Windows":
    logging.warning(
        "RadMC is not supported on Windows out of the box (GNU make...). "
        "To use it, install Windows Subsystem for Linux, and install the RadMC package in $PATH. "
        "radmc3dpy is also not available for Windows. Please proceed with caution."
    )
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
    wavelengths: u.Quantity = field(default=np.geomspace(0.0912, 1000, 100) * u.um)
    modified_random_walk: bool = True
    scattering_mode_max: int = None
    nphot_therm: int = None
    nphot_mono: int = None
    nphot_spec: int = None
    coordinate: Union[str, SkyCoord] = None
    external_source_type: Literal["Draine1978", "WeingartnerDraine2001", "None"] = None
    external_source_multiplier: float = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.table.is_in_zr_regular_grid:
            raise CHEFNotImplementedError("Grids, which are not cartesian in r, z/r are not supported.")
        try:
            Path(self.folder).mkdir(parents=True, exist_ok=False)
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
        self.polar_table['Velocity Phi'] = (
                np.sqrt(
                    c.G * self.chemistry.physics.star_mass
                    / self.polar_table['Distance to star']
                )
                * np.sin(self.polar_table['Theta'])
        ).to(u.cm / u.s)

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
            if "In the molecular data file" in match.group(1):
                self.logger.info(match.group(1))
            else:
                self.logger.warn(match.group(1))
        for match in re.finditer(r"ERROR:(.*\n(?:  .*\n){2,})", proc.stdout):
            self.logger.error(match.group(1))

    def interpolate(self, column: str) -> None:
        """Adds a new `column` to `self.polar_table` with the data iterpolated from `self.table`"""
        self.polar_table[column] = self.table.interpolate(column)(self.polar_table.r, np.abs(self.polar_table.z))

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
            if self.nphot_mono is not None:
                print(f"nphot_mono = {int(self.nphot_mono)}", file=file)
            if self.nphot_spec is not None:
                print(f"nphot_spec = {int(self.nphot_spec)}", file=file)

    def wavelength_micron(self, out_file: PathLike = None) -> None:
        """Creates a `wavelength_micron.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'wavelength_micron.inp')

        with open(out_file, 'w') as file:
            print(len(self.wavelengths), file=file)
            print('\n'.join(f"{entry.to(u.um).value:.7e}" for entry in self.wavelengths), file=file)

    def external_source(self, out_file: PathLike = None) -> None:
        """Creates an `external_source.inp` file"""
        if self.external_source_type is None or self.external_source_type == "None":
            return
        elif self.external_source_type == "Draine1978":
            isrf = diskchef.maps.radiation_fields.draine1978
        elif self.external_source_type == "WeingartnerDraine2001":
            isrf = diskchef.maps.radiation_fields.weingartner_draine_2001
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
                    f"{entry:.7e}"
                    for entry in self.external_source_multiplier * isrf(self.wavelengths).to(
                        u.erg / u.cm ** 2 / u.s / u.Hz / u.sr).value),
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
        logging.warning("Deprecated")
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

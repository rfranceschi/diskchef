import glob
from typing import Union
import os
import shutil
from collections import Counter
from dataclasses import dataclass
import subprocess
import re
import time
from warnings import warn

PathLike = Union[str, os.PathLike]

import numpy as np

from astropy import units as u
from astropy import constants as c
from astropy.io import fits

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import radmc3dPy

from diskchef.maps.base import MapBase, Line
from diskchef.engine.exceptions import CHEFNotImplementedError, RADMCWarning
from diskchef.engine.ctable import CTable
from diskchef.lamda import file


@dataclass
class RadMCRT(MapBase):
    """
    """
    executable: PathLike = 'radmc3d'
    verbosity: int = 0
    folder: PathLike = 'radmc'

    def __post_init__(self):
        super(RadMCRT, self).__post_init__()
        if not self.table.is_in_zr_regular_grid:
            raise CHEFNotImplementedError

        radii = np.sort(np.unique(self.table.r)).to(u.cm)
        zr = np.sort(np.unique(self.table.zr))
        theta = np.pi / 2 - np.arctan(zr)
        self.radii_edges = u.Quantity([radii[0], *np.sqrt(radii[1:] * radii[:-1]), radii[-1]]).value
        self.zr_edges = np.array([zr[0], *(0.5 * (zr[1:] + zr[:-1])), zr[-1]])
        self.theta_edges = np.sort(np.pi / 2 - np.arctan(self.zr_edges))

        R, THETA = np.meshgrid(radii, theta)
        self.polar_table = CTable()
        self.polar_table['Radius'] = R.flatten()
        self.polar_table['Theta'] = THETA.flatten() << u.rad
        self.polar_table['Altitude'] = (np.pi / 2 * u.rad - self.polar_table['Theta']) << u.rad
        self.polar_table['Height'] = self.polar_table['Radius'] * np.sin(self.polar_table['Altitude'])
        self.polar_table.sort(['Theta', 'Radius'])
        self.interpolate('n(H+2H2)')
        self.polar_table['Velocity R'] = 0 * u.cm / u.s
        self.polar_table['Velocity Theta'] = 0 * u.cm / u.s
        self.polar_table['Velocity Phi'] = \
            np.sqrt(c.G * self.chemistry.physics.star_mass / self.polar_table['Radius']).to(u.cm / u.s)

        self.nrcells = (len(self.radii_edges) - 1) * (len(self.theta_edges) - 1)

    def interpolate(self, column: str):
        self.polar_table[column] = self.table.interpolate(column)(self.polar_table.r, self.polar_table.z)

    def create_files(self):
        self.radmc3d()
        self.wavelength_micron()
        self.amr_grid()
        self.gas_temperature()
        self.lines()
        self.gas_velocity()

        for molecule in self.molecules_list:
            self.numberdens(species=molecule)
            self.molecule(species=molecule)
        self.logger.info("Files written to %s", self.folder)

    def radmc3d(self, out_file: PathLike = None) -> None:
        """Creates an empty `radmc3d.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'radmc3d.inp')

        with open(out_file, 'a') as file:
            pass

    def wavelength_micron(self, out_file: PathLike = None) -> None:
        """Creates a `wavelength_micron.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'wavelength_micron.inp')

        wavelengths = np.geomspace(0.1, 1000, 100)

        with open(out_file, 'w') as file:
            print(len(wavelengths), file=file)
            print('\n'.join(f"{entry:.7e}" for entry in wavelengths), file=file)

    def amr_grid(self, out_file: PathLike = None) -> None:
        """
        Can call the method using out_file=sys.stdout

        Returns:

        """

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

    def gas_temperature(self, out_file: PathLike = None) -> None:
        """
        Writes the gas temperature file
    
        Returns:
    
        """

        if out_file is None:
            out_file = os.path.join(self.folder, 'gas_temperature.inp')

        self.interpolate('Gas temperature')

        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print(self.nrcells, file=file)
            print('\n'.join(f"{entry:.7e}" for entry in self.polar_table['Gas temperature'].to(u.K).value), file=file)

    def numberdens(self, species: str, out_file: PathLike = None) -> None:
        """
        Writes the gas number density file
    
        Returns:
    
        """

        if out_file is None:
            out_file = os.path.join(self.folder, f'numberdens_{species}.inp')

        self.interpolate(species)

        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print(self.nrcells, file=file)
            print(
                '\n'.join(
                    f"{entry:.7e}" for entry
                    in self.polar_table['n(H+2H2)'].to(u.cm ** (-3)).value * self.table[species]),
                file=file
            )

    def molecule(self, species: str, out_file: PathLike = None) -> None:
        """
        Writes the molecule transition file

        species:    str     name of the molecule

    
        Returns:
    
        """

        if out_file is None:
            out_file = os.path.join(self.folder, f'molecule_{species}.inp')

        shutil.copy(file(species)[0], out_file)

    def lines(self, out_file: PathLike = None) -> None:
        """
        """

        self.molecules_list = sorted(list(set([line.molecule for line in self.line_list])))

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
        """

        Args:
            out_file:

        Returns:

        """

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
            doppcatch: bool = False, threads: int = 1
    ):
        self.logger.info("Running radmc3d")
        for line in self.line_list:
            self._run_single(
                molecule=self.molecules_list.index(line.molecule) + 1,
                line=line.transition,
                inclination=inclination.to(u.deg).value, position_angle=position_angle.to(u.deg).value,
                velocity_offset=velocity_offset.to(u.km / u.s).value,
                name=f"{line.name}_image.out",
                distance=distance.to(u.pc).value,
                doppcatch=doppcatch,
                threads=threads
            )

    def _run_single(
            self, molecule: int, line: int, inclination: float = 0, position_angle: float = 0,
            name: PathLike = None, distance: float = 140, doppcatch: bool = False, velocity_offset: float = 0,
            n_channels: int = 100, threads: int = 1
    ):
        start = time.time()
        command = (f"{self.executable} image "
                   f"imolspec {molecule} "
                   f"iline {line} "
                   f"widthkms 10 "
                   f"incl {inclination} "
                   f"posang {position_angle} "
                   f"vkms {velocity_offset} "
                   f"linenlam {n_channels} "
                   f"setthreads {threads} "
                   # + "doppcatch " if doppcatch else " "
                   )
        self.logger.info("Running radmc3d for molecule %d and transition %d: %s", molecule, line, command)
        proc = subprocess.run(
            command,
            cwd=self.folder,
            text=True,
            capture_output=True,
            shell=True
        )
        self.logger.info("radmc3d finished after %7.2e s", time.time() - start)
        if proc.stderr: self.logger.error(proc.stderr)
        self.logger.debug(proc.stdout)
        for match in re.finditer(r"WARNING:(.*\n(?:  .*\n){2,})", proc.stdout):
            self.logger.warn(match.group(1))
        if name is not None:
            newname = os.path.join(self.folder, name)
            shutil.move(os.path.join(self.folder, "image.out"), newname)
            im = radmc3dPy.image.readImage(fname=newname)
            fitsname = newname.replace(".out", ".fits")
            molfile = radmc3dPy.molecule.radmc3dMolecule()
            molfile.read(fname=os.path.join(self.folder, f"molecule_{self.molecules_list[molecule - 1]}.inp"))
            restfreq = molfile.freq[line - 1]
            if os.path.exists(fitsname):
                os.remove(fitsname)
            im.writeFits(fname=fitsname, nu0=restfreq, dpc=distance,
                         fitsheadkeys={"CUNIT3": "Hz", "BUNIT": "Jy pix**(-1)", "RESTFREQ": restfreq})
            self.logger.info("Saved as %s and %s", newname, fitsname)
            # with fits.open(fitsname, 'update') as f:
            #     for hdu in f:
            #         hdu.header['CUNIT3'] = "      Hz"
            #         hdu.header['BUNIT'] = "  Jy/pix"
            #         hdu.header['RESTFREQ'] = restfreq
            self.logger.debug("Modified FITS header unit: HZ to Hz, JY/PX to Jy pix**(-1), set RESTFREQ %s Hz",
                              restfreq)


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

import os
import shutil
from collections import Counter

from dataclasses import dataclass
from sys import modules
import numpy as np

from astropy import units as u
from astropy import constants as c

from diskchef.maps.base import MapBase
from diskchef.engine.exceptions import CHEFNotImplementedError
from diskchef.engine.ctable import CTable
from diskchef.lamda import file


@dataclass
class RadMCRT(MapBase):
    """
    Testing docstring

    >>> 5*5
    25
    """
    folder: str = 'radmc'
    verbosity: int = 0

    def __post_init__(self):
        if not self.table.is_in_zr_regular_grid:
            raise CHEFNotImplementedError

        radii = np.sort(np.unique(self.table.r)).to(u.cm)
        zr = np.sort(np.unique(self.table.zr))
        theta = np.arctan(zr)
        self.radii_edges = u.Quantity([radii[0], *np.sqrt(radii[1:] * radii[:-1]), radii[-1]]).value
        self.zr_edges = np.array([zr[0], *0.5 * (zr[1:] + zr[:-1]), zr[-1]])
        self.theta_edges = np.arctan(self.zr_edges)

        R, THETA = np.meshgrid(radii, theta)
        self.polar_table = CTable()
        self.polar_table['Radius'] = R.flatten()
        self.polar_table['Theta'] = THETA.flatten()
        self.polar_table['Height'] = self.polar_table['Radius'] * np.sin(self.polar_table['Theta'])
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

        for line in self.line_list:
            self.numberdens(species=line.molecule)
            self.molecule(species=line.molecule)

    def radmc3d(self, out_file: str = None) -> None:
        """Creates an empty `radmc3d.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'radmc3d.inp')

        with open(out_file, 'a') as file:
            pass


    def wavelength_micron(self, out_file: str = None) -> None:
        """Creates a `wavelength_micron.inp` file"""

        if out_file is None:
            out_file = os.path.join(self.folder, 'wavelength_micron.inp')

        wavelengths = np.geomspace(0.1, 1000, 100)

        with open(out_file, 'w') as file:
            print(len(wavelengths), file=file)
            print('\n'.join(str(entry) for entry in wavelengths), file=file)

    def amr_grid(self, out_file: str = None) -> None:
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
            print(' '.join(str(entry) for entry in self.radii_edges), file=file)
            print(' '.join(str(entry) for entry in self.theta_edges), file=file)
            print(0, 2 * np.pi, file=file)

    def gas_temperature(self, out_file: str = None) -> None:
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
            print('\n'.join(str(entry) for entry in self.polar_table['Gas temperature'].to(u.K).value), file=file)

    def numberdens(self, species: str, out_file: str = None) -> None:
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
            print('\n'.join(str(entry) for entry in self.polar_table['n(H+2H2)'].to(u.cm ** (-3)).value), file=file)

    def molecule(self, species: str, out_file: str = None) -> None:
        """
        Writes the molecule transition file

        species:    str     name of the molecule

    
        Returns:
    
        """

        if out_file is None:
            out_file = os.path.join(self.folder, f'molecule_{species}.inp')

        shutil.copy(file(species)[0], out_file)

    def lines(self, out_file: str = None):
        """
        """

        self.molecules_list = set([line.molecule for line in self.line_list])

        if out_file is None:
            out_file = os.path.join(self.folder, 'lines.inp')
        with open(out_file, 'w') as file:
            print('2', file=file)  # 2 for molecules
            print(len(set([line.molecule for line in self.line_list])), file=file)  # unique list of molecules

            molecules = Counter([line.molecule for line in self.line_list])
            for molecule in molecules:
                coll_partners = next(line for line in self.line_list if line.molecule == molecule).collision_partner
                print(f'{molecule} leiden 0 0 {len(coll_partners)}', file=file)
                print('\n'.join(coll_partners), file=file)

    def gas_velocity(self, out_file: str = None):
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
                print(vr, vtheta, vphi, file=file)

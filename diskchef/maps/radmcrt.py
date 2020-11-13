import os

from dataclasses import dataclass
import numpy as np

from astropy import units as u

from diskchef.maps.base import MapBase
from diskchef.engine.exceptions import CHEFNotImplementedError
from diskchef.engine.ctable import CTable


@dataclass
class RadMCRT(MapBase):
    """
    Testing docstring

    >>> 5*5
    21
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
        self.interpolate('Dust density')

    def interpolate(self, column: str):
        self.polar_table[column] = self.table.interpolate(column)(self.polar_table.r, self.polar_table.z)

    def create_files(self):
        self.amr_grid()

    def amr_grid(self, out_file=None):
        """
        Can call the method using out_file=sys.stdout

        Returns:

        """

        if not self.table.is_in_zr_regular_grid:
            raise CHEFNotImplementedError

        print(self.folder)

        if out_file is None:
            out_file = os.path.join(self.folder, 'amr_grid.inp')
        with open(out_file, 'w') as file:
            print('1', file=file)  # Typically 1 at present
            print('0', file=file)  # Grid style (regular = 0)
            print('100', file=file)  # Spherical coordinate system
            print('1' if self.verbosity else '0', file=file)  # Grid info
            print('1 1 0', file=file)  # Included coordinates
            print(len(self.radii_edges) - 1, len(self.theta_edges) - 1, 1, file=file)
            print(str(self.radii_edges)[1:-1], file=file)
            print(str(self.theta_edges)[1:-1], file=file)
            print(0, 2 * np.pi, file=file)

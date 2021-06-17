import os

import astropy.io.ascii
from astropy import units as u

import scipy.interpolate

class YorkeBodenheimer2008:
    """Class to calculate stellar temperature and radius from its mass for age 2 Myr

    Based on Yorke, Bodenheimer 2008, + priv. comm (via Tamara Molyarova)"""
    def __init__(self):
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trktime0210.dat")
        self.table = astropy.io.ascii.read(file)
        self.table.sort('M')
        self._radius_callable = scipy.interpolate.interp1d(self.table['M'], self.table['R'])
        self._temp_callable = scipy.interpolate.interp1d(self.table['M'], self.table['T_eff'])

    @u.quantity_input
    def radius(self, mass: u.solMass) -> u.cm:
        return (self._radius_callable(mass.to(u.solMass)) * u.solRad).to(u.cm)

    @u.quantity_input
    def effective_temperature(self, mass: u.solMass) -> u.K:
        return self._temp_callable(mass.to(u.solMass)) * u.K
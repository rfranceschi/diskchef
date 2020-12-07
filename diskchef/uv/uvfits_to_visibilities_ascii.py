"""Module with functions to convert GILDAS UVTable to GALARIO visibilities format"""
from typing import Union, Literal
import os

from astropy.table import Table, QTable
from astropy import units as u
from astropy import constants as c

from matplotlib import pyplot as plt

import uvplot


class UVFits:
    """
    Reads GILDAS-outputted visibilities UVFits file


    Usage:

    >>> uv = UVFits(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "s-Wide-1+C.uvfits"))
    >>> uv.table[0:5].pprint_all()  # doctest: +NORMALIZE_WHITESPACE
             u                   v                   Re                    Im                 Weight
             m                   m                   Jy                    Jy                  Jy2
    ------------------- ------------------- -------------------- --------------------- -------------------
    -0.8546872724500936 -111.88782560913619  0.08782878543466197   0.03792351358711706 0.20431383019990396
      35.68536730327412   61.11044271472704    0.155928085996764  0.033993665872594336 0.22821656060995169
      36.53999262455663  172.99830358929054 0.045044034643390775   0.00792304818270634 0.22375530051944253
     -82.12765585917153 -41.241167812120636  0.12146219850511554   0.01376124881887358 0.12433690106328266
     -81.27303053788901   70.64653734266903  0.13093146835634065 -0.010734218118883455 0.12252077523637628
    """

    def __init__(self, path: str, channel: Union[int, u.Quantity, Literal['all', 'sum']] = 0):
        self.fits = Table.read(path, hdu=0)
        self.table = QTable()
        self.u = self.fits['UU'].data * (u.s * c.c)
        self.v = self.fits['VV'].data * (u.s * c.c)
        data = self.fits['DATA'][:, 0, 0, 0, channel, 0, :]
        self.re = data[:, 0].data << u.Jy
        self.im = data[:, 1].data << u.Jy
        self.weight = data[:, 2].data << (u.Jy ** 2)
        self.table['u'] = self.u
        self.table['v'] = self.v
        self.table['Re'] = self.re
        self.table['Im'] = self.im
        self.table['Weight'] = self.weight
        self.wavelength = c.c / (
                (self.fits.meta['CRVAL4'] +
                 (self.fits.meta['CRPIX4'] - channel + 1) * self.fits.meta['CDELT4']
                 ) * u.Hz)

    def plot(self):
        uv = uvplot.UVTable(
            (self.u.to(u.m), self.v.to(u.m), self.re.to(u.Jy), self.im.to(u.Jy), self.weight),
            columns=uvplot.COLUMNS_V0,
            wle=self.wavelength.to(u.m)
        )
        uvbin_size = 3e4
        axes = uv.plot(uvbin_size=uvbin_size)
        axes[0].figure.savefig('fig.png')

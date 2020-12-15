"""Module with functions to convert GILDAS UVTable to GALARIO visibilities format"""
from typing import Union, Literal, List, Sequence
import os

import numpy as np

from astropy.table import Table, QTable
from astropy import units as u
from astropy import constants as c

from matplotlib import pyplot as plt

import uvplot

from diskchef.engine.other import PathLike
from diskchef.engine.exceptions import CHEFTypeError


class UVFits:
    """
    Reads GILDAS-outputted visibilities UVFits file

    Args:
        path: PathLike -- path to the uv fits table
        channel: which channel to take from the file
            int -- single channel
            list -- given channels
            slice -- (a:b:c == slice(a,b,c)) -- slice of channels
            'all' -- all channels from the file
        sum: bool -- calculate weigthed mean of all channels (for continuum)

    Usage:

    >>> uv = UVFits(
    ...     os.path.join(os.path.dirname(__file__), "..", "tests", "data", "s-Wide-1+C.uvfits"),
    ...     'all', sum=True
    ... )
    >>> uv.table[0:5].pprint_all()  # doctest: +NORMALIZE_WHITESPACE
             u                   v                 Re [1]               Im [1]            Weight [1]
             m                   m                   Jy                   Jy                 Jy-2
    ------------------- ------------------- ------------------- --------------------- ------------------
    -0.8546872724500936 -111.88782560913619 0.08481375196790505   0.02011162435339697  4.392747296119472
      35.68536730327412   61.11044271472704 0.14395800105016973 0.0011129999808199087  4.906656253315232
      36.53999262455663  172.99830358929054 0.04616766278918842 -0.009757890697913866  4.810738961168015
     -82.12765585917153 -41.241167812120636 0.14572769842806682  0.012205666181597085 2.6732435871384994
     -81.27303053788901   70.64653734266903  0.1260857611090533 -0.018951768666146913  2.634196780195305

    >>> uv = UVFits(
    ...     os.path.join(os.path.dirname(__file__), "..", "tests", "data", "s-Wide-1+C.uvfits"),
    ...     [1,2,3], sum=False
    ... )
    >>> uv.table[0:5].pprint_all()  # doctest: +NORMALIZE_WHITESPACE
             u                   v                             Re [3]                                        Im [3]                                     Weight [3]
             m                   m                               Jy                                            Jy                                          Jy-2
    ------------------- ------------------- ------------------------------------------- ----------------------------------------------- ------------------------------------------
    -0.8546872724500936 -111.88782560913619  0.08013948631211772 .. 0.07302437694644187    0.022133425091247466 .. 0.013778203363983665 0.19701689428642943 .. 0.20917844580050068
      35.68536730327412   61.11044271472704   0.14916945376238983 .. 0.1606281736906061   0.004351084873651592 .. -0.008113799327774212 0.22006599150348807 .. 0.23365029837275292
      36.53999262455663  172.99830358929054 0.02026272564921367 .. 0.051114215373805026 -0.007280084597144622 .. -0.0062913988333082845 0.21576404157411666 .. 0.22908280648299315
     -82.12765585917153 -41.241167812120636  0.13376226427333338 .. 0.18171489805253205     0.014664527234772186 .. 0.03560468549769666 0.11989631175397047 .. 0.12729731479045647
     -81.27303053788901   70.64653734266903   0.1422663387734871 .. 0.15214664685856188  -0.008367051589066395 .. -0.017851764309400768 0.11814501360102375 .. 0.12543794548908482
    """

    def __init__(self, path: PathLike, channel: Union[int, Sequence, slice, Literal['all']] = 'all', sum: bool = True):
        self.path = os.path.abspath(path)
        self.fits = Table.read(path, hdu=0, memmap=False)
        self.table = QTable()
        self.u = self.fits['UU'].data * (u.s * c.c)
        self.v = self.fits['VV'].data * (u.s * c.c)
        if channel in ('all', Ellipsis):
            fetch_channel = Ellipsis
        elif isinstance(channel, int):
            fetch_channel = slice(channel, channel + 1)
        elif isinstance(channel, (slice, Sequence)):
            fetch_channel = channel
        else:
            raise CHEFTypeError("channel should be either: int, Sequence, slice, 'all', Ellipsis'")
        data = self.fits['DATA'][:, 0, 0, 0, fetch_channel, 0, :]
        fetched_channels = np.arange(self.fits['DATA'].shape[4])[fetch_channel]
        self.re = data[:, :, 0] << u.Jy
        self.im = data[:, :, 1] << u.Jy
        self.weight = data[:, :, 2] << (u.Jy ** -2)
        if sum:
            total_weight = np.sum(self.weight, axis=1, keepdims=True)
            self.re = np.sum(self.weight * self.re, axis=1, keepdims=True) / total_weight
            self.im = np.sum(self.weight * self.im, axis=1, keepdims=True) / total_weight
            self.weight = total_weight
            fetched_channels = np.mean(fetched_channels, keepdims=True)
        self.table['u'] = self.u
        self.table['v'] = self.v
        self.table['Re'] = self.re
        self.table['Im'] = self.im
        self.table['Weight'] = self.weight
        self.frequencies = (
                (self.fits.meta['CRVAL4'] +
                 (self.fits.meta['CRPIX4'] - fetched_channels + 1) * self.fits.meta['CDELT4']
                 ) * u.Hz)
        self.wavelengths = c.c / self.frequencies

    def plot(self):
        uv = uvplot.UVTable(
            (self.u.to(u.m), self.v.to(u.m), self.re.to(u.Jy), self.im.to(u.Jy), self.weight),
            columns=uvplot.COLUMNS_V0,
            wle=self.wavelength.to(u.m)
        )
        uvbin_size = 3e4
        axes = uv.plot(uvbin_size=uvbin_size)
        axes[0].figure.savefig('fig.png')

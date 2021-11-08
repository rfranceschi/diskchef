"""Module with functions to convert GILDAS UVTable to GALARIO visibilities format"""
import pickle

import os
from typing import Union, Literal, Sequence
import logging

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.table import Table, QTable
import astropy.wcs
from matplotlib import pyplot as plt
import spectral_cube

try:
    import galario

    if galario.HAVE_CUDA:
        from galario import double_cuda as g_double
        from galario import single_cuda as g_single
    else:
        from galario import double as g_double
        from galario import single as g_single

except ModuleNotFoundError:
    print("Install galario:")
    print("$ conda install -c conda-forge galario")
    print("Works only on linux and osx 64 bit (not Windows)")

from diskchef.engine.exceptions import CHEFTypeError, CHEFValueError, CHEFNotImplementedError
from diskchef.engine.other import PathLike

import warnings
from spectral_cube.utils import SpectralCubeWarning

warnings.filterwarnings(action='ignore', category=SpectralCubeWarning,
                        append=True)


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
             m                   m                   Jy                   Jy               1 / Jy2
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
             m                   m                               Jy                                            Jy                                          1 / Jy2
    ------------------- ------------------- ------------------------------------------- ----------------------------------------------- ------------------------------------------
    -0.8546872724500936 -111.88782560913619  0.08013948631211772 .. 0.07302437694644187    0.022133425091247466 .. 0.013778203363983665 0.19701689428642943 .. 0.20917844580050068
      35.68536730327412   61.11044271472704   0.14916945376238983 .. 0.1606281736906061   0.004351084873651592 .. -0.008113799327774212 0.22006599150348807 .. 0.23365029837275292
      36.53999262455663  172.99830358929054 0.02026272564921367 .. 0.051114215373805026 -0.007280084597144622 .. -0.0062913988333082845 0.21576404157411666 .. 0.22908280648299315
     -82.12765585917153 -41.241167812120636  0.13376226427333338 .. 0.18171489805253205     0.014664527234772186 .. 0.03560468549769666 0.11989631175397047 .. 0.12729731479045647
     -81.27303053788901   70.64653734266903   0.1422663387734871 .. 0.15214664685856188  -0.008367051589066395 .. -0.017851764309400768 0.11814501360102375 .. 0.12543794548908482
    """

    def __init__(self, path: PathLike, channel: Union[int, Sequence, slice, Literal['all']] = 'all', sum: bool = True):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

        self.path = os.path.abspath(path)
        if self.path.lower().endswith((".uvfits", ".fits")):
            self._fits = Table.read(path, hdu=0, memmap=False)
            self.table = QTable()
            self.u = self._fits['UU'].data * (u.s * c.c)
            self.v = self._fits['VV'].data * (u.s * c.c)
            if channel in ('all', Ellipsis):
                self.fetch_channel = Ellipsis
            elif isinstance(channel, int):
                self.fetch_channel = slice(channel, channel + 1)
            elif isinstance(channel, (slice, Sequence)):
                self.fetch_channel = channel
            else:
                raise CHEFTypeError("channel should be either: int, Sequence, slice, 'all', Ellipsis'")
            data = self._fits['DATA'][:, 0, 0, 0, self.fetch_channel, 0, :]
            self.fetched_channels = np.arange(self._fits['DATA'].shape[4])[self.fetch_channel]
            self.re = data[:, :, 0] << u.Jy
            self.im = data[:, :, 1] << u.Jy
            self.weight = data[:, :, 2] << (u.Jy ** -2)
            if sum:
                total_weight = np.sum(self.weight, axis=1, keepdims=True)
                self.re = np.sum(self.weight * self.re, axis=1, keepdims=True) / total_weight
                self.im = np.sum(self.weight * self.im, axis=1, keepdims=True) / total_weight
                self.weight = total_weight
                self.fetched_channels = np.mean(self.fetched_channels, keepdims=True)
            self._update_table()
            self.frequencies = (
                    (self._fits.meta['CRVAL4'] +
                     (self.fetched_channels - self._fits.meta['CRPIX4'] + 1) * self._fits.meta['CDELT4']
                     ) * u.Hz)
        elif self.path.lower().endswith(".pkl"):
            self._fits = None
            with open(self.path, "rb") as pkl:
                self.u, self.v = pickle.load(pkl)
                self.re = np.empty_like(self.u)
                self.im = np.empty_like(self.u)
                self.weight = np.empty_like(self.u)
                self.table = astropy.table.QTable()
                self._update_table()

    @property
    def data(self):
        return self._fits['DATA'][:, 0, 0, 0, :, 0, :]

    @data.setter
    def data(self, value):
        if self._fits["DATA"].shape != value.shape:
            raise CHEFValueError("Shape mismatch! Use UVFits.set_data instead!")
        self._update_fits(value)

    def _update_table(self):
        self.table['u'] = self.u
        self.table['v'] = self.v
        self.table['Re'] = self.re
        self.table['Im'] = self.im
        self.table['Weight'] = self.weight

    def _update_fits(self, data, frequencies=None):
        self._fits['DATA'] = data
        self.re = self.data[:, :, 0] << u.Jy
        self.im = self.data[:, :, 1] << u.Jy
        self.weight = self.data[:, :, 2] << (u.Jy ** -2)
        self.table['Re'] = self.re
        self.table['Im'] = self.im
        self.table['Weight'] = self.weight
        self._update_table()

    @u.quantity_input
    def set_data(self, data: np.ndarray, frequencies: u.Hz = None):
        """Set new visibilities data (and, optionally, frequencies) to UVFits instance"""
        if self._fits is not None:
            if self._fits["DATA"].shape != data.shape:
                if frequencies.shape != self.data.shape[1] and self.frequencies is None:
                    raise CHEFValueError("Shape mismatch!")
                self.frequencies = frequencies
            self._update_fits(data, frequencies)
        else:
            self.re = data[:, 0, 0, 0, :, 0, :][:, :, 0] << u.Jy
            self.im = data[:, 0, 0, 0, :, 0, :][:, :, 1] << u.Jy
            self.weight = data[:, 0, 0, 0, :, 0, :][:, :, 2] << (u.Jy ** -2)
            self.table['Re'] = self.re
            self.table['Im'] = self.im
            self.table['Weight'] = self.weight

    @property
    def wavelengths(self):
        return c.c / self.frequencies

    def plot_uvgrid(self):
        plt.axis('equal')
        plt.scatter([*self.u, *(-self.u)], [*self.v, *(-self.v)], alpha=0.5, s=0.2)

    def plot_total_power(self, rest_freq: u.Hz = None):
        if  rest_freq is not None:
            frequencies = self.frequencies.to(u.km/u.s, equivalencies=u.doppler_radio(rest_freq))
        else:
            frequencies = self.frequencies
        plt.plot(frequencies, np.sum(np.abs(self.visibility), axis=0))

    def image_to_visibilities(self, file: PathLike):
        """Import cube from a FITS `file`, sample it with visibilities of this UVFITS"""
        cube = spectral_cube.SpectralCube.read(file)
        pixel_area_units = u.Unit(cube.wcs.celestial.world_axis_units[0]) * u.Unit(
            cube.wcs.celestial.world_axis_units[1])
        pixel_area = astropy.wcs.utils.proj_plane_pixel_area(cube.wcs.celestial) * pixel_area_units
        dxy = np.sqrt(pixel_area).to_value(u.rad)
        cube = (cube.to(u.Jy / u.sr) * pixel_area).to(u.Jy)

        visibilities = []
        for i, frequency in enumerate(cube.spectral_axis):
            wl = (c.c / frequency).si
            u_wavelengths = (self.u / wl).si
            v_wavelengths = (self.v / wl).si
            vis = g_double.sampleImage(cube[i], dxy, u_wavelengths, v_wavelengths)
            visibilities.append(vis)
        visibilities = np.array(visibilities)
        visibilities_real_imag_weight = np.array(
            [visibilities.real, visibilities.imag, np.ones_like(visibilities.imag)]
        )
        uvdata_arr = visibilities_real_imag_weight.T.reshape(
            visibilities.shape[1], 1, 1, 1, visibilities.shape[0], 1, 3
        )
        self.set_data(uvdata_arr, cube.with_spectral_unit(u.Hz).spectral_axis)
        self.frequencies = cube.spectral_axis

    @property
    @u.quantity_input
    def visibility(self) -> u.Jy:
        """Returns complex array of visibities"""
        return self.re + self.im * 1j

    def pickle(self, filename: PathLike = None):
        """Pickle the UV coordinates so they can be reloaded later

        Args:
            filename: Pathlike, default: self.path + ".pkl". Should end with ".pkl"`"
        """
        if filename is None:
            filename = str(self.path) + ".pkl"
        else:
            if not filename.lower().endswith(".pkl"):
                self.logger.warning("If the pickle filename does not end with .pkl, "
                                    "it won't be automatically recognized")
        with open(filename, "wb") as uvpkl:
            pickle.dump((self.u, self.v), uvpkl)

    def chi2_with(self, data=Union[PathLike, spectral_cube.SpectralCube], check=False) -> float:
        """Method to calculate chi-squared of a given UV set with a data cube

        Args:
            data: PathLike -- path to data cube readable by SpectralCube.read OR the spectral cube itself
        """
        # print("Vis before: ", self.visibility)
        if not isinstance(data, spectral_cube.SpectralCube):
            data = spectral_cube.SpectralCube.read(data)

        if data.spectral_axis.unit != self.frequencies.unit:
            data = data.with_spectral_unit(self.frequencies.unit, velocity_convention="radio")

        self.logger.debug("Data spectral axis: %s", data.spectral_axis)
        self.logger.debug("UVTable spectral axis: %s", self.frequencies)
        if (data.spectral_axis.shape != self.frequencies.shape) or (not np.all(np.equal(data.spectral_axis, self.frequencies))):
            self.logger.info("Interpolate data to UVTable spectral grid")
            data = data.spectral_interpolate(spectral_grid=self.frequencies)

        pixel_area_units = u.Unit(data.wcs.celestial.world_axis_units[0]) \
                           * u.Unit(data.wcs.celestial.world_axis_units[1])
        pixel_area = astropy.wcs.utils.proj_plane_pixel_area(data.wcs.celestial) * pixel_area_units
        dxy = np.sqrt(pixel_area).to_value(u.rad)
        self.logger.debug("Data pixel area %s and size %s radian", pixel_area, dxy)
        data = (data.to(u.Jy / u.sr) * pixel_area).to(u.Jy)
        self.chi_per_channel = []
        for cube_slice, _wavelength, _re, _im, _weight in zip(
                data, self.wavelengths, self.re.T, self.im.T, self.weight.T
        ):
            chi_per_channel = g_double.chi2Image(
                image=cube_slice,
                dxy=dxy,
                u=(self.u / _wavelength).to_value(u.dimensionless_unscaled),
                v=(self.v / _wavelength).to_value(u.dimensionless_unscaled),
                vis_obs_re=_re.astype('float64').to_value(u.Jy),
                vis_obs_im=_im.astype('float64').to_value(u.Jy),
                vis_obs_w=_weight.astype('float64').to_value(u.Jy ** -2),
                check=check
            )
            self.chi_per_channel.append(chi_per_channel)
        return np.sum(self.chi_per_channel)

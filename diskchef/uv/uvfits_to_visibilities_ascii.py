"""Module with functions to convert GILDAS UVTable to GALARIO visibilities format"""
import math
import pickle

import os
import textwrap
from pathlib import Path
from typing import Union, Literal, Sequence, List
import logging
import subprocess

import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.table import Table, QTable
import astropy.io.fits
import astropy.wcs
from matplotlib import pyplot as plt
import matplotlib.axes
import matplotlib.figure
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

    def __init__(
            self,
            path: PathLike,
            channel: Union[int, Sequence, slice, Literal['all']] = 'all',
            sum: bool = False
    ):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

        self.path = os.path.abspath(path)
        self.restfreq = None
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
            self.restfreq = self._fits.meta.get("RESTFREQ", None)
        elif self.path.lower().endswith(".pkl"):
            self._fits = None
            with open(self.path, "rb") as pkl:
                self.u, self.v = pickle.load(pkl)
                self.re = np.empty_like(self.u)
                self.im = np.empty_like(self.u)
                self.weight = np.empty_like(self.u)
                self.table = astropy.table.QTable()
                self._update_table()
            self.restfreq = self._fits.meta.get("RESTFREQ", None)
        else:
            raise CHEFValueError(
                "Unknown file format: "
                "expecting '.fits' / '.uvfits' for casa-style UVFITS file "
                "or '.pkl' for previously pickled UVFits instance"
            )

    kl = u.def_unit('kλ', 1000 * u.dimensionless_unscaled)

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
    def rest_freq(self) -> u.Hz:
        return self._fits.meta.get("RESTFREQ", None) * u.Hz

    @property
    def wavelengths(self):
        return c.c / self.frequencies

    def plot_uvgrid(
            self,
            axes: matplotlib.axes.Axes = None,
            kl: bool = False, restfreq: u.Hz = None,
            symsize: float = 1,
            **kwargs
    ) -> matplotlib.axes.Axes:
        """

        Args:
            axes: axes to draw the map on
            kl: if True, plots in dimensionless [kλ], thousands of wavelength. If more than one frequency is present,
            and restfreq is not set, the mean is taken
            restfreq: u.spectral - equivalent unit, the rest frequency to use when kl is True
            kwargs: to be passed to axes.scatter

        Returns:
            axes with the uv grid plotted
        """
        if axes is None:
            _, axes = plt.subplots(1)
            axes.set_aspect('equal')

        xplot = u.Quantity([*self.u, *(-self.u)])
        yplot = u.Quantity([*self.v, *(-self.v)])

        if restfreq is None:
            restfreq = np.median(self.frequencies)
        restwl = restfreq.to(u.m, equivalencies=u.spectral())
        if kl:
            xplot = (xplot / restwl).to(self.kl)
            yplot = (yplot / restwl).to(self.kl)

        sizes = symsize * self.weight[:, 0] / (self.weight.max())
        sizes = np.array([*sizes, *sizes])
        axes.invert_xaxis()
        axes.scatter(xplot, yplot, alpha=0.5, s=sizes, **kwargs)
        return axes

    def plot_uv_channel_map(
            self,
            nx: int = None, ny: int = None,
            channels: Union[range, List[int], Literal[Ellipsis]] = Ellipsis,
            subplot_kw: dict = None,
            gridspec_kw: dict = (("hspace", 0), ("wspace", 0)),
            fig_kw: dict = None,
            symsize: float = 0.5,
            **kwargs
    ) -> matplotlib.figure.Figure:
        gridspec_kw = dict(gridspec_kw)
        data_to_plot = self.data[:, channels, :]
        re = data_to_plot[:, :, 0]
        im = data_to_plot[:, :, 1]

        frequencies = self.frequencies[channels]
        nchannels = re.shape[1]

        if nx is None and ny is not None:
            nx = math.ceil(nchannels / ny)
        elif nx is not None and ny is None:
            ny = math.ceil(nchannels / nx)
        elif nx is not None and ny is not None:
            if nx * ny > nchannels:
                raise CHEFValueError(f"Inconsistent number of channels: nx * ny = {nx * ny} > nchannels = {nchannels}")
        else:
            ny = nx = math.ceil(math.sqrt(nchannels))

        if fig_kw is None: fig_kw = {}
        fig, axes = plt.subplots(
            ny, nx, squeeze=False, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
            sharex="all", sharey="all",
            **fig_kw
        )
        axes.flatten()[0].invert_xaxis()
        for ax in axes.flatten(): ax.set_visible(False)
        xplot = u.Quantity([*self.u, *(-self.u)])
        yplot = u.Quantity([*self.v, *(-self.v)])

        visibility = re + 1j * im
        visibility = [*visibility, *visibility] * u.Jy
        color = np.angle(visibility)
        sizes = np.abs(visibility)
        sizes /= sizes.max() * symsize

        for vis, freq, s, col, ax in zip(visibility.T, frequencies, sizes.T, color.T, axes.flatten()):
            ax.set_visible(True)
            restwl = freq.to(u.m, equivalencies=u.spectral())
            xxplot = (xplot / restwl).to(self.kl)
            yyplot = (yplot / restwl).to(self.kl)
            ax.scatter(xxplot, yyplot, s=s, c=col, cmap="twilight", **kwargs)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        axes[ny - 1, 0].get_xaxis().set_visible(True)
        axes[ny - 1, 0].get_yaxis().set_visible(True)

        return fig

    def plot_total_power(
            self,
            rest_freq: u.Hz = None,
            axes: matplotlib.axes.Axes = None,
            axes_unit: Union[u.Unit, Literal["channel"]] = None,
    ) -> matplotlib.axes.Axes:
        """Plot the sum of visibilities collected by the interferometer.

        This is not actually the total power, but gives the idea for the bright lines

        Args:
            rest_freq: u.Hz - the rest frequency of the line to convert axes to km/s
            axes: matplotlib.axes.Axes - the axes to plot on
            axes_unit: u.Unit - the unit to convert the axes to, defaults to the original axes unit (likely Hz)
        """
        if rest_freq is not None:
            frequencies = self.frequencies.to(u.km / u.s, equivalencies=u.doppler_radio(rest_freq))
        else:
            frequencies = self.frequencies
        if axes_unit is not None:
            if axes_unit == "channel":
                frequencies = np.arange(len(frequencies))
            else:
                frequencies = frequencies.to(axes_unit)
        if axes is None:
            _, axes = plt.subplots(1)
        axes.plot(
            frequencies,
            np.sum(np.abs((self.visibility * self.weight) / self.weight.sum()), axis=0)
        )
        return axes

    def image_to_visibilities(self, cube: Union[spectral_cube.SpectralCube, PathLike]):
        """Import cube from a FITS `file`, sample it with visibilities of this UVFITS"""
        if isinstance(cube, spectral_cube.SpectralCube):
            pass
        else:
            cube = spectral_cube.SpectralCube.read(cube)
        pixel_area_units = u.Unit(cube.wcs.celestial.world_axis_units[0]) * u.Unit(
            cube.wcs.celestial.world_axis_units[1])
        pixel_area = astropy.wcs.utils.proj_plane_pixel_area(cube.wcs.celestial) * pixel_area_units
        dxy = np.sqrt(pixel_area).to_value(u.rad)
        cube = (cube.to(u.Jy / u.sr) * pixel_area).to(u.Jy)

        visibilities = []
        for i, frequency in enumerate(
                cube.with_spectral_unit(u.Hz, velocity_convention="radio").spectral_axis
        ):
            wl = (c.c / frequency).si
            u_wavelengths = (self.u / wl).to_value(u.dimensionless_unscaled)
            v_wavelengths = (self.v / wl).to_value(u.dimensionless_unscaled)
            vis = g_double.sampleImage(cube[i], dxy, u_wavelengths, v_wavelengths, origin='lower')
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

    def chi2_with(self, data=Union[PathLike, spectral_cube.SpectralCube], threads: int = None, **kwargs) -> float:
        """Method to calculate chi-squared of a given UV set with a data cube

        Args:
            data: PathLike -- path to data cube readable by SpectralCube.read OR the spectral cube itself
            threads: int -- number of threads for galario. Sets globally until called again with non-default value.
        """
        if threads is not None:
            g_double.threads(threads)
        if not isinstance(data, spectral_cube.SpectralCube):
            data = spectral_cube.SpectralCube.read(data)

        if data.spectral_axis.unit != self.frequencies.unit:
            data = data.with_spectral_unit(self.frequencies.unit, velocity_convention="radio")

        self.logger.debug("Data spectral axis: %s", data.spectral_axis)
        self.logger.debug("UVTable spectral axis: %s", self.frequencies)
        if (data.spectral_axis.shape != self.frequencies.shape) or (
                not np.all(np.equal(data.spectral_axis, self.frequencies))):
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
                origin='lower',
                **kwargs
            )
            self.chi_per_channel.append(chi_per_channel)
        return float(np.sum(self.chi_per_channel))

    @classmethod
    def write_visibilities_to_uvfits(
            cls,
            data_to_write: Union[PathLike, spectral_cube.SpectralCube],
            file_to_modify: PathLike,
            output_filename: PathLike = None,
            uv_kwargs: dict = None,
            residual: bool = False,
    ) -> PathLike:
        """
        Replace visibilities of `file_to_modify` to visibilities sampled from `data_to_write`,
        save into `output_filename`.

        Args:
            data_to_write: PathLike or SpectralCube, data to replace the original table
            file_to_modify: PathLike, uvfits file with visibilities and frequencies to use as a reference
            output_filename: PathLike, optional: name of new uvfits file, `file_to_modify`.modified.uvfits by default
            uv_kwargs: kwargs to be passed to UVFits initialization, ie `sum` or `channel`
            residual: if True, write input residuals `data_to_be_modified - data_to_write` instead of `data_to_write`
        Returns:
            output file name

        Usage:
            >>> UVFits.write_visibilities_to_uvfits(  # doctest: +SKIP
            ...     data_to_write="13CO J=2-1_image.fits",
            ...     file_to_modify="13co.uvfits",
            ...     output_filename="13co_model.uvfits",
            ...     uv_kwargs={'sum': False}
            ... )
        """
        if uv_kwargs is None:
            uv_kwargs = {}
        if output_filename is None:
            output_filename = Path(f"{file_to_modify}.modified.uvfits")

        if not isinstance(data_to_write, spectral_cube.SpectralCube):
            data_to_write = spectral_cube.SpectralCube.read(data_to_write)

        uvfits = cls(file_to_modify, **uv_kwargs)
        uvfits.image_to_visibilities(
            data_to_write.spectral_interpolate(uvfits.frequencies)
        )
        data_to_write_array = np.array(uvfits.data)

        hdul_to_modify = astropy.io.fits.open(file_to_modify, lazy_load_hdus=False, memmap=False)
        data_to_be_modified = hdul_to_modify[0].data["DATA"][:, 0, 0, 0, :, 0, :]
        if residual:
            data_to_be_modified[:, :, 0:2] -= data_to_write_array[:, :, 0:2]
        else:
            data_to_be_modified[:, :, 0:2] = data_to_write_array[:, :, 0:2]

        hdul_to_modify.writeto(output_filename, overwrite=True)

        return output_filename

    @classmethod
    def run_gildas_script(
            cls,
            script: str = "SAY HELLO WORLD",
            gildas_executable: PathLike = "imager",
            script_filename: PathLike = "last.imager",
            folder=None,
    ) -> subprocess.CompletedProcess:
        """
        Send script to GILDAS.

        Args:
            script: GILDAS script to run
            gildas_executable: path to GILDAS executable (imager, astro, etc.)
            script_filename: filename for script as it will be saved as a file
            folder: working directory for script

        Returns:
            subprocess.CompletedProcess instance with .stdout and .stderr attributes
        """
        if folder is None:
            folder = Path.cwd()

        script_filename = Path(script_filename)
        with open(script_filename, "w") as fff:
            fff.write(textwrap.dedent(script))

        proc = subprocess.run(
            f'cat {script_filename.resolve()} | {gildas_executable} -nw',
            capture_output=True, encoding='utf8', shell=True,
            cwd=folder
        )
        return proc

    @classmethod
    def run_imaging(
            cls,
            input_file: PathLike,
            name: str,
            imager_executable: PathLike = "imager",
            script_template: str = """
                        FITS {input_file} TO {name}
                        READ UV {name}
                        UV_MAP
                        CLEAN
                        LUT {lut}
                        ! VIEW CLEAN /NOPAUSE
                        LET SIZE 10
                        LET DO_CONTOUR NO
                        SHOW CLEAN
                        HARDCOPY {name}.{device} /DEVICE {device} /OVERWRITE
                """,
            script_filename: PathLike = "last.imager",
            device: Union[Literal["pdf", "png", "eps", "ps"], str] = "pdf",
            lut: str = "inferno",
            **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Send script to GILDAS. The primary use is to send an imaging script to IMAGER

        Args:
            input_file: path to .uvfits file to be imaged, passed to `script_template.format
            name: basename for the output files, passed to `script_template.format
            imager_executable: path to imager executable, if not in system PATH
            script_template: string containing script with parameters listed in {} for .format
            script_filename: the filename for the created script
            device: DEVICE keyword for `[GTVL\]HARDCOPY`, the output figure file format,
                passed to `script_template.format
            lut: argument for `[GTVL\]LUT` for colormap setting, `inferno` by default, passed to `script_template.format
            **kwargs: other arguments passed to `script_template.format`

        Returns:
            subprocess.CompletedProcess instance with .stdout and .stderr attributes

        Usage:
            >>> UVFits.run_imaging(input_file="model.uvfits", name="model")  # doctest: +SKIP
        """

        script_filename = Path(script_filename)
        input_file = Path(input_file)

        proc = cls.run_gildas_script(
            script=script_template.format(
                input_file=input_file.name,
                name=name,
                lut=lut,
                device=device,
                **kwargs
            ),
            gildas_executable=imager_executable,
            script_filename=script_filename,
            folder=input_file.parent
        )
        return proc

    @classmethod
    def run_gildas(cls, *args, **kwargs):
        """Deprecated, renamed to run_imaging. Alsom see run_gildas_script to run arbitrary GILDAS scripts."""
        warnings.warn("run_gildas is deprecated, use run_imaging instead", DeprecationWarning)
        return cls.run_imaging(*args, **kwargs)

import os
import math
from warnings import warn

import numpy as np
from astropy import units as u
import astropy.wcs
import spectral_cube

import galario

if galario.HAVE_CUDA:
    from galario import double_cuda as g_double
    from galario import single_cuda as g_single
else:
    from galario import double as g_double
    from galario import single as g_single

import radmc3dPy

from diskchef.uv.uvfits_to_visibilities_ascii import UVFits
from diskchef.engine.exceptions import CHEFValueWarning


class Residual:
    """
    Class that calculates the residual between data visibilities and model

    Notes:
        There is no check whether wavelength axes of data and model match

    Usage:

    >>> data = UVFits(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "s-Wide-1+C.uvfits"), 'all', sum=True)
    >>> model = radmc3dPy.image.readImage(
    ...     os.path.join(os.path.dirname(__file__), "..", "tests", "data", "radmc", "image.out")
    ... )  # doctest: +ELLIPSIS
    Reading ...image.out
    >>> residual = Residual(model=model, data=data, distance=150 * u.pc)
    >>> math.isclose(residual.chi, 2162.243303848548, rel_tol=1e-4)
    True

    # >>> data = UVFits(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "s-Line-22-CO_1+D.uvfits"), 'all', sum=False)
    # >>> model = radmc3dPy.image.readImage(
    # ...     os.path.join(os.path.dirname(__file__), "..", "tests", "data", "radmc", "image.out")
    # ... )  # doctest: +ELLIPSIS
    # Reading ...image.out
    # >>> residual = Residual(model=model, data=data, distance=150 * u.pc)
    # >>> math.isclose(residual.chi, 2162.243303848548, rel_tol=1e-4)
    # True
    """

    def __init__(
            self,
            model: radmc3dPy.image.radmc3dImage,
            data: UVFits,
            distance: u.pc,
    ):
        if not np.allclose(data.frequencies.to(u.Hz).value, model.freq, atol=1e-7):
            warn(f"Frequency axes of model and data do not match!\n{model.freq}\n{data.frequencies.to(u.Hz).value}",
                 CHEFValueWarning)
        self.model = model
        self.data = data
        self.distance = distance

    @property
    def chi(self):
        model = self.model
        data = self.data
        distance = self.distance

        self.chi_per_channel = [
            g_double.chi2Image(
                image=model.imageJyppix[:, :, channel],
                # TODO refactor this to construct the cube at same frequencies as data
                dxy=(model.sizepix_x * u.cm / distance).to(u.dimensionless_unscaled).value,
                u=(data.u / wavelength).to(u.dimensionless_unscaled).value,
                v=(data.v / wavelength).to(u.dimensionless_unscaled).value,
                vis_obs_re=re.astype('float64').to(u.Jy).value,
                vis_obs_im=im.astype('float64').to(u.Jy).value,
                vis_obs_w=weight.astype('float64').to(u.Jy ** -2).value,
            )
            for channel, wavelength, re, im, weight in zip(
                range(len(data.wavelengths)), data.wavelengths, data.re.T, data.im.T, data.weight.T
            )
        ]
        return sum(self.chi_per_channel)

    def interpolate_model_to_data_grid(self):
        model_all_frequencies = self.model.freq
        data_frequencies = self.data.frequencies
        model_indices = (data_frequencies.min() <= model_all_frequencies) \
                        & (model_all_frequencies <= data_frequencies.max())

        model_array = self.model.imageJyppix[:, :, model_indices]
        model_frequencies = model_all_frequencies[model_indices]

        wcs_dict = dict(
            CRVAL1=0,
            CTYPE1='RA---CAR',
            CRVAL2=0,
            CTYPE2='DEC--CAR',
            CDELT1=(self.model.sizepix_x * u.cm / self.distance).to(u.deg,
                                                                    equivalencies=u.dimensionless_angles()).value,
            CDELT2=(self.model.sizepix_y * u.cm / self.distance).to(u.deg,
                                                                    equivalencies=u.dimensionless_angles()).value,
            CUNIT1='deg',
            CUNIT2='deg',
            NAXIS1=self.model.imageJyppix.shape[0],
            NAXIS2=self.model.imageJyppix.shape[1],
            CTYPE3='FREQ',
            CUNIT3='Hz',
            CDELT3=(model_frequencies[1] - model_frequencies[0]).to(u.Hz).value,
            CRVAL3=model_frequencies[0].to(u.Hz).value,
            CRPIX3=1,
            NAXIS3=len(model_frequencies),
        )
        wcs = astropy.wcs.WCS(wcs_dict)

        cube = spectral_cube.SpectralCube(
            data=model_array << u.Jy/u.pix,
            wcs=wcs
        )
        cube_interpolated = cube.with_mask(cube != np.nan * u.Jy / u.pix).spectral_interpolate(data_frequencies)
        cube_interpolated.write("interpolated.fits")

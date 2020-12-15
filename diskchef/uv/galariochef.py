import os
import math
from warnings import warn

import numpy as np
from astropy import units as u

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

    >>> data = UVFits(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "s-Line-22-CO_1+D.uvfits"), 'all', sum=False)
    >>> model = radmc3dPy.image.readImage(
    ...     os.path.join(os.path.dirname(__file__), "..", "tests", "data", "radmc", "image.out")
    ... )  # doctest: +ELLIPSIS
    Reading ...image.out
    >>> residual = Residual(model=model, data=data, distance=150 * u.pc)
    >>> math.isclose(residual.chi, 2162.243303848548, rel_tol=1e-4)
    True
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
        self.chi_per_channel = [
            g_double.chi2Image(
                image=model.imageJyppix[:, :, channel],
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
        self.chi = sum(self.chi_per_channel)

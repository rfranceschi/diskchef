import os
import math

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


class Residual:
    """
    Class that calculates the residual between data visibilities and model

    Usage:

    >>> data = UVFits(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "s-Wide-1+C.uvfits"))
    >>> model = radmc3dPy.image.readImage(
    ...     os.path.join(os.path.dirname(__file__), "..", "tests", "data", "radmc", "image.out")
    ... )  # doctest: +ELLIPSIS
    Reading ...image.out
    >>> residual = Residual(model=model, data=data, distance=150 * u.pc)
    >>> math.isclose(residual.chi, 100.991107, rel_tol=1e-4)
    True
    """

    def __init__(
            self,
            model: radmc3dPy.image.radmc3dImage,
            data: UVFits,
            distance: u.pc,
    ):
        self.chi = g_double.chi2Image(
            image=model.imageJyppix[:, :, 0],
            dxy=(model.sizepix_x * u.cm / distance).to(u.dimensionless_unscaled).value,
            u=(data.u / data.wavelength).to(u.dimensionless_unscaled).value,
            v=(data.v / data.wavelength).to(u.dimensionless_unscaled).value,
            vis_obs_re=data.re.to(u.Jy).value,
            vis_obs_im=data.im.to(u.Jy).value,
            vis_obs_w=data.weight.to(u.dimensionless_unscaled).value,
        )

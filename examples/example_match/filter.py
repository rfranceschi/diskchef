import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube

from VISIBLE import matched_filter

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
logging.info(sys.version)

folder = Path(".")

# Reading data  TODO upload and make download link -- also create uvfits with gildas
data = folder / "s-Line-18-13CO_1+D_casa.ms"
filter_fits = folder / "DN Tau" / "radmc_gas" / "13CO J=2-1_image.fits"

match = matched_filter(
    filterfile=str(filter_fits),
    datafile=str(data),
)

# uvdata = UVFits(data, sum=False)
# uvdata_ref = UVFits(data, sum=False)

# model = SpectralCube.read(folder / "DN Tau" / "radmc_gas" / "13CO J=2-1_image.fits")
#
# model_kms = model.with_spectral_unit(u.km / u.s, velocity_convention="radio")
# uv_kms = uvdata.frequencies.to(u.km / u.s, equivalencies=u.doppler_radio(rest=model.header["RESTFREQ"] * u.Hz))
# uv_grid_in_model_range = uv_kms[(uv_kms >= model_kms.spectral_axis.min()) & (uv_kms <= model_kms.spectral_axis.max())]
# model_interpolated = model_kms.spectral_interpolate(uv_grid_in_model_range)
# uvdata.image_to_visibilities(model_interpolated.with_spectral_unit(u.Hz, velocity_convention="radio"))
#
# shift = uvdata_ref.data.shape[1] - uvdata.data.shape[1]
# diff = [
#     np.sum(uvdata_ref.weight[:, s:s + len(uvdata.frequencies)] * (
#                 (uvdata_ref.re[:, s:s + len(uvdata.frequencies)] - uvdata.re) ** 2 + (
#                     uvdata_ref.im[:, s:s + len(uvdata.frequencies)] - uvdata.im) ** 2))
#     for s in range(shift)
# ]
#
# plt.plot(uv_kms, [1] * len(uv_kms), ".", label="uvdata")
# plt.plot(model_kms.spectral_axis, [2] * len(model_kms.spectral_axis), ".", label="model")
# plt.plot(uv_grid_in_model_range, [1.5] * len(uv_grid_in_model_range), ".", label="model")
# plt.legend()
# plt.savefig("spectral.png")
# plt.close()
# plt.plot(-np.array(diff))
# plt.savefig("chi2.png")
import sys
from pathlib import Path
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import astropy.units as u

# NB: ultranest requires h5py
import ultranest

import logging

from diskchef.lamda.line import Line
from diskchef.physics.yorke_bodenheimer import YorkeBodenheimer2008
from diskchef.model.model import Model
from diskchef.uv.uvfits_to_visibilities_ascii import UVFits

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s (%(relativeCreated)10d)| %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)
    logging.info(sys.version)

    mass = 0.7 * u.M_sun

    yb = YorkeBodenheimer2008()

    lines = [  # Line(name='CO J=2-1', transition=2, molecule='CO'),
        Line(name='13CO J=2-1', transition=2, molecule='13CO')]

    # demo_model = Model(
    #     disk="Reference",
    #     line_list=lines,
    #     params=dict(r_min=1 * u.au, r_max=300 * u.au, radial_bins=100, vertical_bins=100),
    #     rstar=yb.radius(mass),
    #     tstar=yb.effective_temperature(mass),
    #     inc=30 * u.deg,
    #     PA=25 * u.deg,
    #     distance=150 * u.pc,
    #     npix=120,
    #     channels=21,
    #     run_mctherm=False
    # )

    refmodel_13CO = 'Reference/radmc_gas/13CO J=2-1_image.fits'
    refmodel_12CO = 'Reference/radmc_gas/CO J=2-1_image.fits'

    uvcoverage = UVFits("demo.pkl")

    # uvdata_from_pickle.plot_uvgrid()
    # plt.show()

    uvcoverage.image_to_visibilities(refmodel_13CO)

    chisq = uvcoverage.chi2_with(refmodel_13CO)
    print('Chi^2 of the model with itself: ', chisq)

    parameters = ['tapering_radius', 'gas_mass']


    def prior_transform(quantile_cube):
        params = quantile_cube.copy()

        # Assume it's a nice, flat prior, between 30 and 200 au:
        lower = 30.
        upper = 200.

        params[0] = quantile_cube[0] * (upper - lower) + lower

        # Assume it's a log-flat prior, between 1e-4 and 1e-2 M_sun:
        params[1] = 10 ** (quantile_cube[1] * 2 - 4)

        return params


    def my_likelihood(params):
        tapering_radius = params[0]
        gas_mass = params[1]

        with tempfile.TemporaryDirectory(prefix='fit_', dir='fit_tmp') as temp_dir:
            folder = Path(temp_dir)
            disk_model = Model(disk="Fit",
                               line_list=lines,
                               params=dict(r_min=1 * u.au, r_max=300 * u.au, radial_bins=60, vertical_bins=60,
                                           tapering_radius=tapering_radius * u.au, gas_mass=gas_mass * u.Msun),
                               rstar=yb.radius(mass),
                               tstar=yb.effective_temperature(mass),
                               inc=30 * u.deg,
                               PA=25 * u.deg,
                               distance=150 * u.pc,
                               npix=120,
                               channels=21,
                               run_mctherm=False,
                               folder=folder)
            chi2 = uvcoverage.chi2_with(folder / "radmc_gas" / "13CO J=2-1_image.fits")
        return -0.5 * chi2


    sampler = ultranest.ReactiveNestedSampler(parameters, my_likelihood, prior_transform,
                                              log_dir='testlog', resume=True)

    result = sampler.run(max_ncalls=100, min_num_live_points=10, cluster_num_live_points=5)

    paramnames = result['paramnames']
    data = np.array(result['weighted_samples']['points'])
    weights = np.array(result['weighted_samples']['weights'])
    cumsumweights = np.cumsum(weights)

    mask = cumsumweights > 1e-4

    import corner

    figure = corner.corner(data[mask, :], weights=weights[mask],
                           labels=paramnames, show_titles=True)
    figure.savefig('tt.png')

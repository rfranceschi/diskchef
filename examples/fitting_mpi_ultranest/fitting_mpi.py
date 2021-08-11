from typing import List
import logging

import numpy as np
from matplotlib import pyplot as plt

import mpi4py

from diskchef.fitting import Parameter, UltraNestFitter


def sqr(
        params: List[Parameter],
        x: np.ndarray
):
    return params[0] * x ** 2 + params[1] * x + params[2]


def sqr_model_lnprob(
        params: List[Parameter],
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray = None
):
    y_from_x = sqr(params, x)
    if weights is None:
        weights = np.ones_like(y)
    chi2 = np.sum(weights ** 2 * (y - y_from_x) ** 2)
    return -0.5 * chi2


def rescale_sqr(cube):
    params = np.empty_like(cube)
    lo = 0.1
    hi = 10
    params[0] = 10 ** (cube[0] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))
    lo = -2
    hi = 5
    params[1] = cube[1] * (hi - lo) + lo
    lo = -2
    hi = 5
    params[2] = cube[2] * (hi - lo) + lo
    return params


def example_sqr():
    x = np.linspace(1, 10, 100)
    y = sqr([1, 2, 0], x)

    a = Parameter(name="a", min=0.1, max=2, truth=1)
    b = Parameter(name="b", min=-2, max=5, truth=2)
    c = Parameter(name="c", min=-2, max=5, truth=0)
    params = [a, b, c]

    fitter = UltraNestFitter(lnprob=sqr_model_lnprob, parameters=params, threads=threads,
                             transform=rescale_sqr,  # this can be generated based on fields of Parameter class
                             resume=True,
                             storage_backend='hdf5',
                             run_kwargs={"max_ncalls": 500}
                             )

    bestfit = fitter.fit(x=x, y=y)
    if fitter.sampler.use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            fitter.corner()
    else:
        fitter.corner()

    plt.show()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    threads = 1
    example_sqr()

from typing import List
import logging

import numpy as np
from matplotlib import pyplot as plt

from diskchef.fitting import Parameter, EMCEEFitter, BruteForceFitter, UltraNestFitter


def linear(
        params: List[Parameter],
        x: np.ndarray
):
    return params[0] * x + params[1]


def linear_model_lnprob(
        params: List[Parameter],
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray = None
):
    y_from_x = linear(params, x)
    if weights is None:
        weights = np.ones_like(y)
    chi2 = np.sum(weights ** 2 * (y - y_from_x) ** 2)
    return -0.5 * chi2


def rescale_linear(cube):
    params = np.empty_like(cube)
    lo = 0.1
    hi = 10
    params[0] = 10 ** (cube[0] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))
    lo = -2
    hi = 5
    params[1] = cube[1] * (hi - lo) + lo
    return params


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


def example_lin():
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)

    a = Parameter(name="a", min=0.1, max=2, truth=1)
    b = Parameter(name="b", min=-2, max=5, truth=2)
    fitter = EMCEEFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = BruteForceFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads, n_points=100)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = UltraNestFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads, transform=rescale_linear)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    plt.show()


def example_sqr():
    x = np.linspace(1, 10, 100)
    y = sqr([1, 2, 0], x)

    a = Parameter(name="a", min=0.1, max=2, truth=1)
    b = Parameter(name="b", min=-2, max=5, truth=2)
    c = Parameter(name="c", min=-2, max=5, truth=0)
    params = [a, b, c]

    fitter = EMCEEFitter(lnprob=sqr_model_lnprob, parameters=params, threads=threads)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = BruteForceFitter(lnprob=sqr_model_lnprob, parameters=params, threads=threads, n_points=40)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    fitter = UltraNestFitter(lnprob=sqr_model_lnprob, parameters=params, threads=threads,
                             # transform=rescale_sqr # this can be generated based on fields of Parameter class
                             )
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    threads = 1
    example_lin()
    example_sqr()

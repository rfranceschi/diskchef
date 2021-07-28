from typing import List

import numpy as np
import pytest
from astropy import units as u

from diskchef.fitting.fitters import BruteForceFitter, Parameter, EMCEEFitter, UltraNestFitter
from matplotlib import pyplot as plt


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


@pytest.mark.parametrize(
    "threads",
    [
        1, 2, None
    ]
)
def test_linear_brute(threads):
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)
    assert linear_model_lnprob([1, 2], x, y) == 0
    a = Parameter(name="a", min=0.1, max=10)
    b = Parameter(name="b", min=-2, max=2)
    fitter = BruteForceFitter(lnprob=linear_model_lnprob, parameters=[a, b], n_points=100, threads=threads)
    bestfit = fitter.fit(x=x, y=y)
    assert bestfit["a"], bestfit["b"] == pytest.approx([1, 2])


@pytest.mark.parametrize(
    "threads",
    [
        1, 2, None
    ]
)
def test_linear_emcee(threads):
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)
    assert linear_model_lnprob([1, 2], x, y) == 0
    a = Parameter(name="a", min=0.1, max=10)
    b = Parameter(name="b", min=-2, max=2)
    fitter = EMCEEFitter(
        lnprob=linear_model_lnprob, parameters=[a, b],
        # threads=threads
    )
    bestfit = fitter.fit(x=x, y=y)
    assert bestfit["a"], bestfit["b"] == pytest.approx([1, 2])


def test_linear_ultranest():
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)
    assert linear_model_lnprob([1, 2], x, y) == 0
    a = Parameter(name="a", min=0.1, max=10)
    b = Parameter(name="b", min=-2, max=2)
    fitter = UltraNestFitter(
        lnprob=linear_model_lnprob, parameters=[a, b],
        transform=rescale_linear
    )
    bestfit = fitter.fit(x=x, y=y)
    assert bestfit["a"], bestfit["b"] == pytest.approx([1, 2])


if __name__ == '__main__':
    threads = 1
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)

    a = Parameter(name="a", min=0.1, max=2)
    b = Parameter(name="b", min=-2, max=5)
    fitter = EMCEEFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner(truths=[1, 2])

    fitter = BruteForceFitter(lnprob=linear_model_lnprob, parameters=[a, b], threads=threads, n_points=100)
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner(truths=[1, 2])

    plt.show()

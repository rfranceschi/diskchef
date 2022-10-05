import tempfile
import logging
from typing import List
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from diskchef.fitting.fitters import BruteForceFitter, Parameter, EMCEEFitter, UltraNestFitter, SciPyFitter
from matplotlib import pyplot as plt
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(scope="session")
def dirname():
    """Returns a Path object with a path to unique temporary directory"""
    today = date.today().strftime("%d.%m.%Y")
    dirname = Path(tempfile.mkdtemp(prefix=f"diskchef_test_{today}_"))
    print("Test outputs are in: ", dirname)
    return dirname

def linear(
        params: List[float],
        x: np.ndarray
):
    return params[0] * x + params[1]


def linear_model_lnprob(
        params: List[float],
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


@pytest.mark.parametrize(
    "threads",
    [
        1, 2, None
    ]
)
def test_linear_brute(threads, dirname):
    runname = f"brute_{threads}"
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)
    assert linear_model_lnprob([1, 2], x, y) == 0
    a = Parameter(name="a", min=0.1, max=10)
    b = Parameter(name="b", min=-2, max=5)
    fitter = BruteForceFitter(lnprob=linear_model_lnprob, parameters=[a, b], n_points=100, threads=threads)
    bestfit = fitter.fit(x=x, y=y)
    assert [bestfit["a"], bestfit["b"]] == [1, 2]

    fitter.save(dirname / f"{runname}.sav")
    loaded_fitter = BruteForceFitter.load(dirname / f"{runname}.sav")

    assert [loaded_fitter.parameters_dict["a"], loaded_fitter.parameters_dict["b"]] == [1, 2]

    fig = loaded_fitter.corner()
    fig.savefig(dirname / f"{runname}.pdf")

@pytest.mark.parametrize(
    "threads",
    [
        1, 2, None
    ]
)
def test_linear_emcee(threads, dirname):
    runname = f"emcee_{threads}"
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)
    assert linear_model_lnprob([1, 2], x, y) == 0
    a = Parameter(name="a", min=0.1, max=10)
    b = Parameter(name="b", min=-2, max=5)
    fitter = EMCEEFitter(
        lnprob=linear_model_lnprob, parameters=[a, b],
        threads=threads
    )
    bestfit = fitter.fit(x=x, y=y)
    assert [bestfit["a"], bestfit["b"]] == [1, 2]

    fitter.save(dirname / f"{runname}.sav")
    loaded_fitter = EMCEEFitter.load(dirname / f"{runname}.sav")

    assert [loaded_fitter.parameters_dict["a"], loaded_fitter.parameters_dict["b"]] == [1, 2]

    # not implemented
    # fig = loaded_fitter.corner()
    # fig.savefig(dirname / f"{runname}.pdf")

# WIP
@pytest.mark.parametrize(
    "threads",
    [
        1, 2, None
    ]
)
def test_linear_scipy(threads, dirname):
    runname = f"scipy_{threads}"
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)
    assert linear_model_lnprob([1, 2], x, y) == 0
    a = Parameter(name="a", min=0.1, max=10)
    b = Parameter(name="b", min=-2, max=5)
    fitter = SciPyFitter(
        lnprob=linear_model_lnprob, parameters=[a, b],
        threads=threads
    )
    bestfit = fitter.fit(x=x, y=y)
    # assert [bestfit["a"], bestfit["b"]] == [1, 2]

    fitter.save(dirname / f"{runname}.sav")
    loaded_fitter = EMCEEFitter.load(dirname / f"{runname}.sav")

    # assert [loaded_fitter.parameters_dict["a"], loaded_fitter.parameters_dict["b"]] == [1, 2]

    # fig = loaded_fitter.corner()
    # fig.savefig(dirname / f"{runname}.pdf")


@pytest.mark.parametrize(
    "threads",
    [
        1,
    ]
)
def test_linear_ultranest(threads, dirname):
    runname = dirname / f"ultranest_{threads}"
    x = np.linspace(1, 10, 100)
    y = linear([1, 2], x)
    assert linear_model_lnprob([1, 2], x, y) == 0
    a = Parameter(name="T_{atm}", min=0.1, max=10, truth=1)
    b = Parameter(name="log_{10}(M/M_\odot)", min=-2, max=5, truth=2)
    fitter = UltraNestFitter(
        lnprob=linear_model_lnprob, parameters=[a, b],
        transform=rescale_linear,
        threads=threads,
        log_dir=runname,
        resume="overwrite"
    )
    bestfit = fitter.fit(x=x, y=y)
    fig = fitter.corner()
    fig.savefig(runname/"corner.pdf")
    print(a.math_repr, b.math_repr)
    assert [bestfit["T_{atm}"], bestfit["log_{10}(M/M_\odot)"]] == [1, 2]

    fitter.save(str(runname) + ".sav")
    loaded_fitter = EMCEEFitter.load(str(runname) + ".sav")

    assert [
               loaded_fitter.parameters_dict["T_{atm}"], loaded_fitter.parameters_dict["log_{10}(M/M_\odot)"]
           ] == [1, 2]

    fig = loaded_fitter.corner()
    fig.savefig(str(runname) + ".pdf")


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
                             # transform=rescale_sqr
                             )
    bestfit = fitter.fit(x=x, y=y)
    fitter.corner()

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    threads = 1
    # example_lin()
    example_sqr()

import numpy as np
import pytest

from diskchef.fitting.fitters import UltraNestFitter, Parameter


def test_transform_function():
    parameters = [
        Parameter("a", min=10, max=20),
        Parameter("b", min=-5, max=15)
    ]
    fitter = UltraNestFitter(
        lnprob=lambda params: -((params[0] - 15) ** 2 + params[1] ** 2),
        parameters=parameters,
        log_dir="transform_test"
    )
    assert pytest.approx(fitter.transform(np.array([0.5, 0.2]))) == [15, -1]
    assert pytest.approx(fitter.transform(np.array([0.4, 0.8]))) == [14, 11]
    assert pytest.approx(fitter.transform(np.array([[0.5, 0.4], [0.2, 0.8]]))) == np.array([[15, 14], [-1, 11]])
    assert pytest.approx(fitter.transform([0.5, 0.2])) == [15, -1]
    assert fitter.transform(np.random.uniform(size=(2, 1))).shape == (2, 1)

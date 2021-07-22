import numpy as np
import pytest
from astropy import units as u, constants as c

from diskchef.maps.radiation_fields import draine1978, DRAINE_UV_FIELD, HABING_ISRF
from diskchef.maps.radiation_fields import weingartner_draine_2001, WEINGARTNER_DRAINE_ISRF, ANDES2_G0


@pytest.mark.parametrize(
    "wav",
    [
        np.geomspace(3e10, 6e15, 1000000) << u.Hz,
        np.geomspace(3e10, 6e15, 10000) << u.Hz,  # sparse grid
        np.geomspace(3e9, 6e16, 10000) << u.Hz,  # wider grid
        c.c / (np.geomspace(3e10, 6e15, 10000) << u.Hz),  # also wavelengths
        c.h * (np.geomspace(3e10, 6e15, 10000) << u.Hz),  # and energies
    ]
)
def test_total_field_draine(wav):
    field = draine1978(wav)
    total = np.trapz(field, wav.to(u.Hz, equivalencies=u.spectral()))
    assert (total * (4 * np.pi * u.sr)).to_value(DRAINE_UV_FIELD) == pytest.approx(1, rel=1e-3)


@pytest.mark.parametrize(
    "wav",
    [
        np.geomspace(6, 13.6, 1000) << u.eV,
    ]
)
def test_total_draine_habing(wav):
    field = draine1978(wav)
    total = np.trapz(field, wav.to(u.Hz, equivalencies=u.spectral()))
    assert (total * (4 * np.pi * u.sr)).to_value(HABING_ISRF) == pytest.approx(1.68, rel=1e-2)


@pytest.mark.parametrize(
    "wav",
    [
        np.geomspace(1.45079355e+15, 3.28846537e+15, 1000000) << u.Hz,
        np.geomspace(1.45079355e+15, 3.28846537e+15, 10000) << u.Hz,  # sparse grid
        c.c / (np.geomspace(1.45079355e+15, 3.28846537e+15, 10000) << u.Hz),  # also wavelengths
        c.h * (np.geomspace(1.45079355e+15, 3.28846537e+15, 10000) << u.Hz),  # and energies
    ]
)
def test_total_field_weingartner(wav):
    field = weingartner_draine_2001(wav)
    total = np.trapz(field, wav.to(u.Hz, equivalencies=u.spectral()))
    assert (total * (4 * np.pi * u.sr)).to_value(WEINGARTNER_DRAINE_ISRF) == pytest.approx(1., rel=1e-3)


@pytest.mark.parametrize(
    "wav",
    [
        np.geomspace(0.0912, 0.2000, 1000000) << u.um,
    ]
)
def test_total_field_weingartner_g_factor_andes(wav):
    field = weingartner_draine_2001(wav)
    total = np.abs(np.trapz(field, wav.to(u.Hz, equivalencies=u.spectral())))
    assert (total * (4 * np.pi * u.sr)).to_value(ANDES2_G0) == pytest.approx(1, rel=1e-3)

import numpy as np
from astropy import units as u
from astropy import constants
from astropy.modeling.physical_models import BlackBody


def _planck(e, t):
    """
    Blackbody Bnu function in cgs/eV units

    Args:
        e: photon energy, eV
        t: temperature, K

    Returns:
        Bnu
    """
    e = e << u.eV
    t = t << u.K
    nu = (e / constants.h).to(u.Hz)
    return (2 * constants.h * nu ** 3 / constants.c ** 2 / (np.exp(e / constants.k_B / t) - 1)).to(
        u.erg / u.s / u.cm ** 2 / u.Hz
    )


def draine1978(wav: u.um) -> u.erg / u.s / u.cm ** 2 / u.Hz / u.sr:
    """Return ISRF field according to Draine 1978

    https://ui.adsabs.harvard.edu/abs/1978ApJS...36..595D/abstract

    Args:
        wav: wavelength-equivalent in astropy.units

    Returns:
        intensity
    """
    energy_ev = wav.to(u.eV, equivalencies=u.spectral()).value

    photons = np.where(
        (energy_ev > 5) & (energy_ev < 13.6),
        1.658e6 * energy_ev - 2.152e5 * energy_ev ** 2 + 6.919e3 * energy_ev ** 3,
        0
    ) << (1 / (u.cm ** 2 * u.s * u.sr * u.eV))
    photon_energy = constants.h * wav.to(u.Hz, equivalencies=u.spectral())

    return (
            photons * photon_energy * constants.h
    ).to(u.erg / u.s / u.cm ** 2 / u.Hz / u.sr)


# MMP 1982/1983

def weingartner_draine_2001(wav: u.um) -> u.erg / u.s / u.cm ** 2 / u.Hz / u.sr:
    """Return ISRF field according to Weingartner, Draine 2001 Eq 31

    Based on Merger, Mathis, Panagia 1982 and Mathis, Mezger, Panagia 1983
    https://iopscience.iop.org/article/10.1086/320852/pdf

    Args:
        wav: wavelength-equivalent in astropy.units

    Returns:
        intensity
    """

    energy_ev = wav.to(u.eV, equivalencies=u.spectral()).value

    energy_density_nu_u_nu = np.piecewise(
        energy_ev,
        [
            energy_ev > 13.6,
            (energy_ev > 11.2) & (energy_ev <= 13.6),
            (energy_ev > 9.26) & (energy_ev <= 11.2),
            (energy_ev > 5.04) & (energy_ev <= 9.26),
            energy_ev <= 5.04
        ],
        [
            0,
            lambda e: 3.328e-9 * e ** (-4.4172),
            lambda e: 8.463e-13 * e ** (-1),
            lambda e: 2.055e-14 * e ** 0.6678,
            lambda e: (
                    (4 * np.pi * u.sr * (e * u.eV / constants.h) / constants.c)
                    * ((BlackBody(7500 * u.K, 1e-14)
                        + BlackBody(4000 * u.K, 1.65e-13)
                        + BlackBody(3000 * u.K, 4e-13))(e * u.eV)
                       )
            ).to_value(u.erg / u.cm ** 3)
        ]
    ) << (u.erg / u.cm ** 3)
    intensity_nu = energy_density_nu_u_nu \
                   / wav.to(u.Hz, equivalencies=u.spectral()) \
                   * constants.c.cgs / (4 * np.pi * u.sr)

    return intensity_nu


DRAINE_UV_FIELD = u.def_unit("G(Draine)", 3.0953728371735033e-06 * u.W / u.m ** 2)
"""Total intensity of Draine 1978 ISRF (own integration)"""

WEINGARTNER_DRAINE_ISRF = u.def_unit("G(Weingartner)", 1.802302355940787e-06 * u.W / u.m ** 2)
"""Intensity of Weingartner & Draine 2001 ISRF between 6 and 13.6 eV (own integration)"""

ANDES2_G0 = u.def_unit("G(ANDES2)", 1.7346147770208214e-06 * u.W / u.m ** 2)
"""Intensity of Weingartner & Draine 2001 ISRF between 0.0912 (~13.6) and 0.2000 (~6) um (eV)"""

HABING_ISRF = u.def_unit("G(Habing)", 1.5978938e-06 * u.W / u.m ** 2)
"""Total intensity of Habing 1968 between 6 and 13.6 eV, according to WD2001"""

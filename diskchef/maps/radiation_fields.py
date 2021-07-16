import numpy as np
from astropy import units as u
from astropy import constants


@u.quantity_input
def draine1978(wavelength: u.um) -> u.erg / u.s / u.cm ** 2 / u.Hz / u.sr:
    """Return ISRF field according to Draine 1978

    https://ui.adsabs.harvard.edu/abs/1978ApJS...36..595D/abstract

    Args:
        wavelength: wavelength in astropy.units

    Returns:
        intensity
    """
    energy_ev = wavelength.to(u.eV, equivalencies=u.spectral()).value

    photons = np.where(
        (energy_ev > 5) & (energy_ev < 13.6),
        1.658e6 * energy_ev - 2.152e5 * energy_ev ** 2 + 6.919e3 * energy_ev ** 3,
        0
    ) << (1 / (u.cm ** 2 * u.s * u.sr * u.eV))
    photon_energy = constants.h * wavelength.to(u.Hz, equivalencies=u.spectral())

    return (
            photons * photon_energy * constants.h
    ).to(u.erg / u.s / u.cm ** 2 / u.Hz / u.sr)

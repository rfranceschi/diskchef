"""
This module contains parametrizations of ioniaztion rates
"""
import diskchef.engine.other
import numpy as np
from scipy.interpolate import interp2d

DENSITIES = np.logspace(20, 25, 6)
TEMPERATURES = np.array([3e6, 1e7, 3e7, 1e8, 3e8])
BRUDERER_TABLE = np.array(  # from Bruderer+2009 Table3
    [
        [4.8e-11, 2.4e-11, 1.0e-11, 3.4e-12, 1.2e-12],
        [4.2e-12, 4.5e-12, 2.5e-12, 9.8e-13, 3.7e-13],
        [6.9e-14, 3.7e-13, 4.0e-13, 2.2e-13, 1.0e-13],
        [6.9e-17, 9.6e-15, 3.9e-14, 4.5e-14, 3.3e-14],
        [6.8e-22, 4.5e-17, 2.0e-15, 9.4e-15, 1.4e-14],
        [1.8e-30, 8.6e-22, 1.7e-17, 1.4e-15, 6.4e-15]
    ]
)
"""Array representing Bruderer+09 table 3"""
INTERPOLATED_LOG_BRUDERER_TABLE = diskchef.engine.other.unsorted_interp2d(
    np.log10(DENSITIES), np.log10(TEMPERATURES), np.log10(BRUDERER_TABLE.T), kind='cubic'
)
"""Interpolated table 3 of Bruderer+09"""


def bruderer09(column_density, temperature=3e6):
    """Return x-ray ionization rate as of Bruderer+09 table 3

    https://iopscience.iop.org/article/10.1088/0067-0049/183/2/179/pdf

    Args:
        column_density: number of protons + neutrons from the source, cm**-2 (1e20..1e25)
        temperature: plasma temperature of the source, K (3e6..3e8)

    Returns:
        ionization rate (float, in 1/s) for Fx = 1 erg/s
    """
    log_column_density = np.log10(column_density)
    log_temperature = np.log10(temperature)
    out = 10 ** INTERPOLATED_LOG_BRUDERER_TABLE(
        log_column_density, log_temperature, assume_sorted=False
    )
    if len(out) == 1:
        return out[0]
    return out


def padovani18l(logdens):
    """Return galactic cosmic ray ionization rate as of Padovani+18 App. F model L

    https://www.aanda.org/articles/aa/pdf/2018/06/aa32202-17.pdf

    Args:
        logdens: log10 of number density in g/cm**2 from the source

    Return:
        ionization rate, in 1/s
    """
    logdens = np.where(logdens < 19, 19, logdens)
    logdens = np.where(logdens > 27, 27, logdens)
    return 10 ** (
            -3.331056497233e6 +
            logdens * 1.207744586503e6
            - logdens ** 2 * 1.913914106234e5
            + logdens ** 3 * 1.731822350618e4
            - logdens ** 4 * 9.790557206178e2
            + logdens ** 5 * 3.543830893824e1
            - logdens ** 6 * 8.034869454520e-1
            + logdens ** 7 * 1.048808593086e-2
            - logdens ** 8 * 6.188760100997e-5
            + logdens ** 9 * 3.122820990797e-8
    )

def padovani18h(logdens):
    """Return galactic cosmic ray ionization rate as of Padovani+18 App. F model H

    https://www.aanda.org/articles/aa/pdf/2018/06/aa32202-17.pdf

    Args:
        logdens: log10 of number density in g/cm**2 from the source

    Return:
        ionization rate, in 1/s
    """
    logdens = np.where(logdens < 19, 19, logdens)
    logdens = np.where(logdens > 27, 27, logdens)
    return 10 ** (
            1.001098610761e7 +
            - logdens * 4.231294690194e6
            + logdens ** 2 * 7.921914432011e5
            - logdens ** 3 * 8.623677095423e4
            + logdens ** 4 * 6.015889127529e3
            - logdens ** 5 * 2.789238383353e2
            + logdens ** 6 * 8.595814402406e0
            - logdens ** 7 * 1.698029737474e-1
            + logdens ** 8 * 1.951179287567e-3
            - logdens ** 9 * 9.937499546711e-6
    )

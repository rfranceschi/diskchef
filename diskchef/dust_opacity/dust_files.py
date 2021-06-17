"""Functions to get the file from LAMDA database"""
from glob import glob

import os
import re
from typing import List, Literal

from astropy import units as u

from diskchef.engine.exceptions import CHEFValueError

DUST_OPACITY_FILES = glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "*.inp"))
"""Unsorted list of paths to all dust file. Use `file` to find a file for the species"""


def dust_files(dust_name: Literal["draine03", "diana", "pyr"] = "draine03", size=1e-5 * u.cm) -> List[str]:
    """
    Returns a list of absolute paths to matched dust opacity dust_files

    Args:
        dust_name: dust name

    Usage:

    >>> opacity_files = dust_files("draine03", 1e-5 * u.cm)
    >>> len(opacity_files)
    1
    >>> opacity_files[0].endswith("dustkapscatmat_astrosilicate_draine03_1.0e-05cm.inp")
    True
    >>> dust_files("diana")[0]  #doctest: +ELLIPSIS
    '...dustkapscatmat_diana.inp'
    >>> dust_files("pyr")[0]  #doctest: +ELLIPSIS
    '...dustkapscatmat_pyr.inp'
    """
    if dust_name == "draine03":
        regexp = re.compile(rf".*{dust_name}_{size.to(u.cm).value:.1e}cm\.inp$")
        matching_files = [file for file in DUST_OPACITY_FILES if re.search(regexp, file)]
    elif dust_name in {"diana", "pyr"}:
        matching_files = [file for file in DUST_OPACITY_FILES if re.search(rf'.*dustkapscatmat_{dust_name}.inp$', file)]
    else:
        raise CHEFValueError(f"dust_name {dust_name} is not recognized!")
    return matching_files

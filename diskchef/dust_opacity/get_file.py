"""Functions to get the file from LAMDA database"""
from glob import glob
import os
import re
from typing import List, Literal

from astropy import units as u

from diskchef.engine.exceptions import CHEFValueError

DUST_OPACITY_FILES = glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "*.inp"))
"""Unsorted list of paths to all dust file. Use `file` to find a file for the species"""


def file(dust_name: Literal["draine03"] = "draine03", size=1e-5 * u.cm) -> List[str]:
    """
    Returns a list of absolute paths to matched dust opacity file

    Args:
        dust_name: dust name

    Usage:

    >>> opacity_files = file("draine03", 1e-5 * u.cm)
    >>> len(opacity_files)
    1
    >>> opacity_files[0].endswith("dustkapscatmat_astrosilicate_draine03_1.0e-05cm.inp")
    True
    """
    if dust_name == "draine03":
        regexp = re.compile(rf".*{dust_name}_{size.to(u.cm).value:.1e}cm\.inp$")
        matching_files = [file for file in DUST_OPACITY_FILES if re.search(regexp, file)]
    else:
        raise CHEFValueError(f"dust_name {dust_name} is not recognized!")
    return matching_files

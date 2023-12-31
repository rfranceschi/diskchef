"""Functions to get the file from LAMDA database"""
from glob import glob

import os
import re
from typing import List

LAMDA_FILES = glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files', "*.dat"))
"""Unsorted list of paths to all LAMDA database file. Use `file` to find a file for the species"""


def lamda_files(species: str) -> List[str]:
    """
    Returns a list of absolute paths to matched LAMDA database file

    The right file from this list should be copied to or linked in the radiative transfer directory

    Args:
        species: species name

    Usage:

    >>> out = lamda_files('CO')
    >>> isinstance(out, list)
    True
    >>> os.path.basename(out[0])
    'co.dat'
    >>> os.path.basename(lamda_files('HCO+')[0])
    'hco+@xpol.dat'
    >>> os.path.basename(lamda_files('pH2CO')[0])
    'ph2co-h2.dat'
    """
    regexp = re.compile(fr"\{os.path.sep}{re.escape(species.lower())}(?:@\w*)?(?:[-_]h2)?[^a-z1-9+\{os.path.sep}]*\.dat$")
    matching_files = [file for file in LAMDA_FILES if re.search(regexp, file)]
    return matching_files

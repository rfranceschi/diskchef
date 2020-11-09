"""Functions to get the file from LAMDA database"""
from glob import glob
import os
import re
from typing import List

DUST_OPACITY_FILES = glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.dat"))
"""Unsorted list of paths to all dust file. Use `file` to find a file for the species"""


def file(dust_name: str) -> List[str]:
    """
    Returns a list of absolute paths to matched dust opacity file

    Args:
        species: species name
    """
    regexp = re.compile(fr"\{os.path.sep}{dust_name.lower()}(?:@\w*)?[^a-z1-9+\{os.path.sep}]*\.dat$")
    matching_files = [file for file in DUST_OPACITY_FILES if re.search(regexp, file)]
    return matching_files

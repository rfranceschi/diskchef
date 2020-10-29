"""Functions to get the file from LAMDA database"""
from glob import glob
import os
import re
from typing import List

LAMDA_FILES = glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.dat"))
"""Unsorted list of paths to all LAMDA database file. Use `get_file` to find a file for the species"""


def get_file(species: str) -> List[str]:
    """
    Returns a list of absolute paths to matched LAMDA database file

    Args:
        species: species name
    """
    regexp = re.compile(fr"\{os.path.sep}{species.lower()}(?:@\w*)?[^a-z1-9+\{os.path.sep}]*\.dat$")
    matching_files = [file for file in LAMDA_FILES if re.search(regexp, file)]
    return matching_files

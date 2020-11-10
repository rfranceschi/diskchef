import os
from glob import glob

import pytest

import diskchef.lamda


def test_lamda_is_found():
    files = glob(os.path.join(diskchef.lamda.__path__[0], "files", "*.dat"))
    assert "13co.dat" in [os.path.basename(file) for file in files]


@pytest.mark.parametrize(
    "species, expected",
    [
        ["13co", 1],
        ["13CO", 1],
        ["so2", 2],
        ["OH", 2]
    ]
)
def test_get_lamda(species, expected):
    assert len(diskchef.lamda.file(species)) == expected

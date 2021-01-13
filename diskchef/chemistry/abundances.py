from collections import UserDict

from typing import List


class Abundances(UserDict):
    """
    Dict-like class to store the abundances

    Usage:

    >>> Abundances.from_3abunds_as_lines(
    ... '''# initial abundances, n(X)/n(H), 'frc(1:nfrc)
    ... 15                ! "Low" metals
    ... G0         1.000D-00
    ... HD         1.550D-05
    ... oH2        3.750D-01
    ... pH2        1.250D-01
    ... HE         0.975D-01
    ... C          0.786D-04'''.splitlines()
    ... ) ==  {'G0': 1.0, 'HD': 1.55e-05, 'oH2': 0.375, 'pH2': 0.125, 'HE': 0.0975, 'C': 7.86e-05}
    True
    """

    @classmethod
    def from_3abunds(cls, file: str):
        """Read 3abunds.inp ALCHEMIC abundances file"""
        with open(file) as fff:
            lines = fff.readlines()
        return cls.from_3abunds_as_lines(lines)

    @classmethod
    def from_3abunds_as_lines(cls, lines: List):
        """Process the string containing the 3abunds.inp ALCHEMIC abundances file"""
        species = {}
        for line in lines[2:]:
            _species, _abundance_to_H = line.split()
            species[_species] = float(_abundance_to_H.replace("D", "E"))
        return cls(species)

    def __init__(self, *args, **kwargs):
        if not (bool(args) or bool(kwargs)):
            super().__init__({"H2": 0.5, "CO": 0.5e-4})
        else:
            super().__init__(*args, **kwargs)

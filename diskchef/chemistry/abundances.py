from typing import List
from collections import UserDict


class Abundances(UserDict):
    @classmethod
    def from_3abunds(cls, file: str):
        with open(file) as fff:
            lines = fff.readlines()
        return cls.from_3abunds_as_lines(lines)

    @classmethod
    def from_3abunds_as_lines(cls, lines: List):
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

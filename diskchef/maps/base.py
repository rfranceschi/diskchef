from dataclasses import dataclass
from typing import List

from astropy import units as u

from diskchef.chemistry.base import ChemistryBase


@dataclass
class Line:
    """
    todo: add post_init
    """
    name: str
    transition: str
    molecule: str
    collision_partner: tuple = ('H2',)


@dataclass
class MapBase:
    """The base class for map generation"""
    chemistry: ChemistryBase = None
    line_list: List[Line] = None

    @property
    def table(self):
        return self.chemistry.table
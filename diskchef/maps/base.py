from dataclasses import dataclass
from typing import List
import logging

from diskchef.chemistry.base import ChemistryBase


@dataclass
class Line:
    """
    todo: add post_init
    """
    name: str
    transition: int
    molecule: str
    collision_partner: tuple = ('H2',)


@dataclass
class MapBase:
    """The base class for map generation"""
    chemistry: ChemistryBase = None
    line_list: List[Line] = None

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

    @property
    def table(self):
        return self.chemistry.table
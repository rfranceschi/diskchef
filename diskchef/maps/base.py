from dataclasses import dataclass
from typing import List
import logging

from astropy import units as u

from diskchef.chemistry.base import ChemistryBase
from diskchef.engine.other import PathLike
from diskchef.engine.ctable import CTable
import diskchef.lamda
from diskchef.lamda.line import Line


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

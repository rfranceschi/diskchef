from dataclasses import dataclass

import logging
from typing import List

from diskchef.chemistry.base import ChemistryBase
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

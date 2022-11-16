import logging
from dataclasses import dataclass
from typing import List, Union

from diskchef.chemistry.base import ChemistryBase
from diskchef.lamda.line import Line
from diskchef.physics.base import PhysicsBase


@dataclass
class MapBase:
    """The base class for map generation"""
    chemistry: Union[ChemistryBase, PhysicsBase] = None
    line_list: List[Line] = None

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

    @property
    def table(self):
        return self.chemistry.table

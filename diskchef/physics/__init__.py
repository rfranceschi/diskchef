"""
Package with classes which calculate the physics structure

Important:

`diskchef.physics.PhysicsModel` -- base class for physics, can't be used directly

`diskchef.physics.WilliamsBest2014` -- class that defines physical model as in Williams & Best 2014
"""

from diskchef.physics.base import PhysicsModel
from diskchef.physics.williams_best import WilliamsBest2014

"""
Package with classes which calculate the chemical structure based on the physical structure

Important:

`diskchef.chemistry.ChemistryBase` -- base class for chemistry, with fixed abundances

`diskchef.chemistry.ChemistryWB2014` -- realization of chemistry as described in Williams & Best, 2014

`diskchef.chemistry.NonzeroChemistryWB2014` -- same as `ChemistryWB2014` with more realistic non-zero CO abundances outside
of the molecular layer set by default

`diskchef.chemistry.Abundances` -- dict-like class to store the abundances and read them from files
"""
from diskchef.chemistry.williams_best_co import ChemistryWB2014, NonzeroChemistryWB2014
from diskchef.chemistry.base import ChemistryBase
from diskchef.chemistry.abundances import Abundances

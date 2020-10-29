"""Package with LAMDA database files and functions to process them

https://home.strw.leidenuniv.nl/~moldata/
Schöier, F.L., van der Tak, F.F.S., van Dishoeck E.F., Black, J.H. 2005, A&A 432, 369-379

Important:

`get_file` -- function to get absolute path to a lamda file by species name
"""
from diskchef.lamda.get_file import get_file, LAMDA_FILES
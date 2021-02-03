"""Package with interface to run RadMC

https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html

RADMC-3D was developed from 2007 to 2010/2011 at the Max Planck Institute for Astronomy in Heidelberg, funded by a Max Planck Research Group grant from the Max Planck Society. As of 2011 the development continues at the Institute for Theoretical Astrophysics (ITA) of the Zentrum für Astronomy (ZAH) at the University of Heidelberg.

The use of this software is free of charge. However, it is not allowed to distribute this package without prior consent of the lead author (C.P. Dullemond). Please refer any interested user to the web site of this software where the package is available, which is currently:
http://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d

or the github repository:

https://github.com/dullemond/radmc3d-2.0

The github repository will always have the latest version, but it may not be always the most stable version (though usually it is).

The main author of RADMC-3D is Cornelis P. Dullemond. However, the main author of the radmc3dPy Python package is Attila Juhasz.

Important:

`diskchef.lamda.lamda_files` -- function to get absolute path to LAMDA files by species name
"""
from diskchef.maps import radmcrt
from diskchef.maps.radmcrt import RadMCRTSingleCall, RadMCRT, RadMCVisualize

import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from diskchef.maps.radiation_fields import draine1978

wavelength = np.geomspace(10*u.nm, 1*u.um)
isrf = draine1978(wavelength)

plt.loglog(wavelength, isrf, label="Draine 1978")
plt.legend()
plt.show()

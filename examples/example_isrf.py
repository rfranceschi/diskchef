import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u

from astropy.modeling.physical_models import BlackBody
from diskchef.maps.radiation_fields import draine1978, weingartner_draine_2001

wavelength = np.geomspace(80 * u.nm, 5 * u.mm, 1000)
isrf_draine = draine1978(wavelength)
isrf_mmp = weingartner_draine_2001(wavelength)
cmb = BlackBody(2.7 * u.K)(wavelength)
fifteenk = BlackBody(1.5e4 * u.K, scale=1e-14)(wavelength)

isrf = (
        BlackBody(7500 * u.K, 1e-14)
        + BlackBody(4000 * u.K, 1.65e-13)
        + BlackBody(3000 * u.K, 4e-13)
)(wavelength)

plt.loglog(wavelength, isrf_mmp, label="Weingartner 2001")
plt.loglog(wavelength, isrf_draine, label="Draine 1978")
plt.loglog(wavelength, cmb, label="CMB 2.7K")
plt.loglog(wavelength, fifteenk, label="15000K, 1e-14")
plt.loglog(wavelength, BlackBody(7500 * u.K, 1e-14)(wavelength), label="7500K")
plt.loglog(wavelength, BlackBody(4000 * u.K, 1.65e-13)(wavelength), label="4000K")
plt.loglog(wavelength, BlackBody(3000 * u.K, 4e-13)(wavelength), label="3000K")
plt.axvspan(*([13.6, 6] * u.eV).to(wavelength.unit, equivalencies=u.spectral()), color="k", alpha=0.2)
plt.axvspan(*([0.0912, 0.2] * u.um).to(wavelength.unit, equivalencies=u.spectral()), color="r", alpha=0.2)

plt.ylim([1e-21, 1e-14])
plt.legend()
plt.savefig("example_isrf.pdf")

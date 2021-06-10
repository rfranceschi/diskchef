from pathlib import Path

import astropy.table
from spectral_cube import SpectralCube
from astropy import units as u
import astropy.wcs
import numpy as np

transitions = ['HCN J=3-2', 'HCO+ J=3-2', 'N2H+ J=3-2', 'CO J=2-1']
folder = Path("Default") / "radmc_gas"
fluxes = []
for transition in transitions:
    file = folder / f"{transition}_image.fits"
    scube = SpectralCube.read(file)
    average_brightness = scube.to(u.Jy / u.arcsec ** 2).mean()

    scube = scube.with_spectral_unit(u.km / u.s, velocity_convention='radio')

    pixel_area_units = u.Unit(scube.wcs.celestial.world_axis_units[0]) * u.Unit(scube.wcs.celestial.world_axis_units[1])
    pixel_area = astropy.wcs.utils.proj_plane_pixel_area(scube.wcs.celestial) * pixel_area_units

    spectrum = (scube * pixel_area).to(u.mJy).sum(axis=(1, 2))  # 1d spectrum in Jy
    flux = np.trapz(scube.spectral_axis, spectrum)
    tbl = astropy.table.QTable([scube.spectral_axis, spectrum], meta={"flux": flux}, names=["Velocity", "Flux density"])
    tbl.write(folder / f"{transition}_spectrum.ecsv", overwrite=True)
    fluxes.append(flux)
fluxes_tbl = astropy.table.QTable([transitions, fluxes], names=["Transition", "Flux"], meta={"Source": "Default"})
fluxes_tbl.write(folder / "fluxes.ecsv", overwrite=True)

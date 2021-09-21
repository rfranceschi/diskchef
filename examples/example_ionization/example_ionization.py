import pathlib
import logging
from copy import deepcopy

import astropy.table
import diskchef
import diskchef.chemistry.base
import diskchef.maps.radiation_fields
import diskchef.physics.base
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u

import diskchef.dust_opacity
import diskchef.engine.plot
import diskchef.maps
import diskchef.physics.multidust
import diskchef.chemistry.scikit
from diskchef.physics.williams_best import WilliamsBest2014

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s (%(relativeCreated)10d)  %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True
bins = 100
folder = pathlib.Path("example_ionization")
folder.mkdir(parents=True, exist_ok=True)

physics_for_full_chemistry = WilliamsBest2014(radial_bins=bins, vertical_bins=bins)
physics_for_simple_chemistry = WilliamsBest2014(radial_bins=bins, vertical_bins=bins)

simple_chemistry = diskchef.chemistry.scikit.SciKitChemistry(physics_for_simple_chemistry)
simple_chemistry.run_chemistry()

physics_for_full_chemistry.xray_bruderer()
physics_for_full_chemistry.cosmic_ray_padovani18()
full_chemistry = diskchef.chemistry.scikit.SciKitChemistry(
    physics_for_full_chemistry,
    model="../diskchef/chemistry/scikit_estimators/andes2_atomic_knearest_temp_dens_uv_ioniz.pkl"
)

dust = diskchef.physics.multidust.DustPopulation(diskchef.dust_opacity.dust_files(
    "diana")[0], table=physics_for_full_chemistry.table, name="DIANA dust")
dust.write_to_table()

radmc = diskchef.maps.RadMCThermMono(
    # star_effective_temperature=2.7*u.K,
    chemistry=full_chemistry,
    folder=folder,
    scattering_mode_max=1,
    external_source_type="WeingartnerDraine2001",
    nphot_therm=int(1e7),
    nphot_mono=int(1e6),
    accretion_luminosity=0.1 * u.solLum,
)
radmc.create_files()
radmc.run(threads=8)
radmc.read_radiation_strength()
radmc.read_dust_temperature()

full_chemistry.table["Ionization rate"] = full_chemistry.table["CR ionization rate"] \
                                          + full_chemistry.table["X ray ionization rate"]
full_chemistry.table["UV radiation strength"] = full_chemistry.table["Radiation strength"]
full_chemistry.table["G_UV"] = (
        full_chemistry.table["UV radiation strength"] / diskchef.maps.radiation_fields.ANDES2_G0
).si
full_chemistry.run_chemistry()

fig, ax = plt.subplots(2, 7, sharex=True, sharey=True, figsize=(35, 10))
physics_for_full_chemistry.plot_density(axes=ax[0, 0])
tempplot = physics_for_full_chemistry.plot_temperatures(axes=ax[1, 0])
tempplot.contours("Gas temperature", [20, 40] * u.K)

diskchef.engine.plot.Plot2D(full_chemistry.table, axes=ax[0, 1], data1="CR ionization rate", data2="CR ionization rate",
                            cmap="bone")
diskchef.engine.plot.Plot2D(full_chemistry.table, axes=ax[1, 1], data1="Nucleon column density upwards",
                            data2="Nucleon column density towards star", cmap="pink_r", maxdepth=1e12)

diskchef.engine.plot.Plot2D(full_chemistry.table, axes=ax[0, 2], data1="X ray ionization rate",
                            data2="X ray ionization rate",
                            cmap="bone", maxdepth=1e12)
diskchef.engine.plot.Plot2D(full_chemistry.table, axes=ax[1, 2], data1="G_UV",
                            data2="G_UV", cmap="plasma", maxdepth=1e12)

species = ["CO", "HCO+", "CN", "C2H", "CS", "N2H+", "HCN", "H2CO"]
for spice in species:
    full_chemistry.table[f"{spice} (simple)"] = simple_chemistry.table[spice]

full_chemistry.plot_chemistry("CO (simple)", "CO", axes=ax[0, 3])
full_chemistry.plot_chemistry("HCO+ (simple)", "HCO+", axes=ax[0, 4])
full_chemistry.plot_chemistry("CN (simple)", "CN", axes=ax[0, 5])
full_chemistry.plot_chemistry("C2H (simple)", "C2H", axes=ax[0, 6])
full_chemistry.plot_chemistry("CS (simple)", "CS", axes=ax[1, 3])
full_chemistry.plot_chemistry("N2H+ (simple)", "N2H+", axes=ax[1, 4])
full_chemistry.plot_chemistry("HCN (simple)", "HCN", axes=ax[1, 5])
full_chemistry.plot_chemistry("H2CO (simple)", "H2CO", axes=ax[1, 6])

fig.savefig(folder / "report_ions.pdf")

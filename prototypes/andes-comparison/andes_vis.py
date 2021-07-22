import pathlib
import logging
from copy import deepcopy

import astropy.table
import diskchef
import diskchef.chemistry.andes
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

RUN_RADIATION = False

folder = pathlib.Path(r"D:\astrowork\chemistry_datamine\example\00038\data_output")

andes_chemistry = diskchef.chemistry.andes.ReadAndesData(folder=folder, index=4, read_uv=True, read_ionization=True)
physics_for_andes_chemistry = andes_chemistry.physics

diskchef_chemistry = diskchef.chemistry.andes.ReadAndesData(folder=folder, index=4, read_uv=True, read_ionization=True)
physics_for_diskchef_chemistry = diskchef_chemistry.physics

scikit_chemistry = diskchef.chemistry.scikit.SciKitChemistry(
    physics_for_diskchef_chemistry,
    model="../../diskchef/chemistry/scikit_estimators/andes2_atomic_knearest_temp_dens_uv_ioniz.pkl"
)
folder_simulation = folder / "radmc_4" / ("recalculate" if RUN_RADIATION else "andes")
folder_simulation.mkdir(parents=True, exist_ok=True)

if RUN_RADIATION:
    physics_for_diskchef_chemistry.xray_bruderer()
    physics_for_diskchef_chemistry.cosmic_ray_padovani18()
    dust = diskchef.physics.multidust.DustPopulation(diskchef.dust_opacity.dust_files(
        "diana")[0], table=physics_for_diskchef_chemistry.table, name="DIANA dust")
    dust.write_to_table()
    radmc = diskchef.maps.RadMCThermMono(
        chemistry=diskchef_chemistry,
        folder=folder_simulation,
        scattering_mode_max=1,
        external_source_type="WeingartnerDraine2001",
        nphot_therm=int(1e5),
        nphot_mono=int(1e5),
        accretion_luminosity=0.1 * u.solLum,
    )
    radmc.create_files()
    radmc.run(threads=8)
    radmc.read_radiation_strength()
    radmc.read_dust_temperature()

    diskchef_chemistry.table["Ionization rate"] = diskchef_chemistry.table["CR ionization rate"] \
                                                  + diskchef_chemistry.table["X ray ionization rate"]
    diskchef_chemistry.table["UV radiation strength"] = diskchef_chemistry.table["Radiation strength"]
    diskchef_chemistry.table["G_UV"] = (
            diskchef_chemistry.table["UV radiation strength"] << diskchef.maps.radiation_fields.ANDES2_G0
    ).si

    diskchef_chemistry.run_chemistry = scikit_chemistry.run_chemistry
    diskchef_chemistry.run_chemistry()

else:
    diskchef_chemistry.table["UV radiation strength"] = diskchef_chemistry.table["G_UV"]
    diskchef_chemistry.run_chemistry = scikit_chemistry.run_chemistry
    diskchef_chemistry.run_chemistry()

species = ["CO", "HCO+", "CN", "C2H", "CS", "N2H+", "HCN", "H2CO",
           "CR ionization rate", "X ray ionization rate", "G_UV"]
for spice in species:
    diskchef_chemistry.table[f"{spice} (andes2)"] = andes_chemistry.table[spice]

fig, ax = plt.subplots(2, 7, sharex=True, sharey=True, figsize=(35, 10))
physics_for_diskchef_chemistry.plot_density(axes=ax[0, 0])
tempplot = physics_for_diskchef_chemistry.plot_temperatures(axes=ax[1, 0])
tempplot.contours("Gas temperature", [20, 40] * u.K)

diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[0, 1], data1="CR ionization rate (andes2)",
                            data2="CR ionization rate",
                            cmap="bone", norm_lower=True)
try:
    diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[1, 1], data1="Nucleon column density upwards",
                            data2="Nucleon column density towards star", cmap="pink_r", maxdepth=1e12)
except KeyError:
    pass
diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[0, 2], data1="X ray ionization rate (andes2)",
                            data2="X ray ionization rate",
                            cmap="bone", maxdepth=1e12)
diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[1, 2], data1="G_UV (andes2)",
                            data2="G_UV", cmap="plasma", maxdepth=1e12)

diskchef_chemistry.plot_chemistry("CO (andes2)", "CO", axes=ax[0, 3])
diskchef_chemistry.plot_chemistry("HCO+ (andes2)", "HCO+", axes=ax[0, 4])
diskchef_chemistry.plot_chemistry("CN (andes2)", "CN", axes=ax[0, 5])
diskchef_chemistry.plot_chemistry("C2H (andes2)", "C2H", axes=ax[0, 6])
diskchef_chemistry.plot_chemistry("CS (andes2)", "CS", axes=ax[1, 3])
diskchef_chemistry.plot_chemistry("N2H+ (andes2)", "N2H+", axes=ax[1, 4])
diskchef_chemistry.plot_chemistry("HCN (andes2)", "HCN", axes=ax[1, 5])
diskchef_chemistry.plot_chemistry("H2CO (andes2)", "H2CO", axes=ax[1, 6])

fig.savefig(folder_simulation / f"report_relative.pdf")

fig, ax = plt.subplots(2, 7, sharex=True, sharey=True, figsize=(35, 10))
physics_for_diskchef_chemistry.plot_density(axes=ax[0, 0])
tempplot = physics_for_diskchef_chemistry.plot_temperatures(axes=ax[1, 0])
tempplot.contours("Gas temperature", [20, 40] * u.K)

diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[0, 1], data1="CR ionization rate (andes2)",
                            data2="CR ionization rate",
                            cmap="bone", norm_lower=True)
try:
    diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[1, 1], data1="Nucleon column density upwards",
                            data2="Nucleon column density towards star", cmap="pink_r", maxdepth=1e12)
except KeyError:
    pass
diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[0, 2], data1="X ray ionization rate (andes2)",
                            data2="X ray ionization rate",
                            cmap="bone", maxdepth=1e12)
diskchef.engine.plot.Plot2D(diskchef_chemistry.table, axes=ax[1, 2], data1="G_UV (andes2)",
                            data2="G_UV", cmap="plasma", maxdepth=1e12)

diskchef_chemistry.plot_absolute_chemistry("CO (andes2)", "CO", axes=ax[0, 3])
diskchef_chemistry.plot_absolute_chemistry("HCO+ (andes2)", "HCO+", axes=ax[0, 4])
diskchef_chemistry.plot_absolute_chemistry("CN (andes2)", "CN", axes=ax[0, 5])
diskchef_chemistry.plot_absolute_chemistry("C2H (andes2)", "C2H", axes=ax[0, 6])
diskchef_chemistry.plot_absolute_chemistry("CS (andes2)", "CS", axes=ax[1, 3])
diskchef_chemistry.plot_absolute_chemistry("N2H+ (andes2)", "N2H+", axes=ax[1, 4])
diskchef_chemistry.plot_absolute_chemistry("HCN (andes2)", "HCN", axes=ax[1, 5])
diskchef_chemistry.plot_absolute_chemistry("H2CO (andes2)", "H2CO", axes=ax[1, 6])

fig.savefig(folder_simulation / f"report_absolute.pdf")

fig, ax = plt.subplots(2, 5, sharex=True, figsize=(25, 10))

physics_for_diskchef_chemistry.plot_column_densities(axes=ax[0, 0])
diskchef_chemistry.plot_column_densities(species=["CO (andes2)", "CO"], axes=ax[0, 1])
diskchef_chemistry.plot_column_densities(species=["HCO+ (andes2)", "HCO+"], axes=ax[0, 2])
diskchef_chemistry.plot_column_densities(species=["CN (andes2)", "CN"], axes=ax[0, 3])
diskchef_chemistry.plot_column_densities(species=["C2H (andes2)", "C2H"], axes=ax[0, 4])
diskchef_chemistry.plot_column_densities(species=["CS (andes2)", "CS"], axes=ax[1, 1])
diskchef_chemistry.plot_column_densities(species=["N2H+ (andes2)", "N2H+"], axes=ax[1, 2])
diskchef_chemistry.plot_column_densities(species=["HCN (andes2)", "HCN"], axes=ax[1, 3])
diskchef_chemistry.plot_column_densities(species=["H2CO (andes2)", "H2CO"], axes=ax[1, 4])

fig.savefig(folder_simulation / f"report_coldens.pdf")

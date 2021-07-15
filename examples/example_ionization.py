import pathlib

import diskchef.dust_opacity
import diskchef.engine.plot
import diskchef.maps
import diskchef.physics.multidust
from matplotlib import pyplot as plt
import logging

from astropy import units as u

import diskchef.chemistry.scikit
from diskchef.physics.williams_best import WilliamsBest2014

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
bins = 100
folder = pathlib.Path("example_ionization")
folder.mkdir(parents=True, exist_ok=True)

physics = WilliamsBest2014(radial_bins=bins, vertical_bins=bins)
physics.xray_bruderer()
physics.cosmic_ray_padovani18()

chem = diskchef.chemistry.scikit.SciKitChemistry(physics)
chem.run_chemistry()

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
physics.plot_density(axes=ax[0, 0])
tempplot = physics.plot_temperatures(axes=ax[1, 0])
tempplot.contours("Gas temperature", [20, 40] * u.K)

diskchef.engine.plot.Plot2D(chem.table, axes=ax[0, 1], data1="CR ionization rate", data2="CR ionization rate",
                            cmap="bone")
diskchef.engine.plot.Plot2D(chem.table, axes=ax[1, 1], data1="Nucleon column density upwards",
                            data2="Nucleon column density upwards", cmap="pink_r")

diskchef.engine.plot.Plot2D(chem.table, axes=ax[0, 2], data1="X ray ionization rate", data2="X ray ionization rate",
                            cmap="bone")
diskchef.engine.plot.Plot2D(chem.table, axes=ax[1, 2], data1="Nucleon column density towards star",
                            data2="Nucleon column density towards star", cmap="pink_r")

fig.savefig(folder / "report_ions.pdf")

dust = diskchef.physics.multidust.DustPopulation(diskchef.dust_opacity.dust_files(
    "diana")[0], table=physics.table, name="DIANA dust")
dust.write_to_table()

radmc = diskchef.maps.RadMCTherm(
    chemistry=chem,
    folder=folder
)
radmc.create_files()
radmc.run(threads=8)

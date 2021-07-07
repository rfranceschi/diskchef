import pathlib

from matplotlib import pyplot as plt
import logging

from astropy import units as u

import diskchef.engine.plot
from diskchef.chemistry.williams_best_co import NonzeroChemistryWB2014
from diskchef.physics.williams_best import WilliamsBest2014

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s   %(name)-60s %(levelname)-8s %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
)
logger = logging.getLogger(__name__)
bins = 100
folder = pathlib.Path("example_wb_chemistry")
folder.mkdir(parents=True, exist_ok=True)

physics = WilliamsBest2014(radial_bins=bins, vertical_bins=bins)
chem = NonzeroChemistryWB2014(physics)
chem.run_chemistry()

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
physics.plot_density(axes=ax[0, 0])
tempplot = physics.plot_temperatures(axes=ax[1, 0])
tempplot.contours("Gas temperature", [20] * u.K, colors="white")

coplot = chem.plot_chemistry("CO", "CO", axes=ax[0, 1])
coldensplot = chem.plot_column_densities_2d(axes=ax[1, 1])
coldensplot.contours(data="H2 column density towards star", levels=[1.3e21] / u.cm ** 2, location="upper",
                     colors="orange")
coldensplot.contours(data="H2 column density upwards", levels=[1.3e21] / u.cm ** 2, location="bottom", colors="red")

coplot.contours(data="H2 column density towards star", levels=[1.3e21] / u.cm ** 2, location="upper", colors="orange")
coplot.contours(data="H2 column density upwards", levels=[1.3e21] / u.cm ** 2, location="bottom", colors="red")
coplot.contours("Gas temperature", [20] * u.K, colors="white")

fig.savefig(folder / "report.pdf")

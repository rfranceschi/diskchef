import logging
import sys
from pathlib import Path
from astropy import units as u

from diskchef import Line
from diskchef.model.model import Model
from diskchef.physics import WilliamsBest2014
from diskchef.uv.uvfits_to_visibilities_ascii import UVFits
from diskchef.physics.yorke_bodenheimer import YorkeBodenheimer2008

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
logging.info(sys.version)

folder = Path(".")

# Reading data  TODO upload and make download link -- also create uvfits with gildas
# data = folder / "s-Line-18-13CO_1+D.uvfits"
# uvdata = UVFits(data, sum=False)

lines = [
    Line(name='13CO J=2-1', transition=2, molecule='13CO'),
]

mass = 0.52 * u.M_sun
WilliamsBest2014()
demo_model = Model(
    disk="DN Tau",
    line_list=lines,
    params=dict(
        r_min=1 * u.au, r_max=300 * u.au, radial_bins=300, vertical_bins=300,
        star_mass=mass
    ),
    rstar=1.92 * u.R_sun,
    tstar=3806 * u.K,
    inc=35.18 * u.deg,
    PA=79.19 * u.deg,
    distance=128 * u.pc,
    npix=300,
    channels=151,
)


"""
IM Lup

IM Lup HCN 3-2 3.233 Jy km / s
IM Lup HCO+ 3-2 10.311499999999999 Jy km / s
IM Lup N2H+ 3-2 1.6725000000000003 Jy km / s
IM Lup CO 2-1 23.375999999999998 Jy km / s
IM Lup CN22 22-11 0.35950000000000015 Jy km / s
IM Lup CN23 23-12 4.286500000000001 Jy km / s
IM Lup H2CO32 3-2 0.7219999999999995 Jy km / s
IM Lup H2CO43 4-3 2.7285000000000004 Jy km / s
"""

import pathlib
from astropy import units as u
from diskchef.lamda.line import Line
import logging
import sys
from dataclasses import dataclass
import astropy.units as u
from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.dust_opacity import dust_files
from diskchef.lamda.line import Line
from diskchef.maps import RadMCTherm, RadMCRTSingleCall
from diskchef.physics.multidust import DustPopulation
from diskchef.physics.williams_best import WilliamsBest2014
from divan import Divan
from functools import cached_property
from matplotlib import colors
from pathlib import Path
from diskchef.model.model import Model

lines = [
    Line(name='HCN J=3-2', transition=3, molecule='HCN'),
    Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
    Line(name='N2H+ J=3-2', transition=3, molecule='N2H+'),
    Line(name='CO J=2-1', transition=2, molecule='CO'),
    # Line(name='CN 2_2-1_1', transition=???, molecule='CN'),
    # Line(name='CN 2_3-1_2', transition=???, molecule='CN'),
    # Line(name='H2CO J=3-2', transition=3, molecule='H2CO'),
    # Line(name='H2CO J=4-3', transition=3, molecule='H2CO'),
]

fluxes = [
             3.233,
             10.31,
             1.672,
             23.37
         ] * (u.Jy * u.km / u.s)

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
logging.info(sys.version)



mass = 1 * u.M_sun
from diskchef.physics.yorke_bodenheimer import YorkeBodenheimer2008

yb = YorkeBodenheimer2008()

demo_model = Model(
    disk="Default",
    line_list=lines,
    params=dict(r_min=1 * u.au, r_max=300 * u.au, radial_bins=100, vertical_bins=100),
    rstar=yb.radius(mass),
    tstar=yb.effective_temperature(mass),
    inc=30 * u.deg,
    PA=25 * u.deg,
    distance=150 * u.pc,
    npix=120,
    channels=21,
)

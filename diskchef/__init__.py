"""DiskCheF package

For support, contact author via smirnov@mpia.de, or create an issue on a GitLab page
"""
__author__ = "Grigorii Smirnov-Pinchukov, PRODIGE team"

import logging

from diskchef.engine.ctable import CTable
from diskchef.lamda.line import Line
from diskchef import chemistry, physics, lamda, engine, dust_opacity, maps, model, fitting

from diskchef.physics import WilliamsBest100au, WilliamsBest2014
from diskchef.chemistry import NonzeroChemistryWB2014, SciKitChemistry
from diskchef.lamda.line import Line
from diskchef.engine.ctable import CTable
from diskchef.maps import RadMCRTSingleCall, RadMCTherm
from diskchef.uv import UVFits
from diskchef.fitting import UltraNestFitter, Parameter


def logging_basic_config(
        format='%(asctime)s (%(relativeCreated)10d ms) PID %(process)10d  %(name)-60s %(levelname)-8s %(message)s',
        datefmt='%m.%d.%Y %H:%M:%S',
        level=logging.WARNING,
        **kwargs
):
    """Sets default logging configuratgion"""
    logging.basicConfig(format=format, datefmt=datefmt, level=level, **kwargs)

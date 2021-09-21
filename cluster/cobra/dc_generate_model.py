import sys
from pathlib import Path
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import astropy.units as u

# NB: ultranest requires h5py
import ultranest

import logging

from diskchef.lamda.line import Line
from diskchef.physics.yorke_bodenheimer import YorkeBodenheimer2008
from diskchef.model.model import Model
from diskchef.uv.uvfits_to_visibilities_ascii import UVFits
from diskchef.fitting.fitters import UltraNestFitter as Fitter, Parameter
print("Import done")

mass = 0.7 * u.M_sun
yb = YorkeBodenheimer2008()

lines = [
    # Line(name='CO J=2-1', transition=2, molecule='CO'),
    Line(name='13CO J=2-1', transition=2, molecule='13CO'),
    # Line(name='C18O J=2-1', transition=2, molecule='C18O'),
]


def generate_ref_model():
    demo_model = Model(
        disk="Reference",
        line_list=lines,
        params=dict(
            r_min=1 * u.au, r_max=300 * u.au, radial_bins=100, vertical_bins=100,
            tapering_radius=100*u.au, gas_mass=1e-3*u.Msun, tapering_gamma=0.75,
            temperature_slope=0.55, midplane_temperature_1au=200*u.K, atmosphere_temperature_1au=1e3*u.K
            ),
        rstar=yb.radius(mass),
        tstar=yb.effective_temperature(mass),
        inc=30 * u.deg,
        PA=25 * u.deg,
        distance=150 * u.pc,
        npix=100,
        channels=21,
        run_mctherm=False
    )

if __name__ == "__main__":
    generate_ref_model()

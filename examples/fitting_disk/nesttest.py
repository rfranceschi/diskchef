#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# In[2]:

#logging.basicConfig(format='%(asctime)s (%(relativeCreated)10d)| %(levelname)s : %(message)s',
#                         level=logging.INFO, stream=sys.stdout)
# logging.info(sys.version)


# In[3]:

GENERATE_MODEL = False
mass = 0.7 * u.M_sun

yb = YorkeBodenheimer2008()

lines = [  # Line(name='CO J=2-1', transition=2, molecule='CO'),
    Line(name='13CO J=2-1', transition=2, molecule='13CO')]


def generate_ref_model():
    if not GENERATE_MODEL:
        return
    demo_model = Model(
        disk="Reference",
        line_list=lines,
        params=dict(
            r_min=1 * u.au, r_max=300 * u.au, radial_bins=100, vertical_bins=100,
            tapering_radius=100*u.au, gas_mass=1e-3*u.Msun, tapering_gamma=0.75,
            temperature_slope=0.55
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

print("Reference model")
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Fitter of rank {rank} is online")
    if rank == 0:
        generate_ref_model()
except ImportError:
    print("Could not get to MPI!")
    generate_ref_model()
print("... done")


if GENERATE_MODEL:
    exit("Reference model is generated, exiting...")
# In[5]:

refmodel_13CO = 'Reference/radmc_gas/13CO J=2-1_image.fits'
refmodel_12CO = 'Reference/radmc_gas/CO J=2-1_image.fits'
uvcoverage = UVFits("demo.pkl")
print("Model ready")

# In[6]:


uvcoverage.image_to_visibilities(refmodel_13CO)

chisq = uvcoverage.chi2_with(refmodel_13CO)
print('Chi^2 of the model with itself: ', chisq)
uvcoverage.plot_uvgrid()

# In[7]:


parameters = ['tapering_radius', 'gas_mass']


# In[8]:


def prior_transform(quantile_cube):
    params = quantile_cube.copy()

    # Assume it's a nice, flat prior, between 30 and 200 au:
    lower = 30.
    upper = 200.

    params[0] = quantile_cube[0] * (upper - lower) + lower

    # Assume it's a log-flat prior, between 1e-4 and 1e-2 M_sun:
    params[1] = 10 ** (quantile_cube[1] * 2 - 4)


    # Assume it's a nice, flat prior:
    lower = 0.5
    upper = 1.0

    params[2] = quantile_cube[2] * (upper - lower) + lower

    # Assume it's a nice, flat prior:
    lower = 0.4
    upper = 0.7

    params[3] = quantile_cube[3] * (upper - lower) + lower

    return params


def my_likelihood(params):
    tapering_radius = params[0]
    gas_mass = params[1]
    tapering_gamma = params[2]
    temperature_slope = params[3]
    try:
        with tempfile.TemporaryDirectory(prefix='fit_', dir='fit_tmp') as temp_dir:
            folder = Path(temp_dir)
            disk_model = Model(disk="Fit",
                           line_list=lines,
                           params=dict(r_min=1 * u.au, r_max=300 * u.au,
                                radial_bins=100, vertical_bins=100,
                                tapering_radius=tapering_radius * u.au, gas_mass=gas_mass * u.Msun,
                                tapering_gamma=tapering_gamma, temperature_slope=temperature_slope
                                ),
                           rstar=yb.radius(mass),
                           tstar=yb.effective_temperature(mass),
                           inc=30 * u.deg,
                           PA=25 * u.deg,
                           distance=150 * u.pc,
                           npix=100,
                           channels=21,
                           run_mctherm=False,
                           folder=folder, run=False)
            disk_model.chemistry()
            disk_model.image()
            chi2 = uvcoverage.chi2_with(folder / "radmc_gas" / "13CO J=2-1_image.fits")
    except Exception as e:
        print(e)
        return -np.inf
    return -0.5 * chi2


# In[9]:


from diskchef.fitting.fitters import UltraNestFitter as Fitter, Parameter

tapering_radius = Parameter(name="tapering_radius", min=30, max=200, truth=100)
gas_mass = Parameter(name="gas_mass", min=1e-4, max=1e-2, truth=1e-3)
tapering_gamma = Parameter(name="tapering_gamma", min=0.5, max=1, truth=0.75)
temperature_slope = Parameter(name="temperature_slope", min=0.4, max=0.7, truth=0.55)
print("Parameters initialized")

fitter = Fitter(
    my_likelihood, [tapering_radius, gas_mass, tapering_gamma, temperature_slope],
    progress=True,
    transform=prior_transform,
    storage_backend='hdf5',
    resume=True
)

print("Fitter initialized")
res = fitter.fit()
print("MPI? ", fitter.sampler.use_mpi)

if fitter.sampler.use_mpi:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        fig = fitter.corner()
        fig.savefig("corner.pdf")
else:
    fig = fitter.corner()
    fig.savefig("corner.pdf")
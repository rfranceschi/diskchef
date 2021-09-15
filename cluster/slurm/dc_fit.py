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

print("Reference model")
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"Fitter of rank {rank} is online")
except ImportError:
    print("Could not get to MPI!")
print("... done")

refmodel_13CO = 'Reference/radmc_gas/13CO J=2-1_image.fits'
#refmodel_C18O = 'Reference/radmc_gas/C18O J=2-1_image.fits'

uv13co = UVFits("noema_c_uv.pkl")
#uvc18o = UVFits("noema_c_uv.pkl")
uv13co.image_to_visibilities(refmodel_13CO)
#uvc18o.image_to_visibilities(refmodel_C18O)


def my_likelihood(params):
    tapering_radius = params[0]
    log_gas_mass = params[1]
    temperature_slope = params[2]
    atmosphere_temperature_1au = params[3]
    midplane_temperature_1au = params[4]
    try:
        with tempfile.TemporaryDirectory(prefix='fit_', dir='.') as temp_dir:
            folder = Path(temp_dir)
            disk_model = Model(disk="Fit",
                           line_list=lines,
                           params=dict(r_min=1 * u.au, r_max=300 * u.au,
                                radial_bins=100, vertical_bins=100,
                                tapering_radius=tapering_radius * u.au, gas_mass=10**log_gas_mass * u.Msun,
                                temperature_slope=temperature_slope,
                                midplane_temperature_1au=midplane_temperature_1au*u.K, 
                                atmosphere_temperature_1au=atmosphere_temperature_1au*u.K
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
            chi2 = (
                uv13co.chi2_with(folder / "radmc_gas" / "13CO J=2-1_image.fits") 
            #    + uvc18o.chi2_with(folder / "radmc_gas" / "C18O J=2-1_image.fits")
            )
    except Exception as e:
        print(e)
        return -np.inf
    return -0.5 * chi2

parameters = [
    Parameter(name="Tapering radius, au", min=30, max=200, truth=100),
    Parameter(name="log_{10}(M_{gas}/M_\odot)", min=-4, max=-2, truth=-3),
    Parameter(name="Temperature slope", min=0.4, max=0.7, truth=0.55),
    Parameter(name="T_{atm}, K", min=200, max=2000, truth=1000),
    Parameter(name="T_{mid}, K", min=50, max=400, truth=200),
]  
print("Parameters initialized")

fitter = UltraNestFitter(
    my_likelihood, parameters,
    progress=True,
    storage_backend='hdf5',
    resume=True,
    run_kwargs={'min_num_live_points': 10},
)

print("Fitter initialized")
res = fitter.fit()
print("MPI? ", fitter.sampler.use_mpi)

if fitter.sampler.use_mpi:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        fitter.table.write("table.ecsv")
        fig = fitter.corner()
        fig.savefig("corner.pdf")
else:
    fig = fitter.corner()
    fig.savefig("corner.pdf")


import time

from .model import ProposalModel
from pathlib import Path
import astropy.table
from diskchef.lamda.line import Line
from diskchef.physics.yorke_bodenheimer import YorkeBodenheimer2008
from astropy import units as u
import numpy as np
import emcee

data_folder = Path("Default")

data = astropy.table.QTable.read(data_folder)

lines = [
    Line(name='HCN J=3-2', transition=3, molecule='HCN'),
    Line(name='HCO+ J=3-2', transition=3, molecule='HCO+'),
    Line(name='N2H+ J=3-2', transition=3, molecule='N2H+'),
    Line(name='CO J=2-1', transition=2, molecule='CO'),
]

yb = YorkeBodenheimer2008()

call_index = 0


def lnprob(params):
    mass = params[0] * u.M_sun
    global call_index
    demo_model = ProposalModel(
        disk="Fit",
        line_list=lines,
        params=dict(r_min=1 * u.au, r_max=300 * u.au, radial_bins=40, vertical_bins=40),
        rstar=yb.radius(mass),
        tstar=yb.effective_temperature(mass),
        inc=params[1] * u.deg,
        PA=params[2] * u.deg,
        distance=params[3] * u.pc,
        npix=80,
        channels=51,
        folder=Path(f"{call_index}")
    )
    call_index += 1


nwalkers = 16
ndims = 4
nsteps = 20
trues = np.array([1, 30, 25, 150])

initials = np.random.uniform(0.5, 2, [nwalkers, ndims]) * trues

sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob, )
start = time.time()
sampler.run_mcmc(initials, nsteps, progress=True)
end = time.time()
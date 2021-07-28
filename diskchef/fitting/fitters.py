"""Brut-force fitter for diskchef"""
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Union, Literal
from itertools import product, cycle, zip_longest

import numpy as np
import ultranest
from astropy import units as u
from astropy.table import QTable
import corner
from matplotlib import pyplot as plt

from emcee import EnsembleSampler
from matplotlib.colors import LogNorm

from diskchef.engine import CHEFNotImplementedError
from diskchef.engine.overplot_scatter import overplot_scatter


@dataclass
class Parameter:
    name: str
    min: Union[u.Quantity, float] = None
    max: Union[u.Quantity, float] = None


@dataclass
class Fitter:
    """Base class for finding parameters for lnprob which minimize its output"""
    lnprob: Callable
    parameters: List[Parameter]
    threads: int = None
    progress: bool = False

    def __post_init__(self):
        self._table = None

    @property
    def table(self):
        return self._table

    def fit(
            self,
            *args, **kwargs
    ):
        raise CHEFNotImplementedError

    def corner(self, scatter_kwargs=None, **kwargs):
        if scatter_kwargs is None:
            scatter_kwargs = {}
        data = np.array([self.table[param.name] for param in self.parameters]).T
        fig = corner.corner(data, **kwargs)
        overplot_scatter(fig, data, c=-self.table["lnprob"], norm=LogNorm(), **scatter_kwargs)
        return fig


@dataclass
class BruteForceFitter(Fitter):
    n_points: Union[int, List[int]] = 10

    def fit(
            self,
            *args, **kwargs
    ):
        pars = []
        if not hasattr(self.n_points, "__len__"):
            self.n_points = [self.n_points] * len(self.parameters)

        for parameter, length in zip_longest(self.parameters, self.n_points):
            par_range = np.linspace(parameter.min, parameter.max, length)
            pars.append(par_range)

        all_parameter_combinations = product(*pars)
        tbl = QTable(np.array([comb for comb in all_parameter_combinations]),
                     names=[par.name for par in self.parameters])
        if self.threads != 1:
            lnprob = partial(self.lnprob, *args, **kwargs)
            with Pool(self.threads) as pool:
                tbl["lnprob"] = pool.map(lnprob, tbl)
        else:
            tbl["lnprob"] = [self.lnprob(parameters, *args, **kwargs) for parameters in tbl]
        self._table = tbl
        return tbl[np.argmax(tbl["lnprob"])]


@dataclass
class EMCEEFitter(Fitter):
    nwalkers: int = 100
    nsteps: int = 100

    def fit(
            self,
            *args, **kwargs
    ):
        pos0 = (np.random.random((self.nwalkers, len(self.parameters)))  # 0--1
                * np.array([param.max - param.min for param in self.parameters])  # * range
                + np.array([param.min for param in self.parameters]))  # + min

        if self.threads != 1:
            pool = Pool(self.threads)
        else:
            pool = None

        sampler = EnsembleSampler(self.nwalkers, len(self.parameters), self.lnprob, args=args, kwargs=kwargs, pool=pool)
        sampler.run_mcmc(pos0, self.nsteps, progress=self.progress)
        if pool is not None:
            pool.close()
        tbl = QTable(sampler.flatchain, names=[par.name for par in self.parameters])
        tbl["lnprob"] = sampler.flatlnprobability
        self._table = tbl
        return tbl[np.argmax(tbl["lnprob"])]


@dataclass
class UltraNestFitter(Fitter):
    nwalkers: int = 100
    nsteps: int = 100
    transform: Callable = None
    resume: Literal['resume', 'overwrite', 'subfolder'] = 'overwrite'

    def fit(
            self,
            *args, **kwargs
    ):
        if self.threads != 1:
            pool = Pool(self.threads)
        else:
            pool = None

        lnprob = partial(self.lnprob, *args, **kwargs)
        lnprob.__name__ = self.lnprob.__name__
        sampler = ultranest.ReactiveNestedSampler(
            [param.name for param in self.parameters],
            lnprob,
            self.transform,
            log_dir="myanalysis",
            resume=True,
        )
        result = sampler.run()
        if pool is not None:
            pool.close()
        sampler.plot_run()
        sampler.plot_trace()
        sampler.plot_corner()


        # tbl = QTable(sampler.flatchain, names=[par.name for par in self.parameters])
        # tbl["lnprob"] = sampler.flatlnprobability
        # self._table = tbl
        # return tbl[np.argmax(tbl["lnprob"])]
        return {
            parname: parvalue
            for parname, parvalue in
            zip(sampler.paramnames, sampler.results['posterior']['mean'])
        }

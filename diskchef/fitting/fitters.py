"""Fitter for diskchef"""
import inspect
import pickle
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool

from matplotlib.figure import Figure
from pathlib import Path
from typing import Callable, List, Union, Literal, Dict
from itertools import product, cycle, zip_longest

import numpy as np
import ultranest
from astropy import units as u
from astropy.table import QTable
import corner
from matplotlib import pyplot as plt
import scipy.optimize
from emcee import EnsembleSampler
from matplotlib.colors import LogNorm

import diskchef.engine.other
from diskchef.engine.exceptions import CHEFValueError, CHEFNotImplementedError
from diskchef.engine.overplot_scatter import overplot_scatter, overplot_hexbin


@dataclass
class Parameter:
    """
    Class that handles parameters for diskchef Fitters

    Args:
        name: str - name of the parameter
        min: float
        max: float - prior minimal and maximal value of the parameter
        truth: optional, float - expected value of the parameter
        format_: str - python format-string for the parameter output
        log: bool - whether the logarithm of the valus should be used for fitting instead

    Fields:
        fitted: float - fitted value
        fitted_error: float - 1-sigma error of the fitted value
        fitted_error_up: float - upper error of the fitted value
        fitted_error_down: float - lower error of the fitted value
    """
    name: str
    min: Union[u.Quantity, float] = None
    max: Union[u.Quantity, float] = None
    truth: float = None
    format_: str = "{:.2f}"
    log: bool = False

    def __post_init__(self):
        self.fitted = None
        self.fitted_error = None
        self.fitted_error_up = None
        self.fitted_error_down = None

    @property
    def math_repr(self) -> str:
        """
        Returns matplotlib/LaTeX-formatted representation of the parameter and its fitted value
        """
        out = "$"
        if self.fitted is None:
            out += self.name
        elif self.fitted_error is None:
            out += f"{self.name} = {self.format_.format(self.fitted)}"
        elif (self.fitted_error_down is None) or (self.fitted_error_up is None):
            out += f"{self.name} = {self.format_.format(self.fitted)} ± {self.format_.format(self.fitted_error)}"
        else:
            out += f"{self.name} = {self.format_.format(self.fitted)}" \
                   f"^{{+{self.format_.format(self.fitted_error_up)}}}" \
                   f"_{{-{self.format_.format(self.fitted_error_down)}}}"
        if self.truth is not None:
            out += f" ({self.format_.format(self.truth)})"
        out += "$"
        return out

    def __str__(self):
        return f"${self.name}$"

    def __eq__(self, other):
        """
        Checks whether right parameter is within left parameter's error bar
        """
        if self.fitted is None:
            return False
        elif self.fitted_error is None:
            return self.fitted == other
        elif (self.fitted_error_up is None) or (self.fitted_error_down is None):
            return self.fitted - self.fitted_error <= other <= self.fitted + self.fitted_error
        else:
            return self.fitted - self.fitted_error_down <= other <= self.fitted + self.fitted_error_up


@dataclass
class Fitter:
    """Base class for finding parameters for lnprob which maximizes its output"""
    lnprob: Callable
    parameters: List[Parameter]
    threads: int = None
    progress: bool = False
    hexbin: bool = True
    fitter_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)
        self._table = None
        self.sampler = None

        self._check_lnprob()

        for parameter in self.parameters:
            if any(
                    fit_result is not None for fit_result in
                    [parameter.fitted, parameter.fitted_error,
                     parameter.fitted_error_up, parameter.fitted_error_down]
            ):
                self.logger.info("Parameter %s is already fit! Clearing the fitted values.", parameter)
                parameter.fitted = parameter.fitted_error = None
                parameter.fitted_error_up = parameter.fitted_error_down = None
                self.logger.info("Parameter %s is cleaned.", parameter)

    def _check_lnprob(self):
        if not callable(self.lnprob):
            raise CHEFValueError("lnprob must be callable!")
        defaults = [
            param.default
            for name, param
            in inspect.signature(self.lnprob).parameters.items()
            if name != 'self'
        ]
        if defaults[0] is not inspect.Parameter.empty:
            self.logger.warning("lnprob first argument has a default value!")
            if not isinstance(defaults[0], Iterable):
                self.logger.error("First argument of lnprob should be an array of parameters! Continuing anyway.")
        if inspect.Parameter.empty in defaults[1:]:
            self.logger.error("lnprob should have only one non-default argument! Continuing anyway.")

    def _post_fit(self):
        """
        Private method to run after the fitting is complete
        """
        fig = self.corner()
        fig.savefig("corner.pdf")
        self.save()

    def lnprob_fixed(self, *args, **kwargs):
        """Decorates self.lnprob so that minimum and maximum from self.parameters are considered"""
        for parameter, arg in zip(self.parameters, *args):
            if (parameter.min is not None) and (parameter.min > arg):
                return -np.inf
            if (parameter.max is not None) and (parameter.max < arg):
                return -np.inf

        try:
            return self.lnprob(*args, **kwargs)
        except Exception as e:
            self.logger.error("lnprob crushed during the function call with %s %s",
                              args, kwargs)
            self.logger.error("%s", e)
            return -np.inf

    @property
    def table(self):
        return self._table

    def fit(
            self,
            *args, **kwargs
    ) -> Dict[str, Parameter]:
        raise CHEFNotImplementedError

    def corner(self, scatter_kwargs=None, hexbin_kwargs=None, **kwargs) -> Figure:
        if scatter_kwargs is None:
            scatter_kwargs = {}
        if hexbin_kwargs is None:
            hexbin_kwargs = {}
        if "cmap" not in hexbin_kwargs.items():
            hexbin_kwargs["cmap"] = "afmhot"
        data = np.array([self.table[param.name] for param in self.parameters]).T
        labels = [str(param) for param in self.parameters]
        truths = [parameter.truth for parameter in self.parameters]

        if "weight" in self.table.colnames:
            weights = self.table["weight"]
            cumsumweights = np.cumsum(weights)
            mask = cumsumweights > 1e-4
        else:
            weights = np.ones_like(data[:, 0])
            mask = Ellipsis
        try:
            fig = corner.corner(
                data[mask], weights=weights[mask], labels=labels, show_titles=False, truths=truths, **kwargs
            )
            mins = []
            maxs = []
            medians = []
            for param in self.parameters:
                if param.fitted_error_down is not None:
                    min_ = param.fitted - param.fitted_error_down
                elif param.fitted_error is not None:
                    min_ = param.fitted - param.fitted_error
                else:
                    min_ = None
                if param.fitted_error_up is not None:
                    max_ = param.fitted + param.fitted_error_up
                elif param.fitted_error is not None:
                    max_ = param.fitted + param.fitted_error
                else:
                    max_ = None
                if param.fitted is not None:
                    median_ = param.fitted
                else:
                    median_ = None
                mins.append(min_)
                maxs.append(max_)
                medians.append(median_)
            corner.overplot_lines(fig, mins, color='k', alpha=0.2, linestyle=(0, (5, 5)))
            corner.overplot_lines(fig, maxs, color='k', alpha=0.2, linestyle=(0, (5, 5)))
            corner.overplot_lines(fig, medians, color='k', alpha=1, linestyle=(0, (5, 5)))

            self._decorate_corner(fig)
        except AssertionError:
            return Figure()
        # overplot_scatter(fig, data, c=-self.table["lnprob"], norm=LogNorm(), **scatter_kwargs)
        if self.hexbin:
            overplot_hexbin(
                fig, data[mask], C=-self.table["lnprob"][mask],
                norm=diskchef.engine.other.LogNormMaxOrders(maxdepth=1e4), reduce_C_function=np.nanmin,
                **hexbin_kwargs
            )
        return fig

    def _decorate_corner(self, fig: Figure):
        ndim = len(self.parameters)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for ax, parameter in zip(axes.diagonal(), self.parameters):
            ax.set_title(parameter.math_repr, size="small")

    @property
    def parameters_dict(self) -> Dict[str, Parameter]:
        return {parameter.name: parameter for parameter in self.parameters}

    def save(self, filename: Path = "fitter.sav"):
        """Saves the fitter to a file"""
        try:
            with open(filename, "wb") as fff:
                pickle.dump(self, fff)
            self.logger.info("Fitter successfully pickled to %s", filename)
        except Exception as e:
            self.logger.error("Could not save fitter! %s", e)

    @classmethod
    def load(cls, filename: Path = "fitter.sav") -> "Fitter":
        """Return the loaded fitter from a file"""
        try:
            with open(filename, "rb") as fff:
                fitter = pickle.load(fff)
            fitter.logger.info("Fitter was successfully unpickled from %s", filename)
            return fitter
        except Exception as e:
            raise e


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
            lnprob = partial(self.lnprob_fixed, *args, **kwargs)
            with Pool(self.threads) as pool:
                tbl["lnprob"] = pool.map(lnprob, tbl)
        else:
            tbl["lnprob"] = [self.lnprob_fixed(parameters, *args, **kwargs) for parameters in tbl]
        tbl["weight"] = np.exp(tbl["lnprob"] - tbl["lnprob"].max())
        self._table = tbl
        argmax_row = tbl[np.argmax(tbl["lnprob"])]
        for parameter, n_points in zip(self.parameters, self.n_points):
            parameter.fitted = argmax_row[parameter.name]
            err = (parameter.max - parameter.min) / (n_points - 1)
            parameter.fitted_error_up = err
            parameter.fitted_error_down = err
            parameter.fitted_error = err

        self._post_fit()
        return self.parameters_dict


@dataclass
class EMCEEFitter(Fitter):
    nwalkers: int = 100
    nsteps: int = 100
    burn_steps: int = 30
    burn_strategy: Literal[None, 'best'] = None

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

        sampler = EnsembleSampler(self.nwalkers, len(self.parameters), self.lnprob_fixed, args=args, kwargs=kwargs,
                                  pool=pool, **self.fitter_kwargs)
        sampler.run_mcmc(pos0, self.burn_steps, progress=self.progress)
        if self.burn_strategy is None:
            pos1 = None
        elif self.burn_strategy == 'best':
            pos1 = np.tile(sampler.flatchain[np.argmax(sampler.flatlnprobability)], [self.nwalkers, 1])
        else:
            raise CHEFValueError("burn_strategy should be None or 'best'")
        sampler.run_mcmc(pos1, self.nsteps - self.burn_steps, progress=self.progress)
        if pool is not None:
            pool.close()
        self.sampler = sampler
        tbl = QTable(sampler.flatchain[self.burn_steps * self.nwalkers:], names=[par.name for par in self.parameters])
        tbl["lnprob"] = sampler.flatlnprobability[self.burn_steps * self.nwalkers:]
        self._table = tbl

        for i, parameter in enumerate(self.parameters):
            results = np.percentile(sampler.flatchain[self.burn_steps * self.nwalkers:, i], [16, 50, 84])
            parameter.fitted = results[1]
            parameter.fitted_error_up = results[2] - results[1]
            parameter.fitted_error_down = results[1] - results[0]
            parameter.fitted_error = (parameter.fitted_error_up + parameter.fitted_error_down) / 2
        self._post_fit()
        return self.parameters_dict


@dataclass
class UltraNestFitter(Fitter):
    nwalkers: int = 100
    nsteps: int = 100
    transform: Callable = None
    resume: Literal[True, 'resume', 'resume-similar', 'overwrite', 'subfolder'] = 'overwrite'
    log_dir: Union[str, Path] = "ultranest"
    run_kwargs: dict = field(default_factory=dict)
    plot_corner: bool = True

    storage_backend: Literal['hdf5', 'csv', 'tsv'] = 'hdf5'

    DEFAULT_FOR_RUN_KWARGS = dict(
        Lepsilon=0.01,  # Increase when lnprob is inaccurate
        frac_remain=0.05,  # Decrease if lnprob is expected to have peaks
        min_num_live_points=100,
        dlogz=1.,
        dKL=1.,
    )
    INFINITY = 1e50

    def __post_init__(self):
        super().__post_init__()
        if self.transform is None:
            self.transform = self.rescale

        self.sampler = None
        for key, value in self.DEFAULT_FOR_RUN_KWARGS.items():
            if key not in self.run_kwargs:
                self.run_kwargs[key] = value

    def rescale(self, cube):
        params = np.empty_like(cube)
        for i, parameter in enumerate(self.parameters):
            if parameter.log:
                params[i] = 10 ** (cube[i] * (np.log10(parameter.max) - np.log10(parameter.min))
                                   + np.log10(parameter.min))
            else:
                params[i] = cube[i] * (parameter.max - parameter.min) + parameter.min
        return params

    def lnprob_fixed(self, *args, **kwargs):
        return np.nan_to_num(
            super().lnprob_fixed(*args, **kwargs),
            neginf=-self.INFINITY, posinf=self.INFINITY, nan=-self.INFINITY
        )

    def save(self, filename: Path = "fitter.sav"):
        """Saves the fitter to a file. Does not save ultranest.ReactiveNestedSampler object!"""
        sampler = self.sampler
        del self.sampler
        try:
            with open(filename, "wb") as fff:
                pickle.dump(self, fff)
            self.logger.info("Fitter successfully pickled to %s. Sampler was not picked!", filename)
        except Exception as e:
            self.logger.error("Could not save fitter! %s", e)
        finally:
            self.sampler = sampler

    def fit(
            self,
            *args, **kwargs
    ):
        self.log_dir = Path(self.log_dir)

        lnprob = partial(self.lnprob_fixed, *args, **kwargs)
        lnprob.__name__ = self.lnprob.__name__
        self.sampler = ultranest.ReactiveNestedSampler(
            [str(param) for param in self.parameters],
            lnprob,
            self.transform,
            log_dir=self.log_dir,
            resume=self.resume,
            storage_backend=self.storage_backend,
            **self.fitter_kwargs
        )

        for i, result in enumerate(self.sampler.run_iter(**self.run_kwargs)):
            if not self.sampler.use_mpi or self.sampler.mpi_rank == 0:
                self.logger.info("Step %d: %s", i, result)
                tbl = QTable(self.sampler.results['weighted_samples']['points'],
                             names=[par.name for par in self.parameters])
                tbl["lnprob"] = self.sampler.results['weighted_samples']['logl']
                tbl["lnprob"][tbl["lnprob"] <= -self.INFINITY] = -np.inf
                tbl["weight"] = self.sampler.results['weighted_samples']['weights']
                self._table = tbl

                results = self.sampler.results['posterior']
                for iparam, parameter in enumerate(self.parameters):
                    parameter.fitted = results['mean'][iparam]
                    parameter.fitted_error_up = results['errup'][iparam] - results['mean'][iparam]
                    parameter.fitted_error_down = results['mean'][iparam] - results['errlo'][iparam]
                    parameter.fitted_error = results['stdev'][iparam]

                try:
                    self.sampler.plot()
                except Exception as e:
                    self.logger.error("Could not plot! %s", e)

                if self.plot_corner:
                    try:
                        fig = self.corner()
                        fig.savefig(self.log_dir / f"corner_{i:06d}.pdf")
                        fig.savefig(self.log_dir / "corner.pdf")
                    except ValueError as e:
                        self.logger.error("Could not make corner plot for %d:", i)
                        self.logger.error(e)
                self.save(self.log_dir / f"fitter_{i:06d}.sav")

        if not self.sampler.use_mpi or self.sampler.mpi_rank == 0:
            self._post_fit()

        return self.parameters_dict


@dataclass
class SciPyFitter(Fitter):
    method: Union[None, str] = None

    def fit(self, rel_step: float = None, *args, **kwargs):
        x0 = np.asarray([(param.min + param.max)/2 for param in self.parameters])
        scipy_result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            # lambda params: -self.lnprob_fixed(params, *args, **kwargs),
            lambda params: -self.lnprob_fixed(params, *args, **kwargs),
            x0=x0, method='CG', jac='2-point',
            options={
                'finite_diff_rel_step': rel_step,
            }
        )
        self.scipy_result = scipy_result

        for parameter, result in zip(self.parameters, self.scipy_result.x):
            parameter.fitted = result

        if rel_step is not None:
            try:
                for i, _parameter in enumerate(self.parameters):
                    _parameter.fitted_error = np.absolute(rel_step * max(1, _parameter.fitted) * self.scipy_result.jac[i])
            except AttributeError:
                pass
        return self.parameters_dict, self.scipy_result

    def jac(self, params, dp: float = 1e-3, *args, **kwargs):
        all_params = [params]
        for i, _params in enumerate(params):
            dparams = np.copy(params)
            dparams[i] += dp
            all_params.append(dparams)
        with Pool(self.threads) as pool:
            results = pool.map(lambda param: -self.lnprob_fixed(param, *args, *kwargs), all_params)
        jac = (results[1:] - results[0]) / dp
        return jac

    def corner(self, scatter_kwargs=None, hexbin_kwargs=None, **kwargs) -> Figure:
        pass


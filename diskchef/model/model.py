import pickle

from functools import cached_property
import pathlib
from typing import List, Type, Dict, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import astropy.wcs
import astropy.table
from astropy import units as u
import spectral_cube
from spectral_cube.utils import SpectralCubeWarning
from matplotlib import pyplot as plt

import diskchef.physics.base
from diskchef.chemistry import NonzeroChemistryWB2014
from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.dust_opacity import dust_files
from diskchef import Line
from diskchef.maps import RadMCTherm, RadMCRTLines
from diskchef.physics import YorkeBodenheimer2008
from diskchef.physics.multidust import DustPopulation
from diskchef.physics.williams_best import WilliamsBest2014, WilliamsBest100au
from diskchef.engine.other import PathLike

from diskchef.uv import UVFits

warnings.filterwarnings(action='ignore', category=SpectralCubeWarning,
                        append=True)


@dataclass
class ModelFit:
    disk: str
    directory: Union[pathlib.Path, str] = None
    line_list: List[Line] = None
    physics_class: Type[diskchef.physics.base.PhysicsModel] = WilliamsBest100au
    physics_params: dict = field(default_factory=dict)
    chemistry_class: Type[diskchef.chemistry.base.ChemistryBase] = NonzeroChemistryWB2014
    chemical_params: dict = field(default_factory=dict)
    mstar: u.Msun = 1 * u.Msun
    rstar: u.au = None
    tstar: u.K = None
    inclination: u.deg = 0 * u.deg
    position_angle: u.deg = 0 * u.deg
    distance: u.pc = 100 * u.pc
    nphot_therm: int = 1e7
    npix: int = 100
    channels: int = 31
    dust_opacity_file: Union[pathlib.Path, str] = dust_files("diana")[0]
    radial_bins_rt: int = None
    vertical_bins_rt: int = None
    line_window_width: u.km / u.s = 15 * u.km / u.s
    radmc_lines_run_kwargs: dict = field(default_factory=dict)
    mctherm_threads = 4

    def __post_init__(self):
        self.dust = None
        self.disk_physical_model = None
        self.disk_chemical_model = None
        self.chi2_dict = None
        self._check_radius_temperature()
        self._update_defaults()
        self._initialize_working_dir()
        self.initialize_physics()
        self.initialize_dust()
        self.initialize_chemistry()

    def run_chemistry(self):
        self.disk_chemical_model.run_chemistry()
        self.add_co_isotopologs()

    def run_line_radiative_transfer(
            self,
            run_kwargs: dict = None,
            **kwargs
    ):
        """Run line radiative transfer"""
        folder_rt_gas = self.gas_directory
        folder_rt_gas.mkdir(parents=True, exist_ok=True)

        disk_map = RadMCRTLines(
            chemistry=self.disk_chemical_model, line_list=self.line_list,
            radii_bins=self.radial_bins_rt, theta_bins=self.vertical_bins_rt,
            folder=folder_rt_gas, **kwargs
        )

        disk_map.create_files(channels_per_line=self.channels, window_width=self.line_window_width)

        disk_map.run(
            inclination=self.inclination,
            position_angle=self.position_angle,
            distance=self.distance,
            npix=self.npix,
            **self.radmc_lines_run_kwargs
        )

    def add_co_isotopologs(self, _13c: float = 77, _18o: float = 560):
        self.disk_chemical_model.table['13CO'] = self.disk_chemical_model.table['CO'] / _13c
        self.disk_chemical_model.table['C18O'] = self.disk_chemical_model.table['CO'] / _18o
        self.disk_chemical_model.table['13C18O'] = self.disk_chemical_model.table['CO'] / (_13c * _18o)
        # remember to fix the next abundance
        # self.disk_chemical_model.table['C17O'] = self.disk_chemical_model.table['CO'] / (
        #         560 * 5)  # http://articles.adsabs.harvard.edu/pdf/1994ARA%26A..32..191W

    def initialize_chemistry(self):
        self.disk_chemical_model = self.chemistry_class(physics=self.disk_physical_model, **self.chemical_params)

    def initialize_dust(self):
        self.dust = DustPopulation(self.dust_opacity_file,
                                   table=self.disk_physical_model.table,
                                   name="Dust")
        self.dust.write_to_table()

    def initialize_physics(self):
        self.disk_physical_model = self.physics_class(**self.physics_params)

    def _update_defaults(self):
        if self.radial_bins_rt is None:
            self.radial_bins_rt = self.physics_params.get("radial_bins", 100)
        if self.vertical_bins_rt is None:
            self.radial_bins_rt = self.physics_params.get("vertical_bins", 100)

    def _initialize_working_dir(self):
        if self.directory is None:
            self.directory = Path(self.disk)
        else:
            self.directory = pathlib.Path(self.directory)
        self.directory.mkdir(exist_ok=True, parents=True)
        with open(self.directory / "model_description.txt", "w") as fff:
            fff.write(repr(self))
            fff.write("\n")
            fff.write(str(self.__dict__))

    def _check_radius_temperature(self):
        if self.rstar is None and self.tstar is None:
            yb = YorkeBodenheimer2008()
            self.rstar = yb.radius(self.mstar)
            self.tstar = yb.effective_temperature(self.mstar)

    def mctherm(self, threads=None):
        """Run thermal radiative transfer for dust temperature calculation"""
        if threads is None:
            threads = self.mctherm_threads
        folder_rt_dust = self.dust_directory
        folder_rt_dust.mkdir(parents=True, exist_ok=True)

        self.mctherm_model = RadMCTherm(
            chemistry=self.disk_chemical_model,
            folder=folder_rt_dust,
            nphot_therm=self.nphot_therm,
            star_radius=self.rstar,
            star_effective_temperature=self.tstar
        )

        self.mctherm_model.create_files()
        self.mctherm_model.run(threads=threads)
        self.mctherm_model.read_dust_temperature()

    @property
    def gas_directory(self):
        return self.directory / "radmc_gas"

    @property
    def dust_directory(self):
        return self.directory / "radmc_dust"

    def chi2(self, uvs: Dict[str, UVFits]):
        """

        Args:
            uvs: dictionary in a form of {line.name: uvt for line is self.line_list}

        Returns:
            sum of chi2 between uvs and lines
        """

        self.chi2_dict = {
            name: uv.chi2_with(self.gas_directory / f"{name}_image.fits")
            for name, uv in uvs.items()
            if name in [line.name for line in self.line_list]
        }
        return sum(self.chi2_dict.values())


@dataclass
class Model:
    """Class to run a simulation"""
    disk: str
    rstar: u.au
    tstar: u.K
    params: dict
    inc: u.deg
    PA: u.deg
    distance: u.pc
    nphot_therm: int = 1e7
    run: bool = True
    line_list: List[Line] = None
    npix: int = 100
    channels: int = 31
    run_mctherm: bool = True
    radial_bins_rt: int = None
    vertical_bins_rt: int = None
    folder: pathlib.Path = None
    physics_class: Type[diskchef.physics.base.PhysicsModel] = WilliamsBest2014
    scikit_model: PathLike = None

    def __post_init__(self):
        warnings.warn("Model class is deprecated in favor of ModelFit, which is being developed!", DeprecationWarning)
        if self.folder is None:
            self.folder = pathlib.Path(self.disk)
        else:
            self.folder = pathlib.Path(self.folder)
        self.folder.mkdir(exist_ok=True, parents=True)
        with open(self.folder / "model_description.txt", "w") as fff:
            fff.write(repr(self))
            fff.write("\n")
            fff.write(str(self.__dict__))

        if self.radial_bins_rt is None:
            self.radial_bins_rt = self.params.get("radial_bins", 100)
        if self.vertical_bins_rt is None:
            self.radial_bins_rt = self.params.get("vertical_bins", 100)

        self.disk_physical_model = self.physics_class(**self.params)
        dust = DustPopulation(dust_files("diana")[0],
                              table=self.disk_physical_model.table,
                              name="DIANA dust")
        dust.write_to_table()
        self.disk_chemical_model = SciKitChemistry(self.disk_physical_model, model=self.scikit_model)
        if self.run:
            self.run_simulation()

    def run_simulation(self):
        """Run all the steps of the simulation"""
        if self.run_mctherm: self.mctherm()
        self.chemistry()
        self.plot()
        self.image()
        self.photometry()

    def mctherm(self):
        """Run thermal radiative transfer for dust temperature calculation"""
        folder_rt_dust = self.folder / "radmc_dust"
        # folder_rt_dust.mkdir(parents=True, exist_ok=True)

        self.mctherm_model = RadMCTherm(
            chemistry=self.disk_chemical_model,
            folder=folder_rt_dust,
            nphot_therm=self.nphot_therm,
            star_radius=self.rstar,
            star_effective_temperature=self.tstar
        )

        self.mctherm_model.create_files()
        self.mctherm_model.run(threads=4)
        self.mctherm_model.read_dust_temperature()

    def image(
            self,
            radii_bins: int = 100, theta_bins: int = 100,
            run_kwargs: dict = None, window_width: u.km / u.s = 15 * u.km / u.s,
            **kwargs
    ):
        """Run line radiative transfer"""
        folder_rt_gas = self.folder / "radmc_gas"
        # folder_rt_gas.mkdir(parents=True, exist_ok=True)

        disk_map = RadMCRTLines(
            chemistry=self.disk_chemical_model, line_list=self.line_list,
            radii_bins=radii_bins, theta_bins=theta_bins,
            folder=folder_rt_gas, **kwargs
        )

        disk_map.create_files(channels_per_line=self.channels, window_width=window_width)

        if run_kwargs is None: run_kwargs = {}
        disk_map.run(inclination=self.inc,
                     position_angle=self.PA,
                     distance=self.distance,
                     npix=self.npix,
                     **run_kwargs
                     )

    def chemistry(self):
        """Run chemistry"""
        self.disk_chemical_model.run_chemistry()

        self.disk_chemical_model.table['13CO'] = self.disk_chemical_model.table['CO'] / 77
        self.disk_chemical_model.table['C18O'] = self.disk_chemical_model.table['CO'] / 560
        # remember to fix the next abundance
        self.disk_chemical_model.table['C17O'] = self.disk_chemical_model.table['CO'] / (
                560 * 5)  # http://articles.adsabs.harvard.edu/pdf/1994ARA%26A..32..191W
        self.disk_chemical_model.table['13C18O'] = self.disk_chemical_model.table['CO'] / (77 * 560)

    def plot(self, save=None, **kwargs):
        """Plot physical and chemical structure"""
        fig, ax = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(11, 10))

        self.disk_physical_model.plot_density(axes=ax[0, 0])
        tempplot = self.disk_physical_model.plot_temperatures(axes=ax[1, 0])
        tempplot.contours("Gas temperature", [20, 40] * u.K)

        self.disk_chemical_model.plot_chemistry("CO", "CO", axes=ax[0, 1])
        coplot = self.disk_chemical_model.plot_absolute_chemistry("CO", "CO", axes=ax[1, 1], maxdepth=1e9)
        coplot.contours("Gas temperature", [20, 40] * u.K, clabel_kwargs={"fmt": "T=%d K"})
        if save:
            fig.savefig(self.folder / "structure.png")
            fig.savefig(self.folder / "structure.pdf")
        return fig

    def pickle(self):
        """Pickle the model class in a file -- not working"""
        with open(self.folder / "model.pkl", "wb") as pkl:
            pickle.dump(self, pkl)

    @classmethod
    def from_picke(cls, path: pathlib.Path):
        """Return the model class from the pickle file  -- experimental"""
        with open(path, "rb") as pkl:
            return pickle.load(pkl)

    @cached_property
    def output_fluxes(self) -> pathlib.Path:
        """Path to output `fluxes.ecsv` table"""
        return self.folder / "fluxes.ecsv"

    def photometry(self):
        fluxes = []
        transitions = [line.name for line in self.line_list]
        for transition in transitions:
            file = self.folder / "radmc_gas" / f"{transition}_image.fits"
            scube = spectral_cube.SpectralCube.read(file)
            scube = scube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
            pixel_area_units = u.Unit(scube.wcs.celestial.world_axis_units[0]) * u.Unit(
                scube.wcs.celestial.world_axis_units[1])
            pixel_area = astropy.wcs.utils.proj_plane_pixel_area(scube.wcs.celestial) * pixel_area_units

            spectrum = (scube * pixel_area).to(u.mJy).sum(axis=(1, 2))  # 1d spectrum in Jy
            flux = np.abs(np.trapz(spectrum, scube.spectral_axis))
            tbl = astropy.table.QTable(
                [scube.spectral_axis, u.Quantity(spectrum)],
                meta={"flux": flux},
                names=["Velocity", "Flux density"],
            )
            tbl.write(self.folder / f"{transition}_spectrum.ecsv", overwrite=True)
            fluxes.append(flux)
        fluxes_tbl = astropy.table.QTable(
            [transitions, fluxes], names=["Transition", "Flux"], meta={"Source": "Default"}
        )
        fluxes_tbl.write(self.output_fluxes, overwrite=True)
        self.fluxes_tbl = fluxes_tbl

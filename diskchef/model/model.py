from functools import cached_property
import pathlib
from typing import List, Type
from dataclasses import dataclass, field

import numpy as np
import astropy.wcs
import astropy.table
from astropy import units as u
import spectral_cube

import diskchef.physics.base
from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.dust_opacity import dust_files
from diskchef import Line
from diskchef.maps import RadMCTherm, RadMCRTSingleCall
from diskchef.physics.multidust import DustPopulation
from diskchef.physics.williams_best import WilliamsBest2014
from diskchef.engine.other import PathLike


@dataclass
class BaseModel:
    """Base class for `diskchef` model

    Needs to be subclussed to run properly

    Fields:
      name: `str` -- name of the model
      folder: `PathLike` -- root directory for running the model. Same as `self.name` if not specified
      run: bool -- whether to run automatically after initialization
      physics_class: `PhysicsBase` subclass -- class for physics
      params_physics: dict -- parameters for `self.physics_class`
      chemistry_class: `ChemistryBase` subclass -- class for chemistry
      params_chemistry: dict -- parameters for `self.chemistry_class`
      line_rt_class: `RadMCRT` subclass -- class for line radiative transfer
      params_line_rt: dict -- parameters for `self.line_rt_class`

      Usage:
      >>> model = BaseModel(
      ...   name="DQ Tau", folder="DQ Tau model", run=True,
      ...   physics_class=WilliamsBest2014, params_physics={"star_mass": 1*u.solMass},
      ... )
    """
    name: str = "default"
    folder: PathLike = None
    run: bool = False
    physics_class: Type[diskchef.physics.base.PhysicsBase] = None
    params_physics: dict = field(default_factory=dict)
    chemistry_class: Type[diskchef.chemistry.base.ChemistryBase] = None
    params_chemistry: dict = field(default_factory=dict)
    line_rt_class: Type[diskchef.maps.RadMCRT] = None
    params_line_rt: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.folder is None:
            self.folder = pathlib.Path(self.name)
        else:
            self.folder = pathlib.Path(self.folder)

        if self.run:
            self.run_simulation()

    @cached_property
    def physics(self):
        """Run physical model"""
        return self.physics_class(**self.params_physics)

    @cached_property
    def dust(self):
        """Add dust to physical model"""
        dust = DustPopulation(dust_files("diana")[0], table=self.physics.table, name="DIANA dust")
        dust.write_to_table()
        return dust

    @cached_property
    def chemistry(self):
        """Run chemistry"""
        return self.chemistry_class(physics=self.physics, **self.params_chemistry)

    def run_simulation(self):
        pass


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

    def __post_init__(self):
        if self.folder is None:
            self.folder = pathlib.Path(self.disk)
        else:
            self.folder = pathlib.Path(self.folder)

        if self.radial_bins_rt is None:
            self.radial_bins_rt = self.params.get("radial_bins", 100)
        if self.vertical_bins_rt is None:
            self.radial_bins_rt = self.params.get("vertical_bins", 100)

        disk_physical_model = WilliamsBest2014(**self.params)
        dust = DustPopulation(dust_files("diana")[0],
                              table=disk_physical_model.table,
                              name="DIANA dust")
        dust.write_to_table()
        self.disk_chemical_model = SciKitChemistry(disk_physical_model)
        if self.run:
            self.run_simulation()

    def run_simulation(self):
        """Run all the steps of the simulation"""
        if self.run_mctherm: self.mctherm()
        self.chemistry()
        # self.plot()
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
        self.mctherm_model.run(threads=1)
        self.mctherm_model.read_dust_temperature()

    def image(
            self,
            radii_bins=100, theta_bins=100,
    ):
        """Run line radiative transfer"""
        folder_rt_gas = self.folder / "radmc_gas"
        # folder_rt_gas.mkdir(parents=True, exist_ok=True)

        disk_map = RadMCRTSingleCall(
            chemistry=self.disk_chemical_model, line_list=self.line_list,
            radii_bins=radii_bins, theta_bins=theta_bins,
            folder=folder_rt_gas,
        )

        disk_map.create_files(channels_per_line=self.channels, window_width=15 * (u.km / u.s))

        disk_map.run(inclination=self.inc,
                     position_angle=self.PA,
                     distance=self.distance,
                     npix=self.npix,
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

    def plot(self, **kwargs):
        """Plot physical and chemical structure using `divan`"""
        # dvn = Divan(matplotlib_style='divan.mplstyle')
        # dvn.physical_structure = self.disk_chemical_model.table
        # dvn.chemical_structure = self.disk_chemical_model.table
        # dvn.generate_figure_volume_densities(extra_gas_to_dust=100, **kwargs)
        # dvn.generate_figure_temperatures(
        #     **kwargs)  # gas_temperature=disk_chemical_model.table["Original Dust temperature"])
        # dvn.generate_figure(
        #     data1='Original Dust temperature',
        #     data2='RadMC Dust temperature',
        #     r=self.disk_chemical_model.table.r,
        #     z=self.disk_chemical_model.table.z,
        #     **kwargs
        # )
        # self.disk_chemical_model.physics.plot_density(**kwargs)
        # self.disk_chemical_model.plot_chemistry(**kwargs)
        # dvn.generate_figure_chemistry(spec1="CO", normalizer=colors.LogNorm(), **kwargs)
        # dvn.generate_figure_chemistry(spec1="HCO+", normalizer=colors.LogNorm(), **kwargs)
        # dvn.generate_figure_chemistry(spec1="N2H+", normalizer=colors.LogNorm(), **kwargs)
        # dvn.generate_figure_chemistry(spec1="HCN", normalizer=colors.LogNorm(), **kwargs)
        # dvn_figure = self.folder / "figs.pdf"
        # dvn.save_figures_pdf(dvn_figure)

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
            flux = np.trapz(scube.spectral_axis, spectrum)
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
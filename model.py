from dataclasses import dataclass

import astropy.units as u
from diskchef.chemistry.scikit import SciKitChemistry
from diskchef.dust_opacity import dust_files
from diskchef.lamda.line import Line
from diskchef.maps import RadMCTherm, RadMCRTLines
from diskchef.physics.multidust import DustPopulation
from diskchef.physics.williams_best import WilliamsBest2014

from functools import cached_property
from matplotlib import colors
from pathlib import Path


@dataclass
class ProposalModel:
    disk: str
    rstar: u.au
    tstar: u.K
    params: dict
    inc: u.deg
    PA: u.deg
    distance: u.pc
    nphot_therm: int = 1e7
    run: bool = True

    def __post_init__(self):
        disk_physical_model = WilliamsBest2014(**self.params)
        dust = DustPopulation(dust_files("draine03", size=1e-5 * u.cm)[0],
                              table=disk_physical_model.table,
                              name="Dust population")
        dust.write_to_table()
        self.disk_chemical_model = SciKitChemistry(disk_physical_model)

        if self.run:
            self.run_simulation()

    def run_simulation(self):
        self.mctherm()
        self.chemistry()
        self.plot()
        self.image()

    @cached_property
    def folder(self):
        folder = Path(self.disk + "_rt_model")
        folder.mkdir(exist_ok=True)

        return folder

    def mctherm(self):
        folder_rt_dust = self.folder / "radmc_dust"
        folder_rt_dust.mkdir(exist_ok=True)

        self.mctherm_model = RadMCTherm(
            chemistry=self.disk_chemical_model,
            folder=folder_rt_dust,
            nphot_therm=self.nphot_therm,
            star_radius=self.rstar,
            star_effective_temperature=self.tstar
        )

        self.mctherm_model.create_files()
        self.mctherm_model.run(threads=8)
        self.mctherm_model.read_dust_temperature()

    def image(
            self, line_list=
            (
                    Line(name='13CO_2-1', transition=1, molecule='13CO'),
                    Line(name='CO_3-2', transition=2, molecule='CO'),
                    Line(name='13CO_3-2', transition=2, molecule='13CO'),
                    Line(name='13C18O_3-2', transition=2, molecule='13C18O',
                         lamda_file='custom_lamda_files/13c18o.dat'),
                    Line(name='C17O_2-1', transition=2, molecule='C17O'),
                    Line(name='C18O_3-2', transition=2, molecule='C18O'),
            ),
            radii_bins=100, theta_bins=100,
    ):
        folder_rt_gas = self.folder / "radmc_gas"
        folder_rt_gas.mkdir(exist_ok=True)

        disk_map = RadMCRTLines(
            chemistry=self.disk_chemical_model, line_list=line_list,
            radii_bins=radii_bins, theta_bins=theta_bins,
            folder=folder_rt_gas,
        )

        disk_map.create_files(channels_per_line=300, window_width=15 * (u.km / u.s))

        disk_map.run(inclination=self.inc,
                     position_angle=self.PA,
                     distance=self.distance,
                     npix=1000,
                     )

    def chemistry(self):
        self.disk_chemical_model.run_chemistry()

        self.disk_chemical_model.table['13CO'] = self.disk_chemical_model.table['CO'] / 77
        self.disk_chemical_model.table['C18O'] = self.disk_chemical_model.table['CO'] / 560
        # remember to fix the next abundance
        self.disk_chemical_model.table['C17O'] = self.disk_chemical_model.table['CO'] / (
                560 * 5)  # http://articles.adsabs.harvard.edu/pdf/1994ARA%26A..32..191W
        self.disk_chemical_model.table['13C18O'] = self.disk_chemical_model.table['CO'] / (77 * 560)

    def plot(self):
        raise NotImplementedError("Deprecated")
        # dvn = Divan(matplotlib_style='divan.mplstyle')
        # dvn.physical_structure = self.disk_chemical_model.table
        # dvn.chemical_structure = self.disk_chemical_model.table
        # dvn.generate_figure_volume_densities(extra_gas_to_dust=100)
        # dvn.generate_figure_temperatures()  # gas_temperature=disk_chemical_model.table["Original Dust temperature"])
        # dvn.generate_figure(
        #     data1='Original Dust temperature',
        #     data2='RadMC Dust temperature',
        #     r=self.disk_chemical_model.table.r,
        #     z=self.disk_chemical_model.table.z
        # )
        # self.disk_chemical_model.physics.plot_density()
        # self.disk_chemical_model.plot_chemistry()
        # dvn.generate_figure_chemistry(spec1="HCO+", spec2="CO", normalizer=colors.LogNorm());
        # dvn_figure = self.folder / "figs.pdf"
        # dvn.save_figures_pdf(dvn_figure)

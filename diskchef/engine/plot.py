"""Module with plotting helper routines for diskchef"""
import copy

import logging
from dataclasses import dataclass

import matplotlib.ticker
import numpy as np
from typing import Literal, Union, List

from astropy import units as u
from astropy.visualization import quantity_support

quantity_support()
import matplotlib.axes
import matplotlib.scale
import matplotlib.colors
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

from diskchef import CTable
from diskchef.engine.other import LogNormMaxOrders
from diskchef.engine.exceptions import CHEFValueError

from chemical_names import from_string


@dataclass
class Plot:
    table: CTable
    axes: matplotlib.axes.Axes = None
    xscale: Union[Literal["linear", "log", "symlog", "logit"], matplotlib.scale.ScaleBase] = "log"
    yscale: Union[Literal["linear", "log", "symlog", "logit"], matplotlib.scale.ScaleBase] = "linear"
    margins: float = 0.
    unit_format: Literal["latex", "cds", None] = "latex"
    maxdepth: float = 1e6

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

        if self.axes is None:
            self.axes = plt.axes()

    def normalize_axes(self):
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)
        self.axes.margins(self.margins)

    def formatted(self, unit: u.Unit):
        if unit == u.dimensionless_unscaled:
            return "[--]"
        else:
            return fr"[{unit.to_string(self.unit_format)}]"


@dataclass
class Plot2D(Plot):
    """2D visualization of a disk"""
    data1: str = None
    data2: str = None
    x_axis: str = "Radius"
    y_axis: str = "Height to radius"
    norm: matplotlib.colors.Normalize = None
    colorbar: bool = True
    labels: bool = True
    cmap: Union[matplotlib.colors.Colormap, str] = None
    multiply_by: Union[str, float] = 1.
    levels: u.Quantity = None
    desired_max: u.Quantity = None
    norm_lower: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.norm is None:
            self.norm = LogNormMaxOrders(vmax=self.desired_max, maxdepth=self.maxdepth)

        try:
            self.multiply_by = self.table[self.multiply_by]
        except (KeyError, ValueError):
            pass

        self.table.check_zeros(self.data1)
        self.table.check_zeros(self.data2)

        data1_q = u.Quantity(self.table[self.data1] * self.multiply_by)
        data1 = data1_q.value
        if self.norm_lower:
            data2_q = u.Quantity(self.table[self.data2] * self.multiply_by)
            self.data_unit = data2_q.unit
            data2 = data2_q.value
            self.norm(data2)
        else:
            self.data_unit = data1_q.unit
            self.norm(data1)
        x_axis = self.table[self.x_axis].value
        y_axis = self.table[self.y_axis].value
        self.x_unit = self.table[self.x_axis].unit
        self.y_unit = self.table[self.y_axis].unit
        self.axes.set_xlabel(f"{self.x_axis} {self.formatted(self.x_unit)}")
        self.axes.set_ylabel(f"{self.y_axis} {self.formatted(self.y_unit)}")
        if self.levels is None:
            minlevel = np.round(np.log10(self.norm.vmin))
            maxlevel = np.round(np.log10(self.norm.vmax))
            if maxlevel - minlevel == 1:
                self.cbar_formatter = LogFormatterMathtext()
                minlevel = np.round(np.log10(self.norm.vmin), 1)
                maxlevel = np.round(np.log10(self.norm.vmax), 1)
            else:
                self.cbar_formatter = None
            self.levels = np.logspace(
                minlevel,
                maxlevel,
                13
            )
        else:
            self.levels = self.levels.to_value(self.data_unit)
        if len(set(self.levels)) == 1:
            self.levels = self.levels[0] * np.array([0.5, 1., 2.])
        self.normalize_axes()
        im = self.axes.tricontourf(
            x_axis, y_axis,
            data1,
            levels=self.levels,
            norm=self.norm,
            extend="both",
            cmap=self.cmap,
        )
        if self.data2 is not None:
            data2 = u.Quantity(self.table[self.data2] * self.multiply_by).to_value(self.data_unit)
            self.axes.tricontourf(
                x_axis, -y_axis,
                data2,
                levels=self.levels,
                norm=self.norm,
                extend="both",
                cmap=self.cmap,
            )
            self.axes.axhline(0, color="black")
            try:
                formatter = matplotlib.ticker.FuncFormatter(lambda x, pos: f'{abs(x):.2f}')  # todo as function def
                self.axes.yaxis.set_major_formatter(formatter)
            except ValueError:
                self.logger.warning("Could not fix yticks for negatives: %s", self.axes.get_yticklabels())

        self.axes.margins(self.margins)
        if self.colorbar:
            im.set_clim(self.norm.vmin, self.norm.vmax)
            self.cbar = self.axes.figure.colorbar(
                im, ax=self.axes,
                format=self.cbar_formatter
            )
            self.cbar.set_label(self.formatted(data1_q.unit), rotation="horizontal")
            self.cbar.ax.minorticks_off()
        if self.labels:
            txt = self.axes.text(
                0.05, 0.05, from_string(self.data2),
                transform=self.axes.transAxes,
                verticalalignment='bottom',
                bbox=dict(
                    boxstyle="round",
                    ec=(1., 1., 1., 0.5),
                    fc=(1., 1., 1., 0.7),
                )
            )
            if self.data2 is not None:
                txt = self.axes.text(
                    0.05, 0.95, from_string(self.data1),
                    transform=self.axes.transAxes,
                    verticalalignment='top',
                    bbox=dict(
                        boxstyle="round",
                        ec=(1., 1., 1., 0.5),
                        fc=(1., 1., 1., 0.7),
                    )
                )

    def contours(
            self,
            data: str,
            levels: Union[u.Quantity, List[float]],
            x_axis: str = "Radius",
            y_axis: str = "Height to radius",
            clabel_kwargs: dict = None,
            colors: Union[str, List[str]] = "black",
            on_colorbar: bool = True,
            location: Literal["upper", "bottom", "both"] = "both",
            **kwargs
    ):
        if clabel_kwargs is None:
            clabel_kwargs = {}
        data_q = u.Quantity(self.table[data])
        data = data_q.value
        dataunit = data_q.unit
        if "fmt" not in clabel_kwargs.keys():
            clabel_kwargs["fmt"] = f"%d {dataunit.to_string(self.unit_format)}"
        x_axis = self.table[x_axis].to_value(self.x_unit)
        y_axis = self.table[y_axis].to_value(self.y_unit)
        if location == "both":
            x_axis = [*x_axis, *x_axis]
            y_axis = [*-y_axis, *y_axis]
            data = [*data, *data]
        elif location == "bottom":
            y_axis = -y_axis
        elif location == "upper":
            pass
        else:
            raise CHEFValueError('location bust be one of ["upper", "bottom", "both"]')
        conts = self.axes.tricontour(
            x_axis, y_axis,
            data,
            levels=levels.to_value(dataunit),
            colors=colors,
            **kwargs
        )
        if on_colorbar:
            try:
                levels_as_data = levels.to_value(self.data_unit)
                new_conts = copy.copy(conts)
                new_conts.levels = levels_as_data
                self.cbar.add_lines(new_conts)
            except u.core.UnitConversionError as e:
                self.logger.info(e)
        try:
            conts.clabel(levels.to_value(dataunit), use_clabeltext=True, inline=True, inline_spacing=1, **clabel_kwargs)
        except ValueError as e:
            self.logger.warning(e)


@dataclass
class Plot1D(Plot):
    data: List[str] = None
    x_axis: u.au = None
    yscale: Union[Literal["linear", "log", "symlog", "logit"], matplotlib.scale.ScaleBase] = "log"
    labels: bool = True
    cmap: Union[matplotlib.colors.Colormap, str] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.data:
            raise ValueError("List of `data` arguments must be specified")
        if self.x_axis is None:
            if self.table.is_in_zr_regular_grid:
                self.x_axis = u.Quantity(sorted(set(self.table.r)))
            else:
                self.x_axis = np.geomspace(np.min(self.table.r), np.max(self.table), 100)
        self.x_unit = self.x_axis.unit
        self.y_unit = (self.table[self.data[0]][0] * self.x_axis[0]).cgs.unit
        self.normalize_axes()
        self.axes.set_xlabel(f"Radius {self.formatted(self.x_unit)}")
        self.axes.set_ylabel(f"{self.formatted(self.y_unit)}")

        for colname in self.data:
            data_to_plot = self.table.column_density(colname, self.x_axis).cgs
            self.axes.plot(self.x_axis, data_to_plot, label=from_string(colname))
        self.axes.legend()

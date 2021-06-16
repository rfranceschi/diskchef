"""Module with plotting helper routines for diskchef"""
import logging
from dataclasses import dataclass

import matplotlib.ticker
import numpy as np
from typing import Literal, Union

from astropy import units as u
from astropy.visualization import quantity_support

quantity_support()
import matplotlib.axes
import matplotlib.scale
import matplotlib.colors
from matplotlib import pyplot as plt

from diskchef import CTable
from diskchef.engine.other import LogNormMaxOrders

from chemical_names import from_string

@dataclass
class Plot2D:
    """2D visualization of a disk"""
    table: CTable
    data1: str = None
    data2: str = None
    x_axis: str = "Radius"
    y_axis: str = "Height to radius"
    axes: matplotlib.axes.Axes = None
    xscale: Union[Literal["linear", "log", "symlog", "logit"], matplotlib.scale.ScaleBase] = "log"
    yscale: Union[Literal["linear", "log", "symlog", "logit"], matplotlib.scale.ScaleBase] = "linear"
    margins: float = 0.
    norm: matplotlib.colors.Normalize = None
    colorbar: bool = True
    labels: bool = True
    unit_format: Literal["latex", "cds", None] = "latex"
    cmap: Union[matplotlib.colors.Colormap, str] = None
    multiply_by: Union[str, float] = 1.

    def __post_init__(self):
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__qualname__)
        self.logger.info("Creating an instance of %s", self.__class__.__qualname__)
        self.logger.debug("With parameters: %s", self.__dict__)

        if self.axes is None:
            self.axes = plt.axes()
        if self.norm is None:
            self.norm = LogNormMaxOrders()

        try:
            self.multiply_by = self.table[self.multiply_by]
        except (KeyError, ValueError):
            pass

        data1_q = u.Quantity(self.table[self.data1] * self.multiply_by)
        data1 = data1_q.value
        dataunit = data1_q.unit
        self.norm(data1)
        x_axis = self.table[self.x_axis].value
        x_unit = self.table[self.x_axis].unit
        y_axis = self.table[self.y_axis].value
        y_unit = self.table[self.y_axis].unit

        levels = np.logspace(np.round(np.log10(self.norm.vmin)), np.round(np.log10(self.norm.vmax)), 10)

        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)
        self.axes.set_xlabel(f"{self.x_axis} {self.formatted(x_unit)}")
        self.axes.set_ylabel(f"{self.y_axis} {self.formatted(y_unit)}")
        im = self.axes.tricontourf(
            x_axis, y_axis,
            data1,
            levels=levels,
            norm=self.norm,
            extend="both",
            cmap=self.cmap,
        )
        if self.data2 is not None:
            data2 = u.Quantity(self.table[self.data2] * self.multiply_by).to_value(dataunit)
            self.axes.tricontourf(
                x_axis, -y_axis,
                data2,
                levels=levels,
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
            cbar = self.axes.figure.colorbar(
                im, ax=self.axes, extend="both",
            )
            cbar.set_label(self.formatted(data1_q.unit), rotation="horizontal")
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

    def formatted(self, unit: u.Unit):
        if unit == u.dimensionless_unscaled:
            return "[--]"
        else:
            return fr"[{unit.to_string(self.unit_format)}]"
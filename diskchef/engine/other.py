import os

from typing import Union
import matplotlib.colors

PathLike = Union[str, os.PathLike]


class LogNormMaxOrders(matplotlib.colors.LogNorm):
    """`matplotlib.colors.LogNorm` subclass with maximal range"""

    def __init__(self, vmin=None, vmax=None, clip=False, maxdepth: float = 1e6):
        self.maxdepth = maxdepth
        super().__init__(vmin, vmax, clip)

    def autoscale_None(self, A):
        super().autoscale_None(A)
        self.vmin = max([self.vmin, self.vmax / self.maxdepth])

import os
from typing import Union

import matplotlib.colors
import numpy as np
import scipy.interpolate

PathLike = Union[str, os.PathLike]


class LogNormMaxOrders(matplotlib.colors.LogNorm):
    """`matplotlib.colors.LogNorm` subclass with maximal range"""

    def __init__(self, vmin=None, vmax=None, clip=False, maxdepth: float = 1e6):
        self.maxdepth = maxdepth
        super().__init__(vmin, vmax, clip)

    def autoscale_None(self, A):
        super().autoscale_None(A)
        self.vmin = max([self.vmin, self.vmax / self.maxdepth])

class unsorted_interp2d(scipy.interpolate.interp2d):
    """interp2d subclass that remembers original order of data points"""

    def __call__(self, x, y, dx=0, dy=0, assume_sorted=None):
        unsorted_idxs = np.argsort(np.argsort(x))
        return scipy.interpolate.interp2d.__call__(self, x, y, dx=dx, dy=dy)[unsorted_idxs]
from pathlib import Path
from dataclasses import dataclass
from typing import Union

import matplotlib.figure, matplotlib.axes
import matplotlib.pyplot as plt

from diskchef.uv.uvfits_to_visibilities_ascii import UVFits


@dataclass
class Matcher:
    model: Path
    data: Union[UVFits, Path]

    def __post_init__(self):
        if not isinstance(self.data, UVFits):
            self.data = UVFits(self.data, sum=False)

    def plot(self, axes: matplotlib.axes.Axes = None) -> matplotlib.figure.Figure:
        if axes is None:
            _, axes = plt.subplots(1)
        return axes.figure

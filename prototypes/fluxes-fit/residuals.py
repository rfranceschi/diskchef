import time
import diskchef.uv.galariochef
from diskchef.model.model import Model
from pathlib import Path
import astropy.table
from diskchef.lamda.line import Line
from diskchef.physics.yorke_bodenheimer import YorkeBodenheimer2008
from diskchef.uv.galariochef import Residual
from diskchef.uv.uvfits_to_visibilities_ascii import UVFits
from astropy import units as u
import numpy as np
import emcee
from spectral_cube import SpectralCube


model_reference_path = Path("Default") / "radmc_gas"
model_comparison_paths = Path("fit2").glob("fit_*/radmc_gas")

print(model_reference_path)
print(list(model_comparison_paths))

def residual(folder1: Path, folder2: Path):
    data1 = Residual(folder1)
    data2 = Residual()
    return 0

from galario.double import sampleImage

# for file in model_reference_path.glob("*_image.fits"):
#     print(file)
#     cube = SpectralCube.read(file)
#     print(cube)

# cube = SpectralCube(model_reference_path)
# vis = sampleImage(image, dxy, u, v)

uvdata = UVFits("s-Line-22-CO_1+D.uvfits", 'all', sum=False)
print(uvdata)


# DiskCheF: Disk Chemical Fitter

## Installation:

### Dependencies:

#### Python

`diskchef` depends on Python >=3.8. It is advisable to install it into its own `conda` environment.
You can download `anaconda` or `miniconda` following the [recommendations here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#).
As we are going to create a separate environment anyway, `miniconda` is enough.

Use the following to configure the environment after `conda` is installed:

```bash
conda create -yn diskchef python=3.8 anaconda
```

After the installation is complete, you can activate the environment with
```bash
conda activate diskchef
```

Or deactivate:
```bash
conda deactivate
```

#### Galario

`diskchef` requires [`galario`, a package for Fourier transfer of the interferometric data](https://mtazzari.github.io/galario/). 
The easiest way to install it is using conda:

```bash
conda install -c conda-forge galario
```

Note that this will install `galario` from `conda-forge` repository, which will make all future environment
solves significantly longer. It is not a big issue if the environment is just for `diskchef`, but might cause troubles 
if the environment is often updated. Also, make sure you use a recent version of `conda`, as it gets performance patches.

#### Cython

`ultranest` installation requires `cython` and `numpy`. 
With `anaconda` distribution proposed above, they are already included. 
Otherwise they can be installed with 
```bash
pip install numpy cython
```

#### RadMC3D

`RadMC3D` is a radiative transfer code developed by Prof. Cornelis Dullemond. 
It is required for radiative transfer within `diskchef` and should be installed manually following the [instruction](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html). 

**Ignore its installation script when it suggests adding `python` to `$PATH`, it is a very bad practice**. 
`radmc3d` executable can be put there for convenience.

By default, `diskchef` will call `radmc3d` from shell (searching in `$PATH`). You can alternatively pass 
`executable=/path/to/radmc3d` to `diskchef.maps.radmcrt.RadMCBase` subclasses, or even install it in docker, virtual machine or 
Windows Subsystem for Linux (WSL) and call with corresponding commands. From Windows, the default RadMC3D callable is `wsl radmc3d`.

### Installation

For development and usage:

Either configure SSH keys as described on https://gitlab.com/-/profile/keys 
(does not require typing passwords, preferred), and use
```bash
git clone git@gitlab.com:SmirnGreg/diskchef.git  
``` 

Otherwise use HTTPS:
```bash
git clone https://gitlab.com/SmirnGreg/diskchef.git
```

Go to the newly downloaded directory and install the package:

```bash
cd diskchef
pip install -e .
```

To update a previous `-e` installation, just go to the `diskchef` 
directory and pull the latest changes:

```bash
git pull
```


For usage only:
```
pip install git+https://gitlab.com/SmirnGreg/diskchef.git
```


## Functionality:

* Generate physical models
  * With parametric distribution of temperature and density
  * As in Williams & Best 2014
* Save multiple dust populations
* Calculate chemistry
  * With a given pre-set of abundances
  * With Williams & Best 2014 (temperature + shielding), 
  also with non-zero CO in photodissociation and regions
  * With prediction based on ANDES2 grid (by Molyarova),
  for just density-temperature, or also for additionally uv-xray
* Read ANDES2 model for following radmc postprocessing
* Run radiative transfer with radmc in modes:
  * `mctherm` -- dust temperature calculation
  * `mcmono` -- radiative transfer to find local radiation field
  * `image` lines -- generate emission line datacubes,
  both in `radmc.out` and `.fits`, get line data from the 
  copy of LAMDA database saved as a subpackage.
* Fit the data with model using `ultranest`
* Some other modules are in development stage and are not listed here
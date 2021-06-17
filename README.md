# DiskCheF: Disk Chemical Fitter

## Installation:

For development and usage:

```
# Preferred, requires SSH key setup on gitlab https://gitlab.com/-/profile/keys
git clone git@gitlab.com:SmirnGreg/diskchef.git  
# OR 
git clone https://gitlab.com/SmirnGreg/diskchef.git

cd diskchef
pip install -e .

git pull # to pull latest changes on currently tracked branch
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
  * mctherm -- dust temperature calculation
  * image lines -- generate emission line datacubes, 
  both in radmc.out and .fits, get line data from the 
  copy of LAMDA database saved as a subpackage.

* Some other modules are in development stage and are not listed here
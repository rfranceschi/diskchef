#!/bin/bash -l

#SBATCH -J diskchef
#SBATCH -o ./diskchef_log.out.%j
#SBATCH -e ./diskchef_log.err.%j

#SBATCH --nodes 32
#SBATCH --ntasks-per-node=80
#SBATCH --ntasks-per-core=2

#SBATCH --time=24:00:00

export DISPLAY=""

srun python -u dc_fit.py

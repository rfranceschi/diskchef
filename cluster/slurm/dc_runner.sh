#!/bin/bash -l

#SBATCH -J diskchef

#SBATCH --mem=100G
#SBATCH --nodes 16
#SBATCH --ntasks-per-node=40

#SBATCH --time=24:00:00

module load mpi
conda activate diskchef

export DISPLAY=""

srun python -u dc_fit.py

# DiskCheF HPC disk fitting example

This example demonstrates how the disk can be fitted on MPCDF cluster cobra

First, run 
```bash
python dc_generate_model.py
```

This will create a folder named `Reference` containing the radmc3d output of the fiducial model 
with the parameters described in the `dc_generate_model.py` file.

Then, submit the slurm job by
```bash
sbatch dc_runner.sh
``` 

This will take time...

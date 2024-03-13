#!/bin/bash

#SBATCH --job-name=randomtest
#SBATCH --output=./outdir01/test-%a.out
#SBATCH --error=./outdir01/test-%a.err
#
#SBATCH --ntasks=1
#SBATCH --time=1440:00
#SBATCH --mem-per-cpu=8000
#
#SBATCH --array=1-200
source activate my_env

#The "output_location" directory must be created before running this program.
output_location='./outdir01/'
program_location='./programs/'

#Copy necessary programs from my folder to JOBFS
cp ${program_location}/* ${JOBFS}
#Make a folder for the results in JOBFS
mkdir ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID
#Run the program
python ${JOBFS}/ns_sim_random_vary_EFAC.py --Nobs 1000 --Tdays 1000 --Tfit 0.0000001 --Nfit_min 3 --T_error_in 1e-16 --out_directory ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID --tag test_$SLURM_ARRAY_TASK_ID
#Move results folder to my directory
mv ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID ${output_location}


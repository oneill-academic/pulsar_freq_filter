#!/bin/bash

#SBATCH --job-name=randomtest
#SBATCH --output=./outdir01/test-%a.out
#SBATCH --error=./outdir01/test-%a.err
#
#SBATCH --ntasks=1
#SBATCH --time=1440:00
#SBATCH --mem-per-cpu=800
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
python ${JOBFS}/ns_sim_random_vary_EM_only_EFAC.py --R_in 1e-27 --R_out 1e-23 --Nobs 1000 --Tdays 1000 --out_directory ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID --tag test_$SLURM_ARRAY_TASK_ID --Nwalks 10 --Npoints 100
#Move results folder to my directory
mv ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID ${output_location}


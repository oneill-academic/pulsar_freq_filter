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
parfile='../../J1359-6038/J1359-6038.par'
timfile='../../J1359-6038/J1359-6038.tim'

#Copy necessary programs from my folder to JOBFS
cp ${program_location}/* ${JOBFS}
#Make a folder for the results in JOBFS
mkdir ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID
#Run the program
python ${JOBFS}/ns_sim_cadence_vary_EFAC.py --parfile ${parfile} --timfile ${timfile} --Tfit 10 --Nfit_min 3 --out_directory ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID --tag test_$SLURM_ARRAY_TASK_ID
#Move results folder to my directory
mv ${JOBFS}/outdir01_$SLURM_ARRAY_TASK_ID ${output_location}


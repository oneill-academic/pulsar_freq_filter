#!/bin/bash

#SBATCH --job-name=randomtest
#SBATCH --output=./outdir01/test.out
#SBATCH --error=./outdir01/test.err
#
#SBATCH --ntasks=1
#SBATCH --time=1440:00
#SBATCH --mem-per-cpu=800
#
#SBATCH --array=1-1
source activate my_env

pulsar="J1359-6038"
Tfit=10
Nfit_min=3
tag="test"

#The "output_location" directory must be created before running this program.
output_location='./outdir01/'
program_location='./programs/'
freqfile="../make_freq_files/outdir01_J1359-6038_10_3/test_${Tfit}_${Nfit_min}_nooverlap_freqs_1e-07cut.freq"

#Copy necessary programs from my folder to JOBFS
cp ${program_location}/* ${JOBFS}
#Make a folder for the results in JOBFS
#Run the program
#Move results folder to my directory
mkdir ${JOBFS}/outdir01_${pulsar}_${Tfit}_${Nfit_min}
python ${JOBFS}/run_on_real_data_onecomp_bins.py --Nwalks 100 --Npoints 10000 --freqfile ${freqfile} --out_directory ${JOBFS}/outdir01_${pulsar}_${Tfit}_${Nfit_min} --tag ${tag}
mv ${JOBFS}/outdir01_${pulsar}_${Tfit}_${Nfit_min} ${output_location}


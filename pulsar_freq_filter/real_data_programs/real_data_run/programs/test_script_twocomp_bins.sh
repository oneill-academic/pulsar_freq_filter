#!/bin/bash

pulsar="J1359-6038"
#Tfit=$1
#Nfit_min=$2
Tfit=10
Nfit_min=3
tag="test"
out_directory="./outdir02_${pulsar}_${Tfit}_${Nfit_min}/"
cut_freqfile="../../make_freq_files/outdir01_J1359-6038_10_3/${tag}_${Tfit}_${Nfit_min}_nooverlap_freqs_1e-07cut.freq"

python run_on_real_data_twocomp_bins.py --Nwalks 100 --Npoints 1000 --freqfile ${cut_freqfile} --out_directory ${out_directory} --tag ${tag}


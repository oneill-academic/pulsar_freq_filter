#!/bin/bash

pulsar="J1359-6038"
Tfit=10
Nfit_min=3
parfile="../../${pulsar}/${pulsar}.par"
input_timfile="../../${pulsar}/${pulsar}.tim"
out_directory="./outdir01_${pulsar}_${Tfit}_${Nfit_min}/"
tag="test_${Tfit}_${Nfit_min}"
threshold=1e-7

python fit_toas_nooverlap.py --Tfit ${Tfit} --Nfit_min ${Nfit_min} --parfile ${parfile} --input_timfile ${input_timfile} --out_directory ${out_directory} --tag ${tag} --threshold ${threshold}


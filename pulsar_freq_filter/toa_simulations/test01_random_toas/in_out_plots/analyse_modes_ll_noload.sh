#!/bin/bash

out_directory='outdir01'
tag='test'
Nsims=200
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_nooverlap' 
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_nooverlap_control' 
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_overlap' 
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_overlap_control'
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_true'
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_true_control' 


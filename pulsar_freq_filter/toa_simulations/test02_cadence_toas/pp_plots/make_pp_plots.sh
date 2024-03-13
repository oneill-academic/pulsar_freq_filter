#!/bin/bash

out_directory='outdir01'
tag='test'
Nsims=200
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_nooverlap'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_nooverlap_control'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_overlap'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_overlap_control'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_true'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_true_control'


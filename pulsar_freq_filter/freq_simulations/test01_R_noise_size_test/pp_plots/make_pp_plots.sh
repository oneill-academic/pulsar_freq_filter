#!/bin/bash

out_directory='outdir01'
tag='test'
Nsims=200
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'

out_directory='outdir02'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'

out_directory='outdir03'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'

out_directory='outdir04'
python make_pp_plots.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'


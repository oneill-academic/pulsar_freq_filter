#!/bin/bash
out_directory='outdir01'
tag='test'
Nsims=200
Ntests=4
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'

out_directory='outdir02'
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'

out_directory='outdir03'
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'

out_directory='outdir04'
python analyse_modes_ll_noload.py --out_directory ${out_directory} --tag ${tag} --Nsims ${Nsims} --savestr '_noEFAC'


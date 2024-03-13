#!/bin/bash/

pulsar="J1359-6038"
timfile="../../../${pulsar}/${pulsar}.tim"
parfile="../../../${pulsar}/${pulsar}.par"

python ns_sim_cadence_vary_EFAC.py --Tfit 10 --Nfit_min 3 --parfile ${parfile} --timfile ${timfile} --out_directory "./outdir01/" --tag "test"


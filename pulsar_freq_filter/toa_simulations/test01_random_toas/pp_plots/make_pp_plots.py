import bilby
import argparse
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', size=15)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--out_directory", help="location of bilby results", default="./outdir", type=str)
parser.add_argument("--tag", help="tag to include in saving information", default=None)
parser.add_argument("--Nsims", help="Number of simulations", default=10, type=int)
parser.add_argument("--savestr", help="suffix for plot save name", default="", type=str)
params = parser.parse_args()

labels = ['ratio', 'tau', 'Qc', 'Qs', 'lag', 'omgc_dot', 'omgc_0', 'EFAC', 'EQUAD']
newlabels = ['$r$', '$\\tau$', '$Q_{\\rm c}$', '$Q_{\\rm s}$',
             '$\\langle \\Omega_{\\rm c} - \\Omega_{\\rm s}\\rangle$',
             '$\\langle \\dot{\\Omega}_{\\rm c}\\rangle$', 
             '$\\Omega_{\\rm c}(0)$', 'EFAC', 'EQUAD']

newlabs = {}
for ii,l in enumerate(labels):
     newlabs[l] = newlabels[ii]
 
outdirectory = Path(f"{params.out_directory}")
outdirectory.mkdir(parents=True, exist_ok=True)
results = []
for jj in tqdm(range(1, params.Nsims+1)):
    output_directory = f"../{params.out_directory}/{params.out_directory}_{jj}"
    result_fname = Path(f"{output_directory}/{params.tag}_{jj}{params.savestr}_result.json")
    params_fname = Path(f"{output_directory}/{params.tag}_{jj}_simulation_parameters.json")
    if result_fname.exists() and params_fname.exists():
        result = bilby.result.read_in_result(result_fname)
        with open(params_fname, 'r') as mydata:
            mcmc_params = json.load(mydata)
    else:
        print(f'could not find: {jj}')
        continue
    inj_params = {'ratio': mcmc_params['r'], 'tau': mcmc_params['tau'], 
                  'Qc': mcmc_params['Qc'], 'Qs': mcmc_params['Qs'], 
                  'lag': mcmc_params['lag'], 'omgc_dot': mcmc_params['omgc_dot']}
    result.injection_parameters = inj_params
    if len(result.posterior) == 0:
        print('Loading failed')
        continue
    for key in result.priors.keys():
        result.priors[key].latex_label = newlabs[key]
    results.append(result)
fig, pvals = bilby.result.make_pp_plot(results,
        filename=f"./{params.out_directory}/{params.tag}_pp_plot{params.savestr}.png",
        confidence_interval=[0.90], )
#Add a thing to save pvals to a file
print(pvals)
fig.figsize = (36,36)
plt.legend(fontsize=15)

plt.xlabel('C.I.', fontsize=16)
plt.ylabel('Fraction of events in C.I.', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig(f"./{params.out_directory}/{params.tag}_pp_plot{params.savestr}.png")


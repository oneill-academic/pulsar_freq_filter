import bilby
import argparse
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--out_directory", help="location of bilby results", default="./outdir", type=str)
parser.add_argument("--tag", help="tag to include in saving information", default=None)
parser.add_argument("--Nsims", help="Number of simulations", default=10, type=int)
parser.add_argument("--savestr", help="suffix for plot save name", default="", type=str)
params = parser.parse_args()


outdirectory = Path(f"{params.out_directory}")
outdirectory.mkdir(parents=True, exist_ok=True)
print(f"../{params.out_directory}/{params.out_directory}_1/{params.tag}_1{params.savestr}_result.json")
print(f"../{params.out_directory}/{params.out_directory}_1/{params.tag}_1_simulation_parameters.json")
for ii in range(1, params.Nsims+1):
    result_fname = Path(f"../{params.out_directory}/{params.out_directory}_{ii}/{params.tag}_{ii}{params.savestr}_result.json")
    params_fname = Path(f"../{params.out_directory}/{params.out_directory}_{ii}/{params.tag}_{ii}_simulation_parameters.json")
    if result_fname.exists() and params_fname.exists():
        result = bilby.result.read_in_result(result_fname)
        with open(params_fname, 'r') as mydata:
            mcmc_params = json.load(mydata)
        break
    else:
        print(f'could not find: {ii}')
        continue

#Read in min and max
log_r_min = np.log10(result.priors['ratio'].minimum)
log_r_max = np.log10(result.priors['ratio'].maximum)
log_tau_min = np.log10(result.priors['tau'].minimum)
log_tau_max = np.log10(result.priors['tau'].maximum)
log_Qc_min = np.log10(result.priors['Qc'].minimum)
log_Qc_max = np.log10(result.priors['Qc'].maximum)
log_Qs_min = np.log10(result.priors['Qs'].minimum)
log_Qs_max = np.log10(result.priors['Qs'].maximum)
lag_min = result.priors['lag'].minimum
lag_max = result.priors['lag'].maximum
#omgc_dot_min = result.priors['omgc_dot'].minimum
#omgc_dot_max = result.priors['omgc_dot'].maximum
omgc_dot_min = mcmc_params['sim_omgc_dot_min']
omgc_dot_max = mcmc_params['sim_omgc_dot_max']
#tauc = tau(1+1/r), taus = tau(1+r)
log_tauc_min = np.log10(result.priors['tau'].minimum*(1+1/result.priors['ratio'].maximum))
log_tauc_max = np.log10(result.priors['tau'].maximum*(1+1/result.priors['ratio'].minimum))
log_taus_min = np.log10(result.priors['tau'].minimum*(1+result.priors['ratio'].minimum))
log_taus_max = np.log10(result.priors['tau'].maximum*(1+result.priors['ratio'].maximum))

#Store injected parameter values
log_r_trues = []
log_r_modes = []
log_r_lowers = []
log_r_mids = []
log_r_uppers = []
log_r_widths = []

log_tau_trues = []
log_tau_modes = []
log_tau_lowers = []
log_tau_mids = []
log_tau_uppers = []
log_tau_widths = []

log_Qc_trues = []
log_Qc_modes = []
log_Qc_lowers = []
log_Qc_mids = []
log_Qc_uppers = []
log_Qc_widths = []

log_Qs_trues = []
log_Qs_modes = []
log_Qs_lowers = []
log_Qs_mids = []
log_Qs_uppers = []
log_Qs_widths = []

lag_trues = []
lag_modes = []
lag_lowers = []
lag_mids = []
lag_uppers = []
lag_widths = []

omgc_dot_trues = []
omgc_dot_modes = []
omgc_dot_lowers = []
omgc_dot_mids = []
omgc_dot_uppers = []
omgc_dot_widths = []

log_tauc_trues = []
log_tauc_modes = []
log_tauc_lowers = []
log_tauc_mids = []
log_tauc_uppers = []
log_tauc_widths = []

log_taus_trues = []
log_taus_modes = []
log_taus_lowers = []
log_taus_mids = []
log_taus_uppers = []
log_taus_widths = []

success_count = 0
for ii in range(1, params.Nsims+1):
    result_fname = Path(f"../{params.out_directory}/{params.out_directory}_{ii}/{params.tag}_{ii}{params.savestr}_result.json")
    params_fname = Path(f"../{params.out_directory}/{params.out_directory}_{ii}/{params.tag}_{ii}_simulation_parameters.json")
    if result_fname.exists() and params_fname.exists():
        result = bilby.result.read_in_result(result_fname)
        if len(result.posterior) == 0:
            print(f"result {ii} was empty")
            continue
        with open(params_fname, 'r') as mydata:
            mcmc_params = json.load(mydata)
        success_count += 1
    else:
        print(f'could not find: {ii}')
        continue

    r = mcmc_params['r']
    tau = mcmc_params['tau']
    Qc = mcmc_params['Qc']
    Qs = mcmc_params['Qs']
    lag = mcmc_params['lag']
    omgc_dot = mcmc_params['omgc_dot']
    tauc = mcmc_params['tauc']
    taus = mcmc_params['taus']

    log_r_trues.append(np.log10(r))
    log_tau_trues.append(np.log10(tau))
    log_Qc_trues.append(np.log10(Qc))
    log_Qs_trues.append(np.log10(Qs))
    lag_trues.append(lag)
    omgc_dot_trues.append(omgc_dot)
    log_tauc_trues.append(np.log10(tauc))
    log_taus_trues.append(np.log10(taus))

    #Read in parameter samples
    #Should I resort the parameters by posterior or just use the current sorting by likelihood.
    log_r_samples = np.log10(result.posterior['ratio'].to_numpy())
    log_tau_samples = np.log10(result.posterior['tau'].to_numpy())
    log_Qc_samples = np.log10(result.posterior['Qc'].to_numpy())
    log_Qs_samples = np.log10(result.posterior['Qs'].to_numpy())
    lag_samples = result.posterior['lag'].to_numpy()
    omgc_dot_samples = result.posterior['omgc_dot'].to_numpy()
    log_tauc_samples = np.log10(10**log_tau_samples*(1+10**(-log_r_samples)))
    log_taus_samples = np.log10(10**log_tau_samples*(1+10**log_r_samples))
    ll_samples = result.posterior['log_likelihood'].to_numpy()

    #Calculate and store the medians and widths of the samples for each parameter
    r_lower, r_mid, r_upper = np.percentile(log_r_samples, [25,50,75])
    tau_lower, tau_mid, tau_upper = np.percentile(log_tau_samples, [25,50,75])
    Qc_lower, Qc_mid, Qc_upper = np.percentile(log_Qc_samples, [25,50,75])
    Qs_lower, Qs_mid, Qs_upper = np.percentile(log_Qs_samples, [25,50,75])
    lag_lower, lag_mid, lag_upper = np.percentile(lag_samples, [25,50,75])
    omgc_dot_lower, omgc_dot_mid, omgc_dot_upper = np.percentile(omgc_dot_samples, [25,50,75])
    tauc_lower, tauc_mid, tauc_upper = np.percentile(log_tauc_samples, [25,50,75])
    taus_lower, taus_mid, taus_upper = np.percentile(log_taus_samples, [25,50,75])

    log_r_modes.append(log_r_samples[-1])
    log_r_lowers.append(r_lower)
    log_r_mids.append(r_mid)
    log_r_uppers.append(r_upper)
    log_r_widths.append(r_upper-r_lower)
    log_tau_modes.append(log_tau_samples[-1])
    log_tau_lowers.append(tau_lower)
    log_tau_mids.append(tau_mid)
    log_tau_uppers.append(tau_upper)
    log_tau_widths.append(tau_upper-tau_lower)
    log_Qc_modes.append(log_Qc_samples[-1])
    log_Qc_lowers.append(Qc_lower)
    log_Qc_mids.append(Qc_mid)
    log_Qc_uppers.append(Qc_upper)
    log_Qc_widths.append(Qc_upper-Qc_lower)
    log_Qs_modes.append(log_Qs_samples[-1])
    log_Qs_lowers.append(Qs_lower)
    log_Qs_mids.append(Qs_mid)
    log_Qs_uppers.append(Qs_upper)
    log_Qs_widths.append(Qs_upper-Qs_lower)
    lag_modes.append(lag_samples[-1])
    lag_lowers.append(lag_lower)
    lag_mids.append(lag_mid) 
    lag_uppers.append(lag_upper)
    lag_widths.append(lag_upper-lag_lower)
    omgc_dot_modes.append(omgc_dot_samples[-1])
    omgc_dot_lowers.append(omgc_dot_lower)
    omgc_dot_mids.append(omgc_dot_mid) 
    omgc_dot_uppers.append(omgc_dot_upper)
    omgc_dot_widths.append(omgc_dot_upper-omgc_dot_lower)
    log_tauc_modes.append(log_tauc_samples[-1])
    log_tauc_lowers.append(tauc_lower)
    log_tauc_mids.append(tauc_mid)
    log_tauc_uppers.append(tauc_upper)
    log_tauc_widths.append(tauc_upper-tauc_lower)
    log_taus_modes.append(log_taus_samples[-1])
    log_taus_lowers.append(taus_lower)
    log_taus_mids.append(taus_mid)
    log_taus_uppers.append(taus_upper)
    log_taus_widths.append(taus_upper-taus_lower)

print("success_count =", success_count)

log_r_trues = np.asarray(log_r_trues)
log_r_modes = np.asarray(log_r_modes)
log_r_lowers = np.asarray(log_r_lowers)
log_r_mids = np.asarray(log_r_mids)
log_r_uppers = np.asarray(log_r_uppers)

log_tau_trues = np.asarray(log_tau_trues)
log_tau_modes = np.asarray(log_tau_modes)
log_tau_lowers = np.asarray(log_tau_lowers)
log_tau_mids = np.asarray(log_tau_mids)
log_tau_uppers = np.asarray(log_tau_uppers)

log_Qc_trues = np.asarray(log_Qc_trues)
log_Qc_modes = np.asarray(log_Qc_modes)
log_Qc_lowers = np.asarray(log_Qc_lowers)
log_Qc_mids = np.asarray(log_Qc_mids)
log_Qc_uppers = np.asarray(log_Qc_uppers)

log_Qs_trues = np.asarray(log_Qs_trues)
log_Qs_modes = np.asarray(log_Qs_modes)
log_Qs_lowers = np.asarray(log_Qs_lowers)
log_Qs_mids = np.asarray(log_Qs_mids)
log_Qs_uppers = np.asarray(log_Qs_uppers)

lag_trues = np.asarray(lag_trues)
lag_modes = np.asarray(lag_modes)
lag_lowers = np.asarray(lag_lowers)
lag_mids = np.asarray(lag_mids)
lag_uppers = np.asarray(lag_uppers)

omgc_dot_trues = np.asarray(omgc_dot_trues)
omgc_dot_modes = np.asarray(omgc_dot_modes)
omgc_dot_lowers = np.asarray(omgc_dot_lowers)
omgc_dot_mids = np.asarray(omgc_dot_mids)
omgc_dot_uppers = np.asarray(omgc_dot_uppers)

log_tauc_trues = np.asarray(log_tauc_trues)
log_tauc_modes = np.asarray(log_tauc_modes)
log_tauc_lowers = np.asarray(log_tauc_lowers)
log_tauc_mids = np.asarray(log_tauc_mids)
log_tauc_uppers = np.asarray(log_tauc_uppers)

log_taus_trues = np.asarray(log_taus_trues)
log_taus_modes = np.asarray(log_taus_modes)
log_taus_lowers = np.asarray(log_taus_lowers)
log_taus_mids = np.asarray(log_taus_mids)
log_taus_uppers = np.asarray(log_taus_uppers)



#Plot the injected vs recovered parameters.
#Colour the points according to the width of the posterior.
#figsize: first number is width, second is height
#fig, axs = plt.subplots(2, 3, figsize=(20,14))
#fig, axs = plt.subplots(2, 3, figsize=(22,15))
fig, axs = plt.subplots(2, 3, figsize=(21,12))

im = axs[0,0].scatter(log_tau_trues, log_tau_modes, c=log_tau_widths)
axs[0,0].plot(log_tau_trues, log_tau_trues, c='tab:red')
axs[0,0].set_xlabel('Injected $\log_{10}(\\tau)$', fontsize=21)
axs[0,0].set_ylabel('Recovered $\log_{10}(\\tau)$', fontsize=21)
axs[0,0].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[0,0])
cbar.ax.tick_params(labelsize=18)

im = axs[1,0].scatter(log_r_trues, log_r_modes, c=log_r_widths)
axs[1,0].plot(log_r_trues, log_r_trues, c='tab:red')
axs[1,0].set_xlabel('Injected $\log_{10}(r)$', fontsize=21)
axs[1,0].set_ylabel('Recovered $\log_{10}(r)$', fontsize=21)
#axs[1,0].set_xlabel('Injected $\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', fontsize=17)
#axs[1,0].set_ylabel('Recovered $\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', fontsize=17)
axs[1,0].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[1,0])
cbar.ax.tick_params(labelsize=18)

im = axs[0,1].scatter(log_Qc_trues, log_Qc_modes, c=log_Qc_widths)
axs[0,1].plot(log_Qc_trues, log_Qc_trues, c='tab:red')
axs[0,1].set_xlabel('Injected $\log_{10}(Q_{\\rm{c}})$', fontsize=21)
axs[0,1].set_ylabel('Recovered $\log_{10}(Q_{\\rm{c}})$', fontsize=21)
axs[0,1].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[0,1])
cbar.ax.tick_params(labelsize=18)

im = axs[1,1].scatter(log_Qs_trues, log_Qs_modes, c=log_Qs_widths)
axs[1,1].plot(log_Qs_trues, log_Qs_trues, c='tab:red')
axs[1,1].set_xlabel('Injected $\log_{10}(Q_{\\rm{s}})$', fontsize=21)
axs[1,1].set_ylabel('Recovered $\log_{10}(Q_{\\rm{s}})$', fontsize=21)
axs[1,1].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[1,1])
cbar.ax.tick_params(labelsize=18)

im = axs[0,2].scatter(omgc_dot_trues, omgc_dot_modes, c=omgc_dot_widths)
axs[0,2].plot([omgc_dot_min, omgc_dot_max], [omgc_dot_min, omgc_dot_max], c='tab:red')
axs[0,2].set_xlim([omgc_dot_min, omgc_dot_max])
axs[0,2].set_ylim([omgc_dot_min, omgc_dot_max])
axs[0,2].set_xscale('symlog', linthresh=1e-14)
axs[0,2].set_yscale('symlog', linthresh=1e-14)
axs[0,2].xaxis.get_offset_text().set_fontsize(18)
axs[0,2].yaxis.get_offset_text().set_fontsize(18)
axs[0,2].set_xlabel('Injected $\\langle \\dot\\Omega_c\\rangle$', fontsize=21)
axs[0,2].set_ylabel('Recovered $\\langle \\dot\\Omega_c\\rangle$', fontsize=21)
axs[0,2].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[0,2])
cbar.ax.tick_params(labelsize=18)
cbar.ax.yaxis.offsetText.set(size=18)

im = axs[1,2].scatter(lag_trues, lag_modes, c=lag_widths)
axs[1,2].plot(lag_trues, lag_trues, c='tab:red')
axs[1,2].set_xlabel('Injected $\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$', fontsize=21)
axs[1,2].set_ylabel('Recovered $\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$', fontsize=21)
axs[1,2].tick_params(rotation=45, labelsize=18)
axs[1,2].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[1,2])
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.subplots_adjust(left=0.06)
plt.savefig(f"{params.out_directory}/in_out_modes_ll_colour{params.savestr}.png")
plt.close()



#Plot the injected vs recovered parameters.
#Colour the points according to the width of the posterior.
#figsize: first number is width, second is height
#fig, axs = plt.subplots(2, 1, figsize=(6,11))
fig, axs = plt.subplots(2, 1, figsize=(7,11))

im = axs[0].scatter(log_tau_trues, log_tau_modes, c=log_tau_widths)
axs[0].plot(log_tau_trues, log_tau_trues, c='tab:red')
axs[0].set_xlabel('Injected $\log_{10}(\\tau)$', fontsize=21)
axs[0].set_ylabel('Recovered $\log_{10}(\\tau)$', fontsize=21)
axs[0].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[0])
cbar.ax.tick_params(labelsize=18)

im = axs[1].scatter(log_Qc_trues, log_Qc_modes, c=log_Qc_widths)
axs[1].plot(log_Qc_trues, log_Qc_trues, c='tab:red')
axs[1].set_xlabel('Injected $\log_{10}(Q_{\\rm{c}})$', fontsize=21)
axs[1].set_ylabel('Recovered $\log_{10}(Q_{\\rm{c}})$', fontsize=21)
axs[1].tick_params(labelsize=18)
cbar = fig.colorbar(im, ax=axs[1])
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig(f"{params.out_directory}/in_out_modes_ll_tau_Qc_colour{params.savestr}.png")
plt.close()


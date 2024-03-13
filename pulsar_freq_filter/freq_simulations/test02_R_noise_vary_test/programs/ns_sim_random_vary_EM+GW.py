#! /usr/bin/env python
print("Beginning program")
import sys
from fake_data import two_component_fake_data
from models import TwoComponentModel, param_map, param_map2
from sample import KalmanLikelihood
import bilby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from corner import corner
from pandas import DataFrame
import argparse
import json
from pathlib import Path
plt.rc('text', usetex=False)
print("Importing modules completed")

def run(params):
    outdirectory = Path(params.out_directory)
    outdirectory.mkdir(parents=True, exist_ok=True)

    if not params.resume_run:
        #Randomly choose the parameters for the simulation.
        sim_r_min = 1e-2
        sim_r_max = 1e2
        sim_tau_min = 1e5
        sim_tau_max = 1e8
        sim_Qc_min = 1e-30
        sim_Qc_max = 1e-20
        sim_Qs_min = 1e-30
        sim_Qs_max = 1e-20
        sim_lag_min = -1e-3
        sim_lag_max = 1e-3
        sim_omgc_dot_min = -1e-10
        sim_omgc_dot_max = -1e-13
        omgc_0 = 10
        priors = bilby.core.prior.PriorDict()
        priors['ratio'] = bilby.core.prior.LogUniform(minimum=sim_r_min, maximum=sim_r_max, name='ratio', latex_label='$\\frac{\\tau_s}{\\tau_c}$')
        priors['tau'] = bilby.core.prior.LogUniform(minimum=sim_tau_min, maximum=sim_tau_max, name='tau', latex_label='$\\frac{\\tau_c + \\tau_s}{\\tau_c\\tau_s}$')
        priors['Qc'] = bilby.core.prior.LogUniform(minimum=sim_Qc_min, maximum=sim_Qc_max, name='Qc', latex_label='$Q_c$')
        priors['Qs'] = bilby.core.prior.LogUniform(minimum=sim_Qs_min, maximum=sim_Qs_max, name='Qs', latex_label='$Q_s$')
        priors['lag'] = bilby.core.prior.Uniform(minimum=sim_lag_min, maximum=sim_lag_max, name='lag', latex_label='$\\Omega_c - \\Omega_s$')
        priors['neg_omgc_dot'] = bilby.core.prior.LogUniform(minimum=-sim_omgc_dot_max, maximum=-sim_omgc_dot_min, name='neg_omgc_dot', latex_label='$-\\langle \\dot{\\Omega}_c \\rangle$')
        priors['omgc_0'] = bilby.core.prior.DeltaFunction(omgc_0, name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')

        vals = priors.sample()
        tauc, taus, Qc, Qs, Nc, Ns = param_map2(vals)

        #Calculate other parameter combinations.
        r = taus / tauc
        tau = tauc * taus / (tauc + taus)
        lag = tau * (Nc - Ns)
        omgc_dot = (Nc * tauc + Ns * taus)/(tauc + taus)
        sigmac = Qc**0.5
        sigmas = Qs**0.5

        #Calculate initial frequencies.
        omgc_0 = vals['omgc_0']
        omgs_0 = omgc_0 - lag

        #Set the measurement noise level
        R_in = params.R_in
        R_out = params.R_out

        #Simulate the pulsar frequenices
        Nobs = params.Nobs
        Tdays = params.Tdays
        Tobs = Tdays*86400
        times = np.linspace(0, Tobs, Nobs*1000)
        data, states = two_component_fake_data(times, tauc=tauc, taus=taus, sigmac=sigmac,
                                               sigmas=sigmas, Nc=Nc, Ns=Ns, omgc_0=omgc_0,
                                               omgs_0=omgs_0, Rc=R_in, Rs=R_in)
        indexs = np.sort(np.random.choice(Nobs*1000, Nobs, replace = False))
        times = times[indexs]
        data = data[indexs]
        states = states[indexs, :]

        #Test Kalman filter ll function
        param_dict = {'ratio': r, 'tau': tau, 'Qc': Qc, 'Qs': Qs, 'lag': lag, 
                      'omgc_dot': omgc_dot, 'omgc_0': data[0, 0], 'omgs_0': data[0, 1],
                      'EFAC': 2, 'EQUAD': 1e-30}
        Nobs = len(times)
        design = np.eye(2)*1.
        measurement_cov = np.asarray([np.eye(2)*R_out for ii in range(Nobs)]).T
        model = TwoComponentModel(times, data, measurement_cov, design, param_map)
        likelihood = KalmanLikelihood(model)
        ll = model.loglike(params = param_dict, return_states = False, loglikelihood_burn=-1)
        print("ll =", ll)

        #Plot omgc data and residuals
        plt.figure(1, figsize=(40,20))
        plt.subplot(2,1,1)
        plt.plot(times, data[:, 0], marker='.', linestyle=None, color="b")
        plt.plot(times, data[:, 1], marker='.', linestyle=None, color="g")
        plt.xlabel("times")
        plt.ylabel("data")
        plt.title("Measurements over time")

        plt.subplot(2,1,2)
        plt.plot(times, data[:, 0]-omgc_0-omgc_dot*times, marker='.', linestyle=None, color="b")
        plt.plot(times, data[:, 1]-omgs_0-omgc_dot*times, marker='.', linestyle=None, color="g")
        plt.xlabel("times")
        plt.ylabel("data")
        plt.title("Residuals over time")

        plt.tight_layout()
        plt.savefig(f"{params.out_directory}/dataplot.png")
        plt.close()
        
        #Store data used in the simulation.
        mcmc_params = {'r': r, 'tau': tau, 'Qc': Qc, 'Qs': Qs, 'lag': lag, 'omgc_dot': omgc_dot, 'omgc_0': omgc_0,
                       'tauc': tauc, 'taus': taus, 'sigmac': sigmac, 'sigmas': sigmas, 'Nc': Nc, 'Ns': Ns, 
                       'R_in': R_in, 'R_out': R_out, 'Nobs': Nobs, 'Tdays': Tdays, 'Tobs': Tobs, 
                       'times': times.tolist(), 'data': data.tolist(),
                       'sim_r_min': sim_r_min, 'sim_r_max': sim_r_max, 'sim_tau_min': sim_tau_min, 'sim_tau_max': sim_tau_max,
                       'sim_Qc_min': sim_Qc_min, 'sim_Qc_max': sim_Qc_max, 'sim_Qs_min': sim_Qs_min, 'sim_Qs_max': sim_Qs_max,
                       'sim_lag_min': sim_lag_min, 'sim_lag_max': sim_lag_max, 'sim_omgc_dot_min': sim_omgc_dot_min, 'sim_omgc_dot_max': sim_omgc_dot_max}
        with open(f'{params.out_directory}/{params.tag}_simulation_parameters.json', 'w') as mydata:
            json.dump(mcmc_params, mydata)
    else:
        #Reload data used in the simulation if the program needs to be rerun.
        with open(f'{params.out_directory}/{params.tag}_simulation_parameters.json', 'r') as mydata:
            mcmc_params = json.load(mydata)
        r = mcmc_params['r']
        tau = mcmc_params['tau']
        Qc = mcmc_params['Qc']
        Qs = mcmc_params['Qs']
        lag = mcmc_params['lag']
        omgc_dot = mcmc_params['omgc_dot']
        omgc_0 = mcmc_params['omgc_0']
        tauc = mcmc_params['tauc']
        taus = mcmc_params['taus']
        sigmac = mcmc_params['sigmac']
        sigmas = mcmc_params['sigmas']
        Nc = mcmc_params['Nc']
        Ns = mcmc_params['Ns']
        R_in = mcmc_params['R_in']
        R_out = mcmc_params['R_out']
        times = np.asarray(mcmc_params['times'])
        data = np.asarray(mcmc_params['data'])
        sim_r_min = mcmc_params['sim_r_min']
        sim_r_max = mcmc_params['sim_r_max']
        sim_tau_min = mcmc_params['sim_tau_min']
        sim_tau_max = mcmc_params['sim_tau_max']
        sim_Qc_min = mcmc_params['sim_Qc_min']
        sim_Qc_max = mcmc_params['sim_Qc_max']
        sim_Qs_min = mcmc_params['sim_Qs_min']
        sim_Qs_max = mcmc_params['sim_Qs_max']
        sim_lag_min = mcmc_params['sim_lag_min']
        sim_lag_max = mcmc_params['sim_lag_max']
        sim_omgc_dot_min = mcmc_params['sim_omgc_dot_min']
        sim_omgc_dot_max = mcmc_params['sim_omgc_dot_max']

    #Set up the Kalman filter
    Nobs = len(times)
    design = np.eye(2)*1.
    measurement_cov = np.asarray([np.eye(2)*R_out for ii in range(Nobs)]).T
    model = TwoComponentModel(times, data, measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)
 
    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times, data[:, 0], 1, w=1/np.sqrt(np.squeeze(measurement_cov[0, 0, :])), cov=True)
    omgd_low = p[0] - np.sqrt(V[0,0]) * 1000
    omgd_high = p[0] + np.sqrt(V[0,0]) * 1000

    #Set priors
    r_min = sim_r_min
    r_max = sim_r_max
    tau_min = sim_tau_min
    tau_max = sim_tau_max
    Qc_min = sim_Qc_min
    Qc_max = sim_Qc_max
    Qs_min = sim_Qs_min
    Qs_max = sim_Qs_max
    lag_min = sim_lag_min
    lag_max = sim_lag_max
    omgc_dot_min = omgd_low
    omgc_dot_max = omgd_high
    priors = bilby.core.prior.PriorDict()
    priors['ratio'] = bilby.core.prior.LogUniform(minimum=r_min, maximum=r_max, name='ratio', latex_label='$\\frac{\\tau_s}{\\tau_c}$')
    priors['tau'] = bilby.core.prior.LogUniform(minimum=tau_min, maximum=tau_max, name='tau', latex_label='$\\frac{\\tau_c + \\tau_s}{\\tau_c\\tau_s}$')
    priors['Qc'] = bilby.core.prior.LogUniform(minimum=Qc_min, maximum=Qc_max, name='Qc', latex_label='$Q_c$')
    priors['Qs'] = bilby.core.prior.LogUniform(minimum=Qs_min, maximum=Qs_max, name='Qs', latex_label='$Q_s$')
    priors['lag'] = bilby.core.prior.Uniform(minimum=lag_min, maximum=lag_max, name='lag', latex_label='$\\Omega_c - \\Omega_s$')
    priors['omgc_dot'] = bilby.core.prior.Uniform(minimum=omgc_dot_min, maximum=omgc_dot_max, name='omgc_dot', latex_label='$\\langle \\dot{\\Omega}_c \\rangle$')
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data[0, 0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['omgs_0'] = bilby.core.prior.DeltaFunction(data[0, 1], name='omgs_0', latex_label='$\\Omega_{\\rm{s}, 0}$')
    priors['EFAC'] = bilby.core.prior.DeltaFunction(1, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.DeltaFunction(0, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=params.Nwalks, npoints=params.Npoints,
                               resume=params.resume_run, outdir=params.out_directory, 
                               label=f'{params.tag}_noEFAC', check_point_plot=False)

    #Make plots of the sampler results
    samples = result.posterior.to_numpy()[:, :6].copy()
    samples[:, 0:4] = np.log10(samples[:, 0:4])
    print("samples[-1] =", samples[-1])
    print("10**samples[-1] =", 10**samples[-1])
    labels = ['$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', '$\\tau$', '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$', 
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$', '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$', 
              '$\\langle \\dot\\Omega_c\\rangle$']
    fig = corner(samples, truths=[np.log10(taus/tauc), np.log10(tau), np.log10(sigmac**2), np.log10(sigmas**2), lag, omgc_dot],
                 #range = [(-2, 2), (5, 8), (-30, -20), (-30, -20), (-1e-2, 1e-2), (-1e-12, -1e-13)],
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (omgc_dot_min, omgc_dot_max)],
                 color='b', smooth=True, smooth1d=True, levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_corner_noEFAC.png')
    plt.close()

    #Replot parameters as histograms
    plt.figure(1, figsize=(20,12))

    plt.subplot(4,2,1)
    plt.hist(samples[:, 0], 50, density=True, facecolor="tab:orange", alpha=0.6)
    plt.axvline(np.log10(taus/tauc), color='k')
    plt.xlabel("$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$")

    plt.subplot(4,2,2)
    plt.hist(samples[:, 1], 50, density=True, facecolor="tab:red", alpha=0.6)
    plt.axvline(np.log10(tau), color='k')
    plt.xlabel("$\\tau$")

    tau_values = 10**samples[:, 1]
    r_values = 10**samples[:, 0]
    tauc_values = (1+1/r_values)*tau_values
    taus_values = (1+r_values)*tau_values

    plt.subplot(4,2,3)
    plt.hist(np.log10(tauc_values), 50, density=True, facecolor="tab:blue", alpha=0.6)
    plt.axvline(np.log10(tauc), color='k')
    plt.xlabel("$\\tau_c$")

    plt.subplot(4,2,4)
    plt.hist(np.log10(taus_values), 50, density=True, facecolor="tab:green", alpha=0.6)
    plt.axvline(np.log10(taus), color='k')
    plt.xlabel("$\\tau_s$")

    plt.subplot(4,2,5)
    plt.hist(samples[:, 2], 50, density=True, facecolor="tab:blue", alpha=0.6)
    plt.axvline(np.log10(Qc), color='k')
    plt.xlabel("$Q_c$")

    plt.subplot(4,2,6)
    plt.hist(samples[:, 3], 50, density=True, facecolor="tab:green", alpha=0.6)
    plt.axvline(np.log10(Qs), color='k')
    plt.xlabel("$Q_s$")

    plt.subplot(4,2,7)
    plt.hist(samples[:, 4], 50, density=True, facecolor="k", alpha=0.4)
    plt.axvline(lag, color='k')
    plt.xlabel("$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$")

    plt.subplot(4,2,8)
    plt.hist(samples[:, 5], 50, density=True, facecolor="k", alpha=0.8)
    plt.axvline(omgc_dot, color='k')
    plt.xlabel("$\\langle \\dot\\Omega_c\\rangle$")

    plt.title("Recovered parameters")
    plt.tight_layout()
    plt.savefig(f"{params.out_directory}/{params.tag}_param_hists_noEFAC.png")
    plt.close()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #Simulation parameters
    parser.add_argument("--Nobs", help="number of observations", type=int, required=True)
    parser.add_argument("--Tdays", help="number of days observations are spread over", type=float, required=True)
    parser.add_argument("--R_in", help="measurement covariance for generating data", type=float)
    parser.add_argument("--R_out", help="measurement covariance for recovery in Kalman filter", type=float)
    #MCMC information
    parser.add_argument("--Nwalks", default=10, type=int, help="number of walks for mcmc")
    parser.add_argument("--Npoints", default=100, type=int, help="number of points for mcmc")
    parser.add_argument("--resume_run", help="resume mcmc run from where it left off", action="store_true", default=False)
    #Save information
    parser.add_argument("--out_directory", help="output directory", default="./outdir/", type=str)
    parser.add_argument("--tag", help="tag to include in saving information", default=None)
    args = parser.parse_args()
    run(args)


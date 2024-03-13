#! /usr/bin/env python
print("Beginning program")
import sys
import libstempo
from simulate_toas_random import simulate_toas_random
from fit_toas_overlap import toas_to_freqs_overlap
from fit_toas_nooverlap import toas_to_freqs_nooverlap
from models import TwoComponentModel, param_map, param_map2
from sample import KalmanLikelihood
import random
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
        #Randomly choose the parameters for the simulation
        sim_r_min = 1e-2
        sim_r_max = 1e2
        sim_tau_min = 1e5
        sim_tau_max = 1e8
        sim_Qc_min = 1e-30
        sim_Qc_max = 1e-16
        sim_Qs_min = 1e-30
        sim_Qs_max = 1e-16
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

        #Calculate other parameter combinations
        r = taus / tauc
        tau = tauc * taus / (tauc + taus)
        lag = tau * (Nc - Ns)
        omgc_dot = (Nc * tauc + Ns * taus)/(tauc + taus)
        sigmac = Qc**0.5
        sigmas = Qs**0.5

        #Calculate initial state
        omgc_0 = vals['omgc_0']
        omgs_0 = omgc_0 - lag
        phi_0 = 0

        #Set the length of the data set, the number of observations and the 
        #number of TOAs used for a frequency fit
        Nobs = params.Nobs
        Tdays = params.Tdays
        Tobs = Tdays*86400 
        Tfit = params.Tfit
        Nfit_min = params.Nfit_min
        T_error_in = params.T_error_in
        #R_out = params.R_out

        #Here 'overlap' means fitting frequencies with TEMPO using overlapping sets of TOAs
        #and 'nooverlap' means fitting frequencies with TEMPO using nonoverlapping sets TOAs.
        #'nooverlap' is the one that should be used.
        new_parfile = f'{params.out_directory}/{params.tag}.par'
        true_timfile = f'{params.out_directory}/{params.tag}_true.tim'
        true_freqfile = f'{params.out_directory}/{params.tag}_true.freq'
        overlap_freqfile = f'{params.out_directory}/{params.tag}_overlap.freq'
        nooverlap_freqfile = f'{params.out_directory}/{params.tag}_nooverlap.freq'
        #Simulate toas and true freqs
        simulate_toas_random(tau=tau, r=r, sigmac=sigmac, sigmas=sigmas, lag=lag, 
                             omgc_dot=omgc_dot, omgc_0=omgc_0, PEPOCH=0, phi_0=phi_0,
                             T_error_in=T_error_in, Nobs=Nobs, Tdays=Tdays, output_parfile=new_parfile,
                             output_timfile=true_timfile, output_freqfile=true_freqfile,
                             out_directory=params.out_directory, tag=params.tag)
        #Fit to simulated toas with tempo
        toas_to_freqs_overlap(Tfit=Tfit, Nfit_min=Nfit_min, parfile=new_parfile, 
                              input_timfile=true_timfile, output_freqfile=overlap_freqfile, 
                              out_directory=params.out_directory, tag=params.tag, threshold=None)
        #Fit to simulated toas with tempo (method 2)
        toas_to_freqs_nooverlap(Tfit=Tfit, Nfit_min=Nfit_min, parfile=new_parfile,
                                input_timfile=true_timfile, output_freqfile=nooverlap_freqfile, 
                                out_directory=params.out_directory, tag=params.tag, threshold=None)

        #Load simulated true freqs and times (no errors)
        data1 = np.loadtxt(true_freqfile)
        times_true = data1[:, 0]*86400
        data_true = data1[:, 1]*2*np.pi
        #Load simulated overlap fitted freqs and times and errors
        data2 = np.loadtxt(overlap_freqfile)
        times_overlap = data2[:, 0]*86400
        data_overlap = data2[:, 1]*2*np.pi
        R_overlap = (data2[:, 2]*2*np.pi)**2
        #Load simulated nooverlap fitted freqs and times and errors
        data3 = np.loadtxt(nooverlap_freqfile)
        times_nooverlap= data3[:, 0]*86400
        data_nooverlap = data3[:, 1]*2*np.pi
        R_nooverlap = (data3[:, 2]*2*np.pi)**2

        R_out = np.mean(R_overlap)
        R_true = np.ones(len(times_true))*R_out

        times_true = times_true.astype(np.float64)
        data_true = data_true.astype(np.float64)
        R_true = R_true.astype(np.float64)
        times_overlap = times_overlap.astype(np.float64)
        data_overlap = data_overlap.astype(np.float64)
        R_overlap = R_overlap.astype(np.float64)
        times_nooverlap = times_nooverlap.astype(np.float64)
        data_nooverlap = data_nooverlap.astype(np.float64)
        R_nooverlap = R_nooverlap.astype(np.float64)

        #Load pets0 (omgc_0 is the trend frequency at pets0)
        newpsr = libstempo.tempopulsar(parfile=new_parfile, timfile=true_timfile)
        omgc_0 = (newpsr['F0'].val*2*np.pi).astype(np.float64)
        pets0 = (newpsr['PEPOCH'].val).astype(np.float64)

        #Plot omgc data and residuals
        plt.figure(1, figsize=(12,12))
        plt.subplot(2,1,1)
        plt.plot(times_true, data_true, marker='.', linestyle=None, color='k', alpha=0.6, label='True')
        plt.plot(times_overlap, data_overlap, marker='.', linestyle=None, color='b', alpha=0.6, label='Tempo')
        plt.plot(times_nooverlap, data_nooverlap, marker='.', linestyle=None, color='r', alpha=0.6, label='Tempo2')
        plt.xlabel("times")
        plt.ylabel("data")
        plt.title("Measurements over time")

        plt.subplot(2,1,2)
        plt.plot(times_true, data_true-omgc_0-omgc_dot*(times_true-pets0*86400), marker='.', linestyle=None, color='k', alpha=0.6, label='True')
        plt.plot(times_overlap, data_overlap-omgc_0-omgc_dot*(times_overlap-pets0*86400), marker='.', linestyle=None, color='b', alpha=0.6, label='Tempo')
        plt.plot(times_nooverlap, data_nooverlap-omgc_0-omgc_dot*(times_nooverlap-pets0*86400), marker='.', linestyle=None, color='r', alpha=0.6, label='Tempo2')
        plt.xlabel("times")
        plt.ylabel("data")
        plt.title("Residuals over time")

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{params.out_directory}/{params.tag}_freqs_residuals_comparison.png")
        plt.close()

        #Store data used in the simulation.
        mcmc_params = {'r': taus/tauc, 'tau': tau, 'Qc': Qc, 'Qs': Qs, 'lag': lag, 'omgc_dot': omgc_dot, 'omgc_0': omgc_0,
                       'tauc': tauc, 'taus': taus, 'sigmac': sigmac, 'sigmas': sigmas, 'Nc': Nc, 'Ns': Ns, 'phi_0': phi_0,
                       'T_error_in': T_error_in, 'R_out': R_out, 'Nobs': Nobs, 'Tdays': Tdays, 'Tobs': Tobs, 'Tfit': Tfit, 'Nfit_min': Nfit_min, 'pets0': pets0, 
                       'times_true': times_true.tolist(), 'data_true': data_true.tolist(), 'R_true': R_true.tolist(),
                       'times_overlap': times_overlap.tolist(), 'data_overlap': data_overlap.tolist(), 'R_overlap': R_overlap.tolist(),
                       'times_nooverlap': times_nooverlap.tolist(), 'data_nooverlap': data_nooverlap.tolist(), 'R_nooverlap': R_nooverlap.tolist(),
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
        pets0 = mcmc_params['pets0']
        T_error_in = mcmc_params['T_error_in']
        R_out = mcmc_params['R_out']
        times_true = np.asarray(mcmc_params['times_true'])
        data_true = np.asarray(mcmc_params['data_true'])
        R_true = np.asarray(mcmc_params['R_true'])
        times_overlap = np.asarray(mcmc_params['times_overlap'])
        data_overlap = np.asarray(mcmc_params['data_overlap'])
        R_overlap = np.asarray(mcmc_params['R_overlap'])
        times_nooverlap = np.asarray(mcmc_params['times_nooverlap'])
        data_nooverlap = np.asarray(mcmc_params['data_nooverlap'])
        R_nooverlap = np.asarray(mcmc_params['R_nooverlap'])
        sim_r_min = np.asarray(mcmc_params['sim_r_min'])
        sim_r_max = np.asarray(mcmc_params['sim_r_max'])
        sim_tau_min = np.asarray(mcmc_params['sim_tau_min'])
        sim_tau_max = np.asarray(mcmc_params['sim_tau_max'])
        sim_Qc_min = np.asarray(mcmc_params['sim_Qc_min'])
        sim_Qc_max = np.asarray(mcmc_params['sim_Qc_max'])
        sim_Qs_min = np.asarray(mcmc_params['sim_Qs_min'])
        sim_Qs_max = np.asarray(mcmc_params['sim_Qs_max'])
        sim_lag_min = np.asarray(mcmc_params['sim_lag_min'])
        sim_lag_max = np.asarray(mcmc_params['sim_lag_max'])

    #Tempo2 (nonoverlapping toa sets) freqs
    print("\nTempo2 freqs\n")

    #Set up the Kalman filter
    Nobs_nooverlap = len(times_nooverlap)
    design = np.asarray([1., 0.]).reshape(1, 2)
    measurement_cov = R_nooverlap.copy().reshape((1, 1, Nobs_nooverlap))
    model = TwoComponentModel(times_nooverlap, data_nooverlap.reshape((Nobs_nooverlap, 1)), measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)

    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times_nooverlap, data_nooverlap, 1, w=1/np.sqrt(np.squeeze(measurement_cov)), cov=True)
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
    EFAC_min = 1e-20
    EFAC_max = 1e20
    EQUAD_min = 1e-40
    EQUAD_max = 1e-10
    priors = bilby.core.prior.PriorDict()
    priors['ratio'] = bilby.core.prior.LogUniform(minimum=r_min, maximum=r_max, name='ratio', latex_label='$\\frac{\\tau_s}{\\tau_c}$')
    priors['tau'] = bilby.core.prior.LogUniform(minimum=tau_min, maximum=tau_max, name='tau', latex_label='$\\frac{\\tau_c + \\tau_s}{\\tau_c\\tau_s}$')
    priors['Qc'] = bilby.core.prior.LogUniform(minimum=Qc_min, maximum=Qc_max, name='Qc', latex_label='$Q_c$')
    priors['Qs'] = bilby.core.prior.LogUniform(minimum=Qs_min, maximum=Qs_max, name='Qs', latex_label='$Q_s$')
    priors['lag'] = bilby.core.prior.Uniform(minimum=lag_min, maximum=lag_max, name='lag', latex_label='$\\Omega_c - \\Omega_s$')
    priors['omgc_dot'] = bilby.core.prior.Uniform(minimum=omgc_dot_min, maximum=omgc_dot_max, name='omgc_dot', latex_label='$\\langle \\dot{\\Omega}_c \\rangle$')
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data_nooverlap[0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['EFAC'] = bilby.core.prior.LogUniform(EFAC_min, EFAC_max, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.LogUniform(EQUAD_min, EQUAD_max, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=params.Nwalks, npoints=params.Npoints,
                               resume=params.resume_run, outdir=params.out_directory,
                               label=f'{params.tag}_nooverlap', check_point_plot=False)

    #Make plots of the sampler results
    nsamples = len(result.posterior.to_numpy()[:, 0])
    samples_nooverlap = np.zeros((nsamples, 9))
    samples_nooverlap[:, 0:8] = result.posterior.to_numpy()[:, 0:8].copy()
    samples_nooverlap[:, 8] = R_out*samples_nooverlap[:, 6]+samples_nooverlap[:, 7]
    samples_nooverlap[:, 0:4] = np.log10(samples_nooverlap[:, 0:4])
    samples_nooverlap[:, 6:9] = np.log10(samples_nooverlap[:, 6:9])
    print("samples_nooverlap[-1] =", samples_nooverlap[-1])
    print("10**samples_nooverlap[-1] =", 10**samples_nooverlap[-1])
    print("R_out =", R_out)
    print("measurement_cov =", measurement_cov)
    print("measurement_cov_new =", measurement_cov*10**samples_nooverlap[-1, 6]+10**samples_nooverlap[-1, 7])
    labels = ['$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', '$\\tau$', '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$',
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$', '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$',
              '$\\langle \\dot\\Omega_c\\rangle$', 'EFAC', 'EQUAD', '$R_{out} \\times EFAC + EQUAD$']
    fig = corner(samples_nooverlap, truths=[np.log10(taus/tauc), np.log10(tau), np.log10(sigmac**2), np.log10(sigmas**2), lag, omgc_dot, np.nan, np.nan, np.log10(R_out)],
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (omgc_dot_min, omgc_dot_max), (np.log10(EFAC_min), np.log10(EFAC_max)),
                          (np.log10(EQUAD_min), np.log10(EQUAD_max)), (np.log10(R_out*EFAC_min+EQUAD_min), np.log10(R_out*EFAC_max+EQUAD_max))],
                 color='b', smooth=True, smooth1d=True, levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_nooverlap_corner.png')
    plt.close()



    #Tempo2 freqs control test (same as the test above but EFAC and EQUAD are fixed).
    print("\nTempo2 freqs control test\n")

    #Set up the Kalman filter
    Nobs_nooverlap = len(times_nooverlap)
    design = np.asarray([1., 0.]).reshape(1, 2)
    measurement_cov = R_nooverlap.copy().reshape((1, 1, Nobs_nooverlap))
    model = TwoComponentModel(times_nooverlap, data_nooverlap.reshape((Nobs_nooverlap, 1)), measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)

    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times_nooverlap, data_nooverlap, 1, w=1/np.sqrt(np.squeeze(measurement_cov)), cov=True)
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
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data_nooverlap[0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['EFAC'] = bilby.core.prior.DeltaFunction(1, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.DeltaFunction(0, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=params.Nwalks, npoints=params.Npoints, 
                               resume=params.resume_run, outdir=params.out_directory,
                               label=f'{params.tag}_nooverlap_control', check_point_plot=False)

    #Make plots of the sampler results
    samples_nooverlap_control = result.posterior.to_numpy()[:, 0:6].copy()
    samples_nooverlap_control[:, 0:4] = np.log10(samples_nooverlap_control[:, 0:4])
    print("samples_nooverlap_control[-1] =", samples_nooverlap_control[-1])
    print("10**samples_nooverlap_control[-1] =", 10**samples_nooverlap_control[-1])
    print("measurement_cov =", measurement_cov)
    labels = ['$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', '$\\tau$', '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$',
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$', '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$',
              '$\\langle \\dot\\Omega_c\\rangle$']
    fig = corner(samples_nooverlap_control, truths=[np.log10(taus/tauc), np.log10(tau), np.log10(sigmac**2), np.log10(sigmas**2), lag, omgc_dot],
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (omgc_dot_min, omgc_dot_max)],
                 color='b', smooth=True, smooth1d=True, levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_nooverlap_control_corner.png')
    plt.close()



    #Tempo freqs (overlapping toa sets)
    print("\nTempo freqs\n")

    #Set up the Kalman filter
    Nobs_overlap = len(times_overlap)
    design = np.asarray([1., 0.]).reshape(1, 2)
    measurement_cov = R_overlap.copy().reshape((1, 1, Nobs_overlap))
    model = TwoComponentModel(times_overlap, data_overlap.reshape((Nobs_overlap, 1)), measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)

    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times_overlap, data_overlap, 1, w=1/np.sqrt(np.squeeze(measurement_cov)), cov=True)
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
    EFAC_min = 1e-20
    EFAC_max = 1e20
    EQUAD_min = 1e-40
    EQUAD_max = 1e-10
    priors = bilby.core.prior.PriorDict()
    priors['ratio'] = bilby.core.prior.LogUniform(minimum=r_min, maximum=r_max, name='ratio', latex_label='$\\frac{\\tau_s}{\\tau_c}$')
    priors['tau'] = bilby.core.prior.LogUniform(minimum=tau_min, maximum=tau_max, name='tau', latex_label='$\\frac{\\tau_c + \\tau_s}{\\tau_c\\tau_s}$')
    priors['Qc'] = bilby.core.prior.LogUniform(minimum=Qc_min, maximum=Qc_max, name='Qc', latex_label='$Q_c$')
    priors['Qs'] = bilby.core.prior.LogUniform(minimum=Qs_min, maximum=Qs_max, name='Qs', latex_label='$Q_s$')
    priors['lag'] = bilby.core.prior.Uniform(minimum=lag_min, maximum=lag_max, name='lag', latex_label='$\\Omega_c - \\Omega_s$')
    priors['omgc_dot'] = bilby.core.prior.Uniform(minimum=omgc_dot_min, maximum=omgc_dot_max, name='omgc_dot', latex_label='$\\langle \\dot{\\Omega}_c \\rangle$')
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data_overlap[0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['EFAC'] = bilby.core.prior.LogUniform(EFAC_min, EFAC_max, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.LogUniform(EQUAD_min, EQUAD_max, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=10, npoints=100,
                               resume=params.resume_run, outdir=params.out_directory, 
                               label=f'{params.tag}_overlap', check_point_plot=False)

    #Make plots of the sampler results
    nsamples = len(result.posterior.to_numpy()[:, 0])
    samples_overlap = np.zeros((nsamples, 9))
    samples_overlap[:, 0:8] = result.posterior.to_numpy()[:, 0:8].copy()
    samples_overlap[:, 8] = R_out*samples_overlap[:, 6]+samples_overlap[:, 7]
    samples_overlap[:, 0:4] = np.log10(samples_overlap[:, 0:4])
    samples_overlap[:, 6:9] = np.log10(samples_overlap[:, 6:9])
    print("samples_overlap[-1] =", samples_overlap[-1])
    print("10**samples_overlap[-1] =", 10**samples_overlap[-1])
    print("R_out =", R_out)
    print("measurement_cov =", measurement_cov)
    print("measurement_cov_new =", measurement_cov*10**samples_overlap[-1, 6]+10**samples_overlap[-1, 7])
    labels = ['$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', '$\\tau$', '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$',
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$', '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$',
              '$\\langle \\dot\\Omega_c\\rangle$', 'EFAC', 'EQUAD', '$R_{out} \\times EFAC + EQUAD$']
    fig = corner(samples_overlap, truths=[np.log10(taus/tauc), np.log10(tau), np.log10(sigmac**2), np.log10(sigmas**2), lag, omgc_dot, np.nan, np.nan, np.log10(R_out)],
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (omgc_dot_min, omgc_dot_max), (np.log10(EFAC_min), np.log10(EFAC_max)),
                          (np.log10(EQUAD_min), np.log10(EQUAD_max)), (np.log10(R_out*EFAC_min+EQUAD_min), np.log10(R_out*EFAC_max+EQUAD_max))],
                 color='b', smooth=True, smooth1d=True, levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_overlap_corner.png')
    plt.close()



    #Tempo freqs control test (same as the test above but EFAC and EQUAD are fixed).
    print("\nTempo freqs control test\n")

    #Set up the Kalman filter
    Nobs_overlap = len(times_overlap)
    design = np.asarray([1., 0.]).reshape(1, 2)
    measurement_cov = R_overlap.copy().reshape((1, 1, Nobs_overlap))
    model = TwoComponentModel(times_overlap, data_overlap.reshape((Nobs_overlap, 1)), measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)

    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times_overlap, data_overlap, 1, w=1/np.sqrt(np.squeeze(measurement_cov)), cov=True)
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
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data_overlap[0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['EFAC'] = bilby.core.prior.DeltaFunction(1, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.DeltaFunction(0, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=params.Nwalks, npoints=params.Npoints, 
                               resume=params.resume_run, outdir=params.out_directory, 
                               label=f'{params.tag}_overlap_control', check_point_plot=False)

    #Make plots of the sampler results
    samples_overlap_control = result.posterior.to_numpy()[:, 0:6].copy()
    samples_overlap_control[:, 0:4] = np.log10(samples_overlap_control[:, 0:4])
    print("samples_overlap_control[-1] =", samples_overlap_control[-1])
    print("10**samples_overlap_control[-1] =", 10**samples_overlap_control[-1])
    print("measurement_cov =", measurement_cov)
    labels = ['$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', '$\\tau$', '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$',
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$', '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$',
              '$\\langle \\dot\\Omega_c\\rangle$']
    fig = corner(samples_overlap_control, truths=[np.log10(taus/tauc), np.log10(tau), np.log10(sigmac**2), np.log10(sigmas**2), lag, omgc_dot],
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (omgc_dot_min, omgc_dot_max)],
                 color='b', smooth=True, smooth1d=True, levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_overlap_control_corner.png')
    plt.close()



    #True freqs
    print("\nTrue freqs\n")

    #Set up the Kalman filter
    Nobs_true = len(times_true)
    design = np.asarray([1., 0.]).reshape(1, 2)
    measurement_cov = R_true.copy().reshape((1, 1, Nobs_true))
    model = TwoComponentModel(times_true, data_true.reshape((Nobs_true, 1)), measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)

    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times_true, data_true, 1, w=1/np.sqrt(np.squeeze(measurement_cov)), cov=True)
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
    EFAC_min = 1e-20
    EFAC_max = 1e20
    EQUAD_min = 1e-40
    EQUAD_max = 1e-10
    priors = bilby.core.prior.PriorDict()
    priors['ratio'] = bilby.core.prior.LogUniform(minimum=r_min, maximum=r_max, name='ratio', latex_label='$\\frac{\\tau_s}{\\tau_c}$')
    priors['tau'] = bilby.core.prior.LogUniform(minimum=tau_min, maximum=tau_max, name='tau', latex_label='$\\frac{\\tau_c + \\tau_s}{\\tau_c\\tau_s}$')
    priors['Qc'] = bilby.core.prior.LogUniform(minimum=Qc_min, maximum=Qc_max, name='Qc', latex_label='$Q_c$')
    priors['Qs'] = bilby.core.prior.LogUniform(minimum=Qs_min, maximum=Qs_max, name='Qs', latex_label='$Q_s$')
    priors['lag'] = bilby.core.prior.Uniform(minimum=lag_min, maximum=lag_max, name='lag', latex_label='$\\Omega_c - \\Omega_s$')
    priors['omgc_dot'] = bilby.core.prior.Uniform(minimum=omgc_dot_min, maximum=omgc_dot_max, name='omgc_dot', latex_label='$\\langle \\dot{\\Omega}_c \\rangle$')
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data_true[0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['EFAC'] = bilby.core.prior.LogUniform(EFAC_min, EFAC_max, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.LogUniform(EQUAD_min, EQUAD_max, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=params.Nwalks, npoints=params.Npoints, 
                               resume=params.resume_run, outdir=params.out_directory,
                               label=f'{params.tag}_true', check_point_plot=False)

    #Make plots of the sampler results
    nsamples = len(result.posterior.to_numpy()[:, 0])
    samples_true = np.zeros((nsamples, 9))
    samples_true[:, 0:8] = result.posterior.to_numpy()[:, 0:8].copy()
    samples_true[:, 8] = R_out*samples_true[:, 6]+samples_true[:, 7]
    samples_true[:, 0:4] = np.log10(samples_true[:, 0:4])
    samples_true[:, 6:9] = np.log10(samples_true[:, 6:9])
    print("samples_true[-1] =", samples_true[-1])
    print("10**samples_true[-1] =", 10**samples_true[-1])
    print("R_out =", R_out)
    print("measurement_cov =", measurement_cov)
    print("measurement_cov_new =", measurement_cov*10**samples_true[-1, 6]+10**samples_true[-1, 7])
    labels = ['$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', '$\\tau$', '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$',
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$', '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$',
              '$\\langle \\dot\\Omega_c\\rangle$', 'EFAC', 'EQUAD', '$R_{out} \\times EFAC + EQUAD$']
    fig = corner(samples_true, truths=[np.log10(taus/tauc), np.log10(tau), np.log10(sigmac**2), np.log10(sigmas**2), lag, omgc_dot, np.nan, np.nan, np.log10(R_out)],
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (omgc_dot_min, omgc_dot_max), (np.log10(EFAC_min), np.log10(EFAC_max)),
                          (np.log10(EQUAD_min), np.log10(EQUAD_max)), (np.log10(R_out*EFAC_min+EQUAD_min), np.log10(R_out*EFAC_max+EQUAD_max))],
                 color='b', smooth=True, smooth1d=True, levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_true_corner.png')
    plt.close()



    #True freqs control test (same as above but no EFAC or EQUAD).
    print("\nTrue freqs control test\n")

    #This section of code uses the peak R value found in the EFAC test on true data
    #as our fixed R value. The original R_out value was the mean of the R values given
    #by tempo for the tempo fits.
    R_out_new = R_out*10**samples_true[-1, 6]+10**samples_true[-1, 7]
    print("R_out_new =", R_out_new)
    R_true_new = np.ones(len(times_true))*R_out_new

    #Set up the Kalman filter
    Nobs_true = len(times_true)
    design = np.asarray([1., 0.]).reshape(1, 2)
    #measurement_cov = R_true.copy().reshape((1, 1, Nobs_true))
    measurement_cov = R_true_new.copy().reshape((1, 1, Nobs_true))
    model = TwoComponentModel(times_true, data_true.reshape((Nobs_true, 1)), measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)

    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times_true, data_true, 1, w=1/np.sqrt(np.squeeze(measurement_cov)), cov=True)
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
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data_true[0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['EFAC'] = bilby.core.prior.DeltaFunction(1, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.DeltaFunction(0, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=params.Nwalks, npoints=params.Npoints, 
                               resume=params.resume_run, outdir=params.out_directory,
                               label=f'{params.tag}_true_control', check_point_plot=False)

    #Make plots of the sampler results
    samples_true_control = result.posterior.to_numpy()[:, 0:6].copy()
    samples_true_control[:, 0:4] = np.log10(samples_true_control[:, 0:4])
    print("samples_true_control[-1] =", samples_true_control[-1])
    print("10**samples_true_control[-1] =", 10**samples_true_control[-1])
    print("measurement_cov =", measurement_cov)
    labels = ['$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$', '$\\tau$', '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$',
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$', '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$',
              '$\\langle \\dot\\Omega_c\\rangle$']
    fig = corner(samples_true_control, truths=[np.log10(taus/tauc), np.log10(tau), np.log10(sigmac**2), np.log10(sigmas**2), lag, omgc_dot],
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (omgc_dot_min, omgc_dot_max)],
                 color='b', smooth=True, smooth1d=True, levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_true_control_corner.png')
    plt.close()

    #Plot data
    plt.figure(1, figsize=(40,20))

    plt.subplot(6,1,1)
    plt.plot(times_true, data_true-omgc_0-omgc_dot*(times_true-pets0*86400), marker='.', linestyle=None, color="k")
    plt.errorbar(times_nooverlap, data_nooverlap-omgc_0-omgc_dot*(times_nooverlap-pets0*86400), yerr=np.sqrt(R_nooverlap), marker='.', linestyle=None, color="r")
    plt.xlabel("times")
    plt.ylabel("$\\Omega_c$ ", fontsize=17)
    plt.title("True and Tempo2 $\\Omega_c$ residuals")

    plt.subplot(6,1,2)
    plt.plot(times_true, data_true-omgc_0-omgc_dot*(times_true-pets0*86400), marker='.', linestyle=None, color="k")
    plt.errorbar(times_nooverlap, data_nooverlap-omgc_0-omgc_dot*(times_nooverlap-pets0*86400), yerr=np.sqrt(R_nooverlap*10**samples_nooverlap[-1,6]+10**samples_nooverlap[-1,7]), marker='.', linestyle=None, color="r")
    plt.xlabel("times")
    plt.ylabel("$\\Omega_c$", fontsize=17)
    plt.title("True and Tempo2 $\\Omega_c$ residuals with EFAC")

    plt.subplot(6,1,3)
    plt.plot(times_true, data_true-omgc_0-omgc_dot*(times_true-pets0*86400), marker='.', linestyle=None, color="k")
    plt.errorbar(times_overlap, data_overlap-omgc_0-omgc_dot*(times_overlap-pets0*86400), yerr=np.sqrt(R_overlap), marker='.', linestyle=None, color="b")
    plt.xlabel("times")
    plt.ylabel("$\\Omega_c$", fontsize=17)
    plt.title("True and Tempo $\\Omega_c$ residuals")

    plt.subplot(6,1,4)
    plt.plot(times_true, data_true-omgc_0-omgc_dot*(times_true-pets0*86400), marker='.', linestyle=None, color="k")
    plt.errorbar(times_overlap, data_overlap-omgc_0-omgc_dot*(times_overlap-pets0*86400), yerr=np.sqrt(R_overlap*10**samples_overlap[-1,6]+10**samples_overlap[-1,7]), marker='.', linestyle=None, color="b")
    plt.xlabel("times")
    plt.ylabel("$\\Omega_c$", fontsize=17)
    plt.title("True and Tempo $\\Omega_c$ residuals with EFAC")

    plt.subplot(6,1,5)
    plt.errorbar(times_true, data_true-omgc_0-omgc_dot*(times_true-pets0*86400), yerr=np.sqrt(R_true*10**samples_true[-1,6]+10**samples_true[-1,7]), marker='.', linestyle=None, color="k")
    plt.xlabel("times")
    plt.ylabel("$\\Omega_c$", fontsize=17)
    plt.title("True $\\Omega_c$ residuals with EFAC")

    plt.subplot(6,1,6)
    plt.errorbar(times_true, data_true-omgc_0-omgc_dot*(times_true-pets0*86400), yerr=np.sqrt(R_true), marker='.', linestyle=None, color="k")
    plt.xlabel("times")
    plt.ylabel("$\\Omega_c$", fontsize=17)
    plt.title("True $\\Omega_c$ residuals")

    plt.tight_layout()
    plt.savefig(f"{params.out_directory}/dataplots.png")
    plt.close()



if __name__=="__main__":
    #Parse command line arguments
    parser = argparse.ArgumentParser()
    #Simulation parameters
    parser.add_argument("--Nobs", help="number of observations", type=int, required=True)
    parser.add_argument("--Tdays", help="number of days observations are spread over", type=float, required=True)
    parser.add_argument("--T_error_in", help="measurement error of ToAs used to create the data", default=1e-16, type=float)
    #Frequency fit parameters
    parser.add_argument("--Tfit", help="timespan for fitting a single frequency using tempo (days)", default=10, type=float)
    parser.add_argument("--Nfit_min", help="minimum number of ToAs per fit", default=3, type=int)
    #Kalman filtering parameters
    #parser.add_argument("--R_out", help="measurement covariance of frequencies used in Kalman filter for recovery", default=1e-30, type=float)
    #mcmc information
    parser.add_argument("--Nwalks", default=10, type=int, help="number of walks for mcmc")
    parser.add_argument("--Npoints", default=100, type=int, help="number of points for mcmc")
    parser.add_argument("--resume_run", help="resume mcmc run from where it left off", action="store_true", default=False)
    #Save information
    parser.add_argument("--out_directory", help="output directory", default="./outdir/", type=str)
    parser.add_argument("--tag", help="tag to include in saving information", default=None)
    args = parser.parse_args()
    run(args)


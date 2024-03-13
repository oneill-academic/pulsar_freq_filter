#! /usr/bin/env python
print("Beginning program")
import sys
from models import TwoComponentModel, param_map
from sample import KalmanLikelihood
import bilby
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from corner import corner
import argparse
from pathlib import Path
plt.rc('text', usetex=False)
print("Importing modules completed")

def run(params):
    outdirectory = Path(params.out_directory)
    outdirectory.mkdir(parents=True, exist_ok=True)

    #Load simulated tempo fitted freqs and times and errors
    data1 = np.loadtxt(params.freqfile)
    times_tempo = data1[:, 0]*86400
    data_tempo = data1[:, 1]*2*np.pi
    R_tempo = (data1[:, 2]*2*np.pi)**2

    times_tempo = times_tempo.astype(np.float64)
    data_tempo = data_tempo.astype(np.float64)
    R_tempo = R_tempo.astype(np.float64)

    #Set up the Kalman filter
    Nobs_tempo = len(times_tempo)
    design = np.asarray([1., 0.]).reshape(1, 2)
    measurement_cov = R_tempo.copy().reshape((1, 1, Nobs_tempo))
    model = TwoComponentModel(times_tempo, data_tempo.reshape((Nobs_tempo, 1)), measurement_cov, design, param_map)
    likelihood = KalmanLikelihood(model)

    print("measurement_cov =", measurement_cov)

    #Get an estimate for omgc_dot by fitting a linear trend
    p, V = np.polyfit(times_tempo, data_tempo, 1, w=1/np.sqrt(np.squeeze(measurement_cov)), cov=True)
    omgd_low = p[0] - np.sqrt(V[0,0]) * 1000
    omgd_high = p[0] + np.sqrt(V[0,0]) * 1000

    #Set priors
    r_min = 1e-2
    r_max = 1e2
    tau_min = 1e5
    tau_max = 1e8
    Qc_min = 1e-30
    Qc_max = 1e-16
    Qs_min = 1e-30
    Qs_max = 1e-16
    lag_min = -1e-3
    lag_max = 1e-3
    omgc_dot_min = omgd_low
    omgc_dot_max = omgd_high
    priors = bilby.core.prior.PriorDict()
    priors['ratio'] = bilby.core.prior.LogUniform(minimum=r_min, maximum=r_max, name='ratio', latex_label='$\\frac{\\tau_s}{\\tau_c}$')
    priors['tau'] = bilby.core.prior.LogUniform(minimum=tau_min, maximum=tau_max, name='tau', latex_label='$\\frac{\\tau_c + \\tau_s}{\\tau_c\\tau_s}$')
    priors['Qc'] = bilby.core.prior.LogUniform(minimum=Qc_min, maximum=Qc_max, name='Qc', latex_label='$Q_c$')
    priors['Qs'] = bilby.core.prior.LogUniform(minimum=Qs_min, maximum=Qs_max, name='Qs', latex_label='$Q_s$')
    priors['lag'] = bilby.core.prior.Uniform(minimum=lag_min, maximum=lag_max, name='lag', latex_label='$\\Omega_c - \\Omega_s$')
    priors['omgc_dot'] = bilby.core.prior.Uniform(minimum=omgc_dot_min, maximum=omgc_dot_max, name='omgc_dot', latex_label='$\\langle \\dot{\\Omega}_c \\rangle$')
    priors['omgc_0'] = bilby.core.prior.DeltaFunction(data_tempo[0], name='omgc_0', latex_label='$\\Omega_{\\rm{c}, 0}$')
    priors['EFAC'] = bilby.core.prior.DeltaFunction(1, name='EFAC', latex_label='EFAC')
    priors['EQUAD'] = bilby.core.prior.DeltaFunction(0, name='EQUAD', latex_label='EQUAD')

    #Run the sampler
    result = bilby.run_sampler(likelihood, priors, sampler='dynesty',
                               sample='rwalk', walks=params.Nwalks, npoints=params.Npoints,
                               resume=params.resume_run, outdir=params.out_directory, 
                               label=f'{params.tag}_tempo', check_point_plot=False)

    print("result.posterior.to_numpy()[-1] =", result.posterior.to_numpy()[-1])
    samples_tempo = result.posterior.to_numpy()[:, :6].copy()
    samples_tempo[:, 0:4] = np.log10(result.posterior.to_numpy()[:, 0:4])
    nsamples = len(samples_tempo[:, 0])
    print("samples_tempo[-1] =", samples_tempo[-1])
    print("10**samples_tempo[-1] =", 10**samples_tempo[-1])
    print("samples_tempo =", samples_tempo)
    print("measurement_cov =", measurement_cov)
    print("measurement_cov_new =", measurement_cov*10**samples_tempo[-1,-2]+10**samples_tempo[-1,-1])

    #Plot the output of the sampler as a corner plot.
    labels = ['$r$', '$\\tau$ $[\\rm{s}]$', 
              '$\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2$\n$[\\rm{rad^2~s^{-3}}]$',
              '$\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2$\n$[\\rm{rad^2~s^{-3}}]$', 
              '$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$\n$[\\rm{rad~s^{-1}}]$',
              '$\\langle \\dot\\Omega_c\\rangle$ $[\\rm{rad~s^{-2}}]$']
    fig = corner(samples_tempo, bins = 50,
                 range = [(np.log10(r_min), np.log10(r_max)), (np.log10(tau_min), np.log10(tau_max)),
                          (np.log10(Qc_min), np.log10(Qc_max)), (np.log10(Qs_min), np.log10(Qs_max)),
                          (lag_min, lag_max), (-2.45e-12, -2.445e-12)],
                 color='b', levels=[1-np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9/2)])
    axarr = np.reshape(fig.axes, (len(labels), len(labels)))
    for ii, label in enumerate(labels):
        axarr[ii,ii].set_title(label)
    plt.savefig(f'{params.out_directory}/{params.tag}_tempo_corner_50bins.png')
    plt.close()

    log_r_samples = samples_tempo[:, 0]
    log_tau_samples = samples_tempo[:, 1]
    r_samples = 10**log_r_samples
    tau_samples = 10**log_tau_samples
    tauc_samples = (1+1/r_samples)*tau_samples
    taus_samples = (1+r_samples)*tau_samples
    log_tauc_samples = np.log10(tauc_samples)
    log_taus_samples = np.log10(taus_samples)
    log_Qc_samples = samples_tempo[:, 2]
    log_Qs_samples = samples_tempo[:, 3]
    lag_samples = samples_tempo[:, 4]
    omgc_dot_samples = samples_tempo[:, 5]

    #Replot parameters as histograms
    plt.figure(1, figsize=(20,12))

    plt.subplot(4,2,1)
    plt.hist(log_tau_samples, 50, density=True, facecolor="tab:orange", alpha=0.6)
    plt.xlabel("$\\tau_{\\rm{s}}/\\tau_{\\rm{c}}$")

    plt.subplot(4,2,2)
    plt.hist(log_r_samples, 50, density=True, facecolor="tab:red", alpha=0.6)
    plt.xlabel("$\\tau$")

    plt.subplot(4,2,3)
    plt.hist(log_tauc_samples, 50, density=True, facecolor="tab:blue", alpha=0.6)
    plt.xlabel("$\\tau_c$")

    plt.subplot(4,2,4)
    plt.hist(log_taus_samples, 50, density=True, facecolor="tab:green", alpha=0.6)
    plt.xlabel("$\\tau_s$")

    plt.subplot(4,2,5)
    plt.hist(log_Qc_samples, 50, density=True, facecolor="tab:blue", alpha=0.6)
    plt.xlabel("$Q_c$")

    plt.subplot(4,2,6)
    plt.hist(log_Qs_samples, 50, density=True, facecolor="tab:green", alpha=0.6)
    plt.xlabel("$Q_s$")

    plt.subplot(4,2,7)
    plt.hist(lag_samples, 50, density=True, facecolor="k", alpha=0.4)
    plt.xlabel("$\\langle\\Omega_{\\rm{c}} - \\Omega_{\\rm{s}}\\rangle$")

    plt.subplot(4,2,8)
    plt.hist(omgc_dot_samples, 50, density=True, facecolor="k", alpha=0.8)
    plt.xlabel("$\\langle \\dot\\Omega_c\\rangle$")

    plt.title("Recovered parameters")
    plt.tight_layout()
    plt.savefig(f"{params.out_directory}/{params.tag}_tempo_param_hists.png")
    plt.close()



    #Replot only the main four parameters.
    samples_tempo2 = np.zeros((len(samples_tempo[:, 0]), 4))
    samples_tempo2[:, 0] = samples_tempo[:, 1].copy()
    samples_tempo2[:, 1] = samples_tempo[:, 2].copy()
    samples_tempo2[:, 2] = samples_tempo[:, 3].copy()
    samples_tempo2[:, 3] = samples_tempo[:, 5].copy()

    myfontsize=18
    facecolor="tab:blue"
    fig, axs = plt.subplots(1, 4, figsize=(22,6))
    axs[0].hist(samples_tempo2[:, 0], 50, range=[5, 8], facecolor=facecolor, alpha=0.6, edgecolor='b')
    axs[0].set_xlabel("$\log_{10}(\\tau)$ $[\\rm{s}]$", fontsize=myfontsize)
    axs[0].set_ylabel("Count", fontsize=myfontsize)
    axs[0].set_xlim([5, 8])
    axs[0].xaxis.set_tick_params(labelsize=myfontsize)
    axs[0].yaxis.set_tick_params(labelsize=myfontsize, rotation=45)
    axs[0].xaxis.get_offset_text().set_fontsize(myfontsize)
    axs[0].yaxis.get_offset_text().set_fontsize(myfontsize)

    axs[1].hist(samples_tempo2[:, 1], 50, range=[-30, -16], facecolor=facecolor, alpha=0.6, edgecolor='b')
    axs[1].set_xlabel("$\log_{10}(\\sigma_{\\rm{c}}^2/I_{\\rm{c}}^2)$ $[\\rm{rad^2~s^{-3}}]$", fontsize=myfontsize)
    axs[1].set_ylabel("Count", fontsize=myfontsize)
    axs[1].set_xlim([-30, -16])
    axs[1].xaxis.set_tick_params(labelsize=myfontsize)
    axs[1].yaxis.set_tick_params(labelsize=myfontsize, rotation=45)
    axs[1].xaxis.get_offset_text().set_fontsize(myfontsize)
    axs[1].yaxis.get_offset_text().set_fontsize(myfontsize)

    axs[2].hist(samples_tempo2[:, 2], 50, range=[-30, -16], facecolor=facecolor, alpha=0.6, edgecolor='b')
    axs[2].set_xlabel("$\log_{10}(\\sigma_{\\rm{s}}^2/I_{\\rm{s}}^2)$ $[\\rm{rad^2~s^{-3}}]$", fontsize=myfontsize)
    axs[2].set_ylabel("Count", fontsize=myfontsize)
    axs[2].set_xlim([-30, -16])
    axs[2].xaxis.set_tick_params(labelsize=myfontsize)
    axs[2].yaxis.set_tick_params(labelsize=myfontsize, rotation=45)
    axs[2].xaxis.get_offset_text().set_fontsize(myfontsize)
    axs[2].yaxis.get_offset_text().set_fontsize(myfontsize)

    axs[3].hist(samples_tempo2[:, 3], 50, range=[-2.45e-12, -2.445e-12], facecolor=facecolor, alpha=0.6, edgecolor='b')
    axs[3].set_xlabel("$\\langle \\dot\\Omega_c\\rangle$ $[\\rm{rad~s^{-2}}]$", fontsize=myfontsize)
    axs[3].set_ylabel("Count", fontsize=myfontsize)
    axs[3].set_xlim([-2.45e-12, -2.445e-12])
    axs[3].xaxis.set_tick_params(labelsize=myfontsize, rotation=45)
    axs[3].yaxis.set_tick_params(labelsize=myfontsize, rotation=45)
    axs[3].xaxis.get_offset_text().set_fontsize(myfontsize)
    axs[3].yaxis.get_offset_text().set_fontsize(myfontsize)

    plt.tight_layout()
    plt.savefig(f"{params.out_directory}/hist_tempo_4params_filled_{params.tag}.png", bbox_inches='tight')
    plt.close()



    print("\n")
    #Print the natural log of the Bayesian evidence and its error.
    print("log_evidence =", result.log_evidence)
    print("log_evidence_err =", result.log_evidence_err)
    print("log10_evidence =", result.log_evidence/np.log(10))
    print("log10_evidence_err =", result.log_evidence_err/np.log(10))
    print("\n")



    #Find sample with the highest likelihood value.
    log_r_peak = samples_tempo[-1, 0]
    log_tau_peak = samples_tempo[-1, 1]
    log_Qc_peak = samples_tempo[-1, 2]
    log_Qs_peak = samples_tempo[-1, 3]
    r_peak = 10**samples_tempo[-1, 0]
    tau_peak = 10**samples_tempo[-1, 1]
    tauc_peak = tau_peak*(1+1/r_peak)
    taus_peak = tau_peak*(1+r_peak)
    log_tauc_peak = np.log10(tauc_peak)
    log_taus_peak = np.log10(taus_peak)
    Qc_peak = 10**samples_tempo[-1, 2]
    Qs_peak = 10**samples_tempo[-1, 3]
    lag_peak = samples_tempo[-1, 4]
    omgc_dot_peak = samples_tempo[-1, 5]
    print("log_r_peak =", log_r_peak)
    print("log_tau_peak =", log_tau_peak)
    print("log_Qc_peak =", log_Qc_peak)
    print("log_Qs_peak =", log_Qs_peak)
    print("r_peak =", r_peak)
    print("tau_peak =", tau_peak)
    print("Qc_peak =", Qc_peak)
    print("Qs_peak =", Qs_peak)
    print("lag_peak =", lag_peak)
    print("omgc_dot_peak =", omgc_dot_peak)
    print("tauc_peak =", tauc_peak)
    print("taus_peak =", taus_peak)
    print("log_tauc_peak =", log_tauc_peak)
    print("log_taus_peak =", log_taus_peak)
    print("\n")

    #Calculate percentiles using the samples generated by the sampler.
    r5, r50, r95 = np.percentile(log_r_samples, [5,50,95])
    tau5, tau50, tau95 = np.percentile(log_tau_samples, [5,50,95])
    tauc5, tauc50, tauc95 = np.percentile(log_tauc_samples, [5,50,95])
    taus5, taus50, taus95 = np.percentile(log_taus_samples, [5,50,95])
    Qc5, Qc50, Qc95 = np.percentile(log_Qc_samples, [5,50,95])
    Qs5, Qs50, Qs95 = np.percentile(log_Qs_samples, [5,50,95])
    lag5, lag50, lag95 = np.percentile(lag_samples, [5,50,95])
    omgc_dot5, omgc_dot50, omgc_dot95 = np.percentile(omgc_dot_samples, [5,50,95])
    print("r5 =", r5)
    print("r95 =", r95)
    print("tau5 =", tau5)
    print("tau95 =", tau95)
    print("Qc5 =", Qc5)
    print("Qc95 =", Qc95)
    print("Qs5 =", Qs5)
    print("Qs95 =", Qs95)
    print("lag5 =", lag5)
    print("lag95 =", lag95)
    print("omgc_dot5 =", omgc_dot5)
    print("omgc_dot95 =", omgc_dot95)

    #Calculate 90% confidence intervals. Be careful if the peak is outside this range.
    log_r_width = r95-r5
    log_tau_width = tau95-tau5
    log_tauc_width = tauc95-tauc5
    log_taus_width = taus95-taus5
    log_Qc_width = Qc95-Qc5
    log_Qs_width = Qs95-Qs5
    lag_width = lag95-lag5
    omgc_dot_width = omgc_dot95-omgc_dot5
    print("log_r_width =", log_r_width)    
    print("log_tau_width =", log_tau_width)
    print("log_tauc_width =", log_tauc_width)
    print("log_taus_width =", log_taus_width)
    print("log_Qc_width =", log_Qc_width)
    print("log_Qs_width =", log_Qs_width)
    print("lag_width =", lag_width)
    print("omgc_dot_width =", omgc_dot_width)
    print("\n")



    #Calculate properties of the samples using the bins of the histogram.
    Nbins = 50

    log_r_min = np.log10(r_min)
    log_r_max = np.log10(r_max)
    log_tau_min = np.log10(tau_min)
    log_tau_max = np.log10(tau_max)
    log_Qc_min = np.log10(Qc_min)
    log_Qc_max = np.log10(Qc_max)
    log_Qs_min = np.log10(Qs_min)
    log_Qs_max = np.log10(Qs_max)
    log_tauc_min = np.min(log_tauc_samples)
    log_tauc_max = np.max(log_tauc_samples)
    log_taus_min = np.min(log_taus_samples)
    log_taus_max = np.max(log_taus_samples)

    #Make a list of the bin positions
    log_r_bins = np.linspace(log_r_min, log_r_max, Nbins+1)
    log_tau_bins = np.linspace(log_tau_min, log_tau_max, Nbins+1)
    log_Qc_bins = np.linspace(log_Qc_min, log_Qc_max, Nbins+1)
    log_Qs_bins = np.linspace(log_Qs_min, log_Qs_max, Nbins+1)
    lag_bins = np.linspace(lag_min, lag_max, Nbins+1)
    omgc_dot_bins = np.linspace(omgc_dot_min, omgc_dot_max, Nbins+1)
    log_tauc_bins = np.linspace(log_tauc_min, log_tauc_max, Nbins+1)
    log_taus_bins = np.linspace(log_taus_min, log_taus_max, Nbins+1)

    #These will contain a manual count of how many points are in each bin
    log_r_bin_counts = np.zeros(Nbins)
    log_tau_bin_counts = np.zeros(Nbins)
    log_Qc_bin_counts = np.zeros(Nbins)
    log_Qs_bin_counts = np.zeros(Nbins)
    lag_bin_counts = np.zeros(Nbins)
    omgc_dot_bin_counts = np.zeros(Nbins)
    log_tauc_bin_counts = np.zeros(Nbins)
    log_taus_bin_counts = np.zeros(Nbins)

    #Count how many samples are in each bin
    for x in log_r_samples:
        for i in range(Nbins):
            if log_r_bins[i] < x < log_r_bins[i+1]:
                log_r_bin_counts[i] += 1

    for x in log_tau_samples:
        for i in range(Nbins):
            if log_tau_bins[i] < x < log_tau_bins[i+1]:
                log_tau_bin_counts[i] += 1

    for x in log_Qc_samples:
        for i in range(Nbins):
            if log_Qc_bins[i] < x < log_Qc_bins[i+1]:
                log_Qc_bin_counts[i] += 1

    for x in log_Qs_samples:
        for i in range(Nbins):
            if log_Qs_bins[i] < x < log_Qs_bins[i+1]:
                log_Qs_bin_counts[i] += 1

    for x in lag_samples:
        for i in range(Nbins):
            if lag_bins[i] < x < lag_bins[i+1]:
                lag_bin_counts[i] += 1

    for x in omgc_dot_samples:
        for i in range(Nbins):
            if omgc_dot_bins[i] < x < omgc_dot_bins[i+1]:
                omgc_dot_bin_counts[i] += 1

    for x in log_tauc_samples:
        for i in range(Nbins):
            if log_tauc_bins[i] < x < log_tauc_bins[i+1]:
                log_tauc_bin_counts[i] += 1

    for x in log_taus_samples:
        for i in range(Nbins):
            if log_taus_bins[i] < x < log_taus_bins[i+1]:
                log_taus_bin_counts[i] += 1

    #Calculate the full width at half maximum (FWHM).
    #Find the highest count in each histogram
    max_r_count = max(log_r_bin_counts)
    max_tau_count = max(log_tau_bin_counts)
    max_Qc_count = max(log_Qc_bin_counts)
    max_Qs_count = max(log_Qs_bin_counts)
    max_lag_count = max(lag_bin_counts)
    max_omgc_dot_count = max(omgc_dot_bin_counts)
    max_tauc_count = max(log_tauc_bin_counts)
    max_taus_count = max(log_taus_bin_counts)

    #Work out which bins are above or equal to half the maximum of the histogram
    #Return the parameter values of the lowest and highest bins and the difference.
    #This FWHM calculation should only be applied if there is only one peak above half of the maximum.
    greater_than_HM = []
    for i in range(Nbins):
        if log_r_bin_counts[i] >= max_r_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("log_r_bins[lower_limit] =", log_r_bins[lower_limit])
    print("log_r_bins[upper_limit] =", log_r_bins[upper_limit])
    print("FWHM_r =", log_r_bins[upper_limit] - log_r_bins[lower_limit])

    greater_than_HM = []
    for i in range(Nbins):
        if log_tau_bin_counts[i] >= max_tau_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("log_tau_bins[lower_limit] =", log_tau_bins[lower_limit])
    print("log_tau_bins[upper_limit] =", log_tau_bins[upper_limit])
    print("FWHM_tau =", log_tau_bins[upper_limit] - log_tau_bins[lower_limit])

    greater_than_HM = []
    for i in range(Nbins):
        if log_Qc_bin_counts[i] >= max_Qc_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("log_Qc_bins[lower_limit] =", log_Qc_bins[lower_limit])
    print("log_Qc_bins[upper_limit] =", log_Qc_bins[upper_limit])
    print("FWHM_Qc =", log_Qc_bins[upper_limit] - log_Qc_bins[lower_limit])

    greater_than_HM = []
    for i in range(Nbins):
        if log_Qs_bin_counts[i] >= max_Qs_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("log_Qs_bins[lower_limit] =", log_Qs_bins[lower_limit])
    print("log_Qs_bins[upper_limit] =", log_Qs_bins[upper_limit])
    print("FWHM_Qs =", log_Qs_bins[upper_limit] - log_Qs_bins[lower_limit])

    greater_than_HM = []
    for i in range(Nbins):
        if lag_bin_counts[i] >= max_lag_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("lag_bins[lower_limit] =", lag_bins[lower_limit])
    print("lag_bins[upper_limit] =", lag_bins[upper_limit])
    print("FWHM_lag =", lag_bins[upper_limit] - lag_bins[lower_limit])

    greater_than_HM = []
    for i in range(Nbins):
        if omgc_dot_bin_counts[i] >= max_omgc_dot_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("omgc_dot_bins[lower_limit] =", omgc_dot_bins[lower_limit])
    print("omgc_dot_bins[upper_limit] =", omgc_dot_bins[upper_limit])
    print("FWHM_omgc_dot =", omgc_dot_bins[upper_limit] - omgc_dot_bins[lower_limit])

    greater_than_HM = []
    for i in range(Nbins):
        if log_tauc_bin_counts[i] >= max_tauc_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("log_tauc_bins[lower_limit] =", log_tauc_bins[lower_limit])
    print("log_tauc_bins[upper_limit] =", log_tauc_bins[upper_limit])
    print("FWHM_tauc =", log_tauc_bins[upper_limit] - log_tauc_bins[lower_limit])

    greater_than_HM = []
    for i in range(Nbins):
        if log_taus_bin_counts[i] >= max_taus_count/2.0:
            greater_than_HM.append(i)
    lower_limit = min(greater_than_HM)
    upper_limit = max(greater_than_HM)+1
    print("log_taus_bins[lower_limit] =", log_taus_bins[lower_limit])
    print("log_taus_bins[upper_limit] =", log_taus_bins[upper_limit])
    print("FWHM_taus =", log_taus_bins[upper_limit] - log_taus_bins[lower_limit])
    print("\n")

    #Find the location of the maximum of the histogram.
    log_r_i = np.argmax(log_r_bin_counts)
    log_tau_i = np.argmax(log_tau_bin_counts)
    log_Qc_i = np.argmax(log_Qc_bin_counts)
    log_Qs_i = np.argmax(log_Qs_bin_counts)
    lag_i = np.argmax(lag_bin_counts)
    omgc_dot_i = np.argmax(omgc_dot_bin_counts)
    log_tauc_i = np.argmax(log_tauc_bin_counts)
    log_taus_i = np.argmax(log_taus_bin_counts)

    log_r_max = (log_r_bins[log_r_i]+log_r_bins[log_r_i+1])/2
    log_tau_max = (log_tau_bins[log_tau_i]+log_tau_bins[log_tau_i+1])/2
    log_Qc_max = (log_Qc_bins[log_Qc_i]+log_Qc_bins[log_Qc_i+1])/2
    log_Qs_max = (log_Qs_bins[log_Qs_i]+log_Qs_bins[log_Qs_i+1])/2
    lag_max = (lag_bins[lag_i]+lag_bins[lag_i+1])/2
    omgc_dot_max = (omgc_dot_bins[omgc_dot_i]+omgc_dot_bins[omgc_dot_i+1])/2
    log_tauc_max = (log_tauc_bins[log_tauc_i]+log_tauc_bins[log_tauc_i+1])/2
    log_taus_max = (log_taus_bins[log_taus_i]+log_taus_bins[log_taus_i+1])/2

    #Upper and lower limits of bin with highest count
    print("log_r_bins[log_r_i] =", log_r_bins[log_r_i])
    print("log_r_bins[log_r_i+1] =", log_r_bins[log_r_i+1])
    print("log_tau_bins[log_tau_i] =", log_tau_bins[log_tau_i])
    print("log_tau_bins[log_tau_i+1] =", log_tau_bins[log_tau_i+1])
    print("log_Qc_bins[log_Qc_i] =", log_Qc_bins[log_Qc_i])
    print("log_Qc_bins[log_Qc_i+1] =", log_Qc_bins[log_Qc_i+1])
    print("log_Qs_bins[log_Qs_i] =", log_Qs_bins[log_Qs_i])
    print("log_Qs_bins[log_Qs_i+1] =", log_Qs_bins[log_Qs_i+1])
    print("lag_bins[lag_i] =", lag_bins[lag_i])
    print("lag_bins[lag_i+1] =", lag_bins[lag_i+1])
    print("omgc_dot_bins[omgc_dot_i] =", omgc_dot_bins[omgc_dot_i])
    print("omgc_dot_bins[omgc_dot_i+1] =", omgc_dot_bins[omgc_dot_i+1])
    print("log_tauc_bins[log_tauc_i] =", log_tauc_bins[log_tauc_i])
    print("log_tauc_bins[log_tauc_i+1] =", log_tauc_bins[log_tauc_i+1])
    print("log_taus_bins[log_taus_i] =", log_taus_bins[log_taus_i])
    print("log_taus_bins[log_taus_i+1] =", log_taus_bins[log_taus_i+1])

    #Mid point of peak bin
    print("log_r_max =", log_r_max)
    print("log_tau_max =", log_tau_max)
    print("log_Qc_max =", log_Qc_max)
    print("log_Qs_max =", log_Qs_max)
    print("lag_max =", lag_max)
    print("omgc_dot_max =", omgc_dot_max)
    print("log_tauc_max =", log_tauc_max)
    print("log_taus_max =", log_taus_max)

    #Difference between peak bin and max ll parameter value
    print("log_r_max - log_r_samples[-1] =", log_r_max - log_r_samples[-1])
    print("log_tau_max - log_tau_samples[-1] =", log_tau_max - log_tau_samples[-1])
    print("log_Qc_max - log_Qc_samples[-1] =", log_Qc_max - log_Qc_samples[-1])
    print("log_Qs_max - log_Qs_samples[-1] =", log_Qs_max - log_Qs_samples[-1])
    print("lag_max - lag_samples[-1] =", lag_max - lag_samples[-1])
    print("omgc_dot_max - omgc_dot_samples[-1] =", omgc_dot_max - omgc_dot_samples[-1])
    print("log_tauc_max - log_tauc_samples[-1] =", log_tauc_max - log_tauc_samples[-1])
    print("log_taus_max - log_taus_samples[-1] =", log_taus_max - log_taus_samples[-1])



if __name__=="__main__":
    #Parse command line arguments
    parser = argparse.ArgumentParser()
    #Input_data
    parser.add_argument("--freqfile", default=None, type=str, help="file with pulsar frequencies")
    #MCMC information
    parser.add_argument("--Nwalks", default=None, type=int, help="number of walks for mcmc")
    parser.add_argument("--Npoints", default=None, type=int, help="number of points for mcmc")
    parser.add_argument("--resume-run", help="resume mcmc run from where it left off", action="store_true", default=False)
    #Save information
    parser.add_argument("--out_directory", help="output directory", default="./outdir/", type=str)
    parser.add_argument("--tag", help="tag to include in saving information", default='real')
    args = parser.parse_args()
    run(args)


#! /usr/bin/env python
print("Importing modules")
import sdeint
import libstempo
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import argparse
import random
from utils import *
plt.rc('text', usetex=False)
print("Done importing modules")

def toas_to_freqs_nooverlap(Tfit, Nfit_min, parfile, input_timfile,
                            out_directory, tag, threshold):
    #Make a directory for the results if it doesn't already exist.
    outdirectory = Path(out_directory)
    outdirectory.mkdir(parents=True, exist_ok=True)
    #Make a directory inside the results directory for the temporary tim and par files.
    temp_file_directory = f'{out_directory}/storetemp2/'
    tempdirectory = Path(temp_file_directory)
    tempdirectory.mkdir(parents=True, exist_ok=True)

    newpsr = libstempo.tempopulsar(parfile=parfile, timfile=input_timfile)
    #pets is in units of days
    pets = np.sort(newpsr.pets())
    Ntoas = pets.size
    #toa_errors is in units of seconds
    toa_errors = newpsr.toaerrs[newpsr.pets().argsort()]*1e-6
    fc_0 = newpsr['F0'].val
    omgc_0 = fc_0*2*np.pi
    fc_dot = newpsr['F1'].val
    omgc_dot = fc_dot*2*np.pi
    #PEPOCH is in units of days 
    PEPOCH = newpsr['PEPOCH'].val
    print(f"Frequency trend from par file at t = PEPOCH = {PEPOCH} MJD is fc = {fc_0}, omgc = {2*np.pi*fc_0}")
    print(f"Frequency trend from par file at initial time, t = {pets[0]} MJD, is fc = {fc_0 + fc_dot*(pets[0]-PEPOCH)*86400}, omgc = {2*np.pi*(fc_0 + fc_dot*(pets[0]-PEPOCH)*86400)}")
    print(f"Frequency trend from par file at t = 0.0 MJD is fc = {fc_0 - fc_dot*PEPOCH*86400}, omgc = {2*np.pi*(fc_0 - fc_dot*PEPOCH*86400)}")
    print("\n")

    #Create a list of sets of TOAs to fit frequencies to.
    #Start with empty list of lists
    indexlist = []
    #Start with empty current list
    current_set = []
    i = 0
    j = 0
    #One condition to stop at the end of the list of toas
    while j < Ntoas:
        #One condition to make sure there are at least Nfit_min toas in a file.
        #One condition to stop adding toas to a set pets[j] > pets[i] + Tfit
        if j < i + Nfit_min or pets[j] < pets[i] + Tfit:
            current_set.append(j)
            j += 1
        else:
            i = j
            if len(current_set) >= Nfit_min:
                indexlist.append(current_set)
                current_set = []
            else:
                pass

    #Fit frequencies to the sets of TOAs.
    print("Fit frequencies to TOAs using tempo2")
    freqs = []
    freqs_errs = []
    times_fit = []
    for ii in tqdm(range(len(indexlist))):
        idxs = indexlist[ii]
        PEPOCH_temp = np.mean(pets[idxs])
        write_tim_file(f'{temp_file_directory}/temp_{PEPOCH_temp}.tim', pets[idxs], toa_errors[idxs])
        write_par(f'{temp_file_directory}/temp_{PEPOCH_temp}.par', fc_0 + fc_dot*(PEPOCH_temp - PEPOCH)*86400, fc_dot, PEPOCH_temp)
        psr = libstempo.tempopulsar(parfile=f'{temp_file_directory}/temp_{PEPOCH_temp}.par', timfile=f'{temp_file_directory}/temp_{PEPOCH_temp}.tim')
        psr.fit()

        freqs.append(psr['F0'].val)
        freqs_errs.append(psr['F0'].err)
        times_fit.append(PEPOCH_temp)

    #Save fitted frequencies
    write_freqs_file(f"{out_directory}/{tag}_nooverlap_freqs_uncut.freq", times_fit, freqs, freqs_errs)

    times_fit = np.asarray(times_fit).astype(np.float64)
    freqs = np.asarray(freqs).astype(np.float64)
    freqs_errs = np.asarray(freqs_errs).astype(np.float64)

    residuals = freqs - fc_0 - fc_dot*(times_fit - PEPOCH)*86400

    #Plot frequency data
    plt.figure(figsize=(20,12))
    plt.subplot(3,1,1)
    plt.errorbar(times_fit, freqs, yerr=freqs_errs, fmt='.', color='b')
    plt.xlabel('MJD')
    plt.ylabel('$f(t) [Hz]$')
    plt.title('Frequencies with error bars')

    plt.subplot(3,1,2)
    plt.errorbar(times_fit, residuals, yerr=freqs_errs, fmt='.', color='b')
    plt.xlabel('MJD')
    plt.ylabel('$f(t) [Hz]$')
    plt.title('Frequency residuals with error bars.')

    plt.subplot(3,1,3)
    plt.plot(times_fit, residuals, marker='.', color='b')
    plt.xlabel('MJD')
    plt.ylabel('$f(t) [Hz]$')
    plt.title('Frequency residuals without errorbars.')

    plt.tight_layout()
    plt.savefig(f"{out_directory}/{tag}_nooverlap_freqs_uncut.png")
    plt.close()


    print(f"\nCut outliers greater than threshold = {threshold/(2*np.pi)} Hz = {threshold} rad/s:")
    if threshold:
        indexs = []
        badcount = 0
        for j in range(len(times_fit)):
            if np.abs(residuals[j]) < threshold/(2*np.pi):
                indexs.append(j)
            else:
                print("j =", j)
                print("times_fit[j] =", times_fit[j])
                print("residuals[j] =", residuals[j])
                print("residuals[j]*2*np.pi =", residuals[j]*2*np.pi)
                print("freqs_errs[j] =", freqs_errs[j])
                print("freqs_errs[j]*2*np.pi =", freqs_errs[j]*2*np.pi)
                print("")

        print("Originally, number of toas =", len(times_fit))
        print(f"After cutting outliers greater than {threshold:.0e}, number of toas =", len(indexs))
        indexs = np.asarray(indexs)
        times_fit = times_fit[indexs]
        freqs = freqs[indexs]
        freqs_errs = freqs_errs[indexs]
        residuals = residuals[indexs]

        #Plot frequency data with outliers cut
        plt.figure(figsize=(20,12))

        plt.subplot(3,1,1)
        plt.errorbar(times_fit, freqs, yerr=freqs_errs, fmt='.', color='b')
        plt.xlabel('MJD')
        plt.ylabel('$f(t) [Hz]$')
        plt.title('Frequencies over time.')

        plt.subplot(3,1,2)
        plt.errorbar(times_fit, residuals, yerr=freqs_errs, fmt='.', color='b')
        plt.xlabel('MJD')
        plt.ylabel('$f(t) [Hz]$')
        plt.title('Frequency residuals with errorbars.')

        plt.subplot(3,1,3)
        plt.plot(times_fit, residuals, linestyle='None', marker='.', color='b')
        plt.xlabel('MJD')
        plt.ylabel('$f(t) [Hz]$')
        plt.title('Frequency residuals without errorbars.')

        plt.tight_layout()
        plt.savefig(f"{out_directory}/{tag}_nooverlap_freqs_{threshold:.0e}cut.png")
        plt.close()


        #Plot residuals only
        fig, ax = plt.subplots(1, 1, figsize=(25, 6))
        ax.errorbar(times_fit, residuals*2*np.pi, yerr=freqs_errs*2*np.pi, linestyle='None', fmt='o', color='b')
        ax.set_xlabel("Time [MJD]", fontsize=20)
        ax.set_ylabel("Frequency residual [rad $s^{-1}$]", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.xaxis.get_offset_text().set_fontsize(20)
        ax.yaxis.get_offset_text().set_fontsize(20)
        plt.tight_layout()
        plt.savefig(f"{out_directory}/{tag}_nooverlap_freqs_{threshold:.0e}cut_resids.png")
        plt.close()


    #Save fitted frequencies with outliers removed
    write_freqs_file(f"{out_directory}/{tag}_nooverlap_freqs_{threshold:.0e}cut.freq", times_fit, freqs, freqs_errs)

    p, V = np.polyfit(times_fit*86400, freqs, 1, w=1/freqs_errs, cov=True)
    print(f"Frequency trend from fit at t = PEPOCH = {PEPOCH} MJD is fc = {p[0]*PEPOCH*86400+p[1]}, omgc = {2*np.pi*(p[0]*PEPOCH*86400+p[1])}")
    print(f"Frequency trend from fit at first time, t = {times_fit[0]} MJD, is fc = {p[0]*times_fit[0]*86400+p[1]}, omgc = {2*np.pi*(p[0]*times_fit[0]*86400+p[1])}")
    print(f"Frequency trend from fit at t = 0.0 MJD is fc = {p[1]}, omgc = {2*np.pi*p[1]}")
    print(f"Error on frequency = {np.sqrt(V[1,1])} Hz/s = {2*np.pi*np.sqrt(V[1,1])} rad/s^2")
    print(f"Frequency derivative = {p[0]} Hz/s, {2*np.pi*p[0]} rad/s^2")
    print(f"Error on frequency derivative = {np.sqrt(V[0,0])} Hz/s = {2*np.pi*np.sqrt(V[0,0])} rad/s^2")

    print("\n")
    print(f"Mean of freq errorbars = {np.mean(freqs_errs)} Hz = {2*np.pi*np.mean(freqs_errs)} rad/s")
    print(f"Median of freq errorbars = {np.median(freqs_errs)} Hz = {2*np.pi*np.median(freqs_errs)} rad/s")
    print(f"Mean of unsigned freq residuals = {np.mean(np.abs(freqs - omgc_0 - omgc_dot*(times_fit - PEPOCH)*86400))} Hz = {2*np.pi*np.mean(np.abs(freqs - omgc_0 - omgc_dot*(times_fit - PEPOCH)*86400))} rad/s")
    print(f"Stdev of freq residuals = {np.std(freqs - omgc_0 - omgc_dot*(times_fit - PEPOCH)*86400)} Hz = {2*np.pi*np.std(freqs - omgc_0 - omgc_dot*(times_fit - PEPOCH)*86400)} rad/s")


    #Calculate the errorbars in big and small errorbar regions
    times_fit2 = times_fit[4:29]
    freqs2 = freqs[4:29]
    freqs_errs2 = freqs_errs[4:29]
    print(f"Median of freq error bars between t = {times_fit[4]} and t = {times_fit[29]} is {np.median(freqs_errs2)} Hz = {2*np.pi*np.median(freqs_errs2)} rad/s")
    freqs_errs3 = np.concatenate((freqs_errs[:4], freqs_errs[29:]), axis=0)
    print(f"Median of freq error bars outside that region = {np.median(freqs_errs3)} Hz = {2*np.pi*np.median(freqs_errs3)} rad/s")


    dts = pets[1:]-pets[:-1]
    print(f"Time span of fitted times = {times_fit[-1]-times_fit[0]} days")
    print(f"Minimum time step = {min(dts)} = {min(dts)*86400}")
    #print("dts[dts.argsort()] =", dts[dts.argsort()]*86400)
    #print("dts[dts.argsort()] =", dts[dts.argsort()])
    #print("pets[-1]-pets[0] =", pets[-1]-pets[0])
    #print("(pets[-1]-pets[0])*86400 =", (pets[-1]-pets[0])*86400)


#The above function can be called in a python program. 
#The section below is called to run it from the terminal.
def run(params):
    toas_to_freqs_nooverlap(Tfit = params.Tfit, Nfit_min = params.Nfit_min,
                            parfile = params.parfile,
                            input_timfile = params.input_timfile,
                            out_directory = params.out_directory, 
                            tag = params.tag, threshold = params.threshold)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Tfit", default=20, type=float, help="number of days per fit")
    parser.add_argument("--Nfit_min", default=5, type=int, help="number of ToAs per fit")
    parser.add_argument("--parfile", default=None, type=str, help="par file for simulating with observing cadence")
    parser.add_argument("--input_timfile", default=None, type=str, help="file of ToAs to fit")
    parser.add_argument("--out_directory", default="./outdir/", help="output directory for sampling and plot making", type=str)
    parser.add_argument("--tag", default="test", type=str, help="output tag prefix for file names")
    parser.add_argument("--threshold", default=1e-8, type=float, help="cut freq residuals greater than threshold")
    params = parser.parse_args()
    run(params)


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

def toas_to_freqs_overlap(Tfit, Nfit_min, parfile, input_timfile,
                          output_freqfile, out_directory, tag, threshold):
    #Make a directory for the results if it doesn't already exist.
    outdirectory = Path(out_directory)
    outdirectory.mkdir(parents=True, exist_ok=True)
    #Make a directory inside the results directory for the temporary tim and par files.
    temp_file_directory = f'{out_directory}/storetemp/'
    tempdirectory = Path(temp_file_directory)
    tempdirectory.mkdir(parents=True, exist_ok=True)

    newpsr = libstempo.tempopulsar(parfile=parfile, timfile=input_timfile)
    #pets is in units of days
    pets = np.sort(newpsr.pets())
    Ntoas = pets.size
    #toa_errors is in units of seconds
    toa_errors = newpsr.toaerrs[newpsr.pets().argsort()]*1e-6
    #fc_0 is in units of Hz (it is the frequency not the angular frequency)
    fc_0 = newpsr['F0'].val
    #fc_dot is in units of Hz/s
    fc_dot = newpsr['F1'].val
    PEPOCH = newpsr['PEPOCH'].val

    #Create the list of sets of TOAs to fit frequencies to.
    #Cut TOAs up into intervals of equal time as well as possible.
    indexlist = []
    #Tfit_min is the largest distance between three TOAs. Tfit can be less than this
    #but it will mean that some groups of TOAs will be bigger than Tfit.
    Tfit_min = max(pets[Nfit_min-1:]-pets[:-(Nfit_min-1)])
    #Tfit is in days
    for i in range(Ntoas-Nfit_min+1):
        #Go to each TOA and take the three TOAs starting from there
        current_set = np.arange(i, i+Nfit_min).tolist()
        #Go through the TOAs and add any TOAs within Tfit of the first one.
        for j in range(i+Nfit_min, Ntoas):
            if pets[j] > pets[i] + Tfit:
                break
            current_set.append(j)

        #If the last TOA added to the current set was also the last TOA added to the previous set 
        #then the current set is a subset of the previous set so we should throw it away as a duplicate.
        #Otherwise, add it to the index list.
        if i == 0 or current_set[-1] != indexlist[-1][-1]:
            indexlist.append(current_set)
        else:
            pass

    #Fit frequencies to the sets of TOAs.
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
    write_freqs_file(f"{out_directory}/{tag}_overlap_uncut.freq", times_fit, freqs, freqs_errs)

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
    plt.savefig(f"{out_directory}/{tag}_freqs_overlap_uncut.png")
    plt.close()


    #Cut points with residuals or errorbars greater than threshold (which has units of rad/s).
    if threshold:
        indexs = []
        for j in range(len(times_fit)):
            if np.abs(residuals[j]) < threshold/(2*np.pi) and np.abs(freqs_errs[j]) < threshold/(2*np.pi):
                indexs.append(j)

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
        plt.title('Frequency residuals.')

        plt.tight_layout()
        plt.savefig(f"{out_directory}/{tag}_freqs_overlap_{threshold:.0e}cut.png")
        plt.close()

    #Save fitted frequencies with outliers removed
    write_freqs_file(output_freqfile, times_fit, freqs, freqs_errs)


#The above function can be called if I want to use it in a python program. 
#This section is called if I want to run it from the terminal.
def run(params):
    toas_to_freqs_overlap(Tfit = params.Tfit, Nfit_min = params.Nfit_min,
                          parfile = params.parfile,
                          input_timfile = params.input_timfile,
                          output_freqfile = params.output_freqfile,
                          out_directory = params.out_directory, 
                          tag = params.tag, threshold = params.threshold)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Tfit", default=10, type=float, help="number of days per fit")
    parser.add_argument("--Nfit_min", default=3, type=int, help="number of TOAs per fit")
    parser.add_argument("--parfile", default=None, type=str, help="par file for simulating with observing cadence")
    parser.add_argument("--input_timfile", default=None, type=str, help="file of TOAs to fit")
    parser.add_argument("--output_freqfile", default=None, type=str, help="file to save the fitted frequencies in")
    parser.add_argument("--out_directory", default="./outdir/", help="output directory for sampling and plot making", type=str)
    parser.add_argument("--tag", default="test", type=str, help="output tag prefix for file names")
    parser.add_argument("--threshold", default=1e-8, type=float, help="cut freq residuals greater than threshold")
    params = parser.parse_args()
    run(params)


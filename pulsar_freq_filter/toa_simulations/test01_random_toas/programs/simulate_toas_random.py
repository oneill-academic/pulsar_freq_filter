import sdeint
import libstempo
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import random
from utils import *
from pathlib import Path
plt.rc('text', usetex=False)

def param_map(A, B, C, D):
    tauc = (1+A) / A * B
    taus = (1+A) * B
    Nc = (D + A/(1+A) * C/B)
    Ns = D - C/B * (1+A)**-1
    return tauc, taus, Nc, Ns

def simulate_toas_random(tau, r, sigmac, sigmas, lag, omgc_dot, omgc_0, phi_0,
                         PEPOCH, T_error_in, Nobs, Tdays, output_parfile, output_timfile, 
                         output_freqfile, out_directory, tag):
    #Make the directory for the results if it doesn't already exist
    outdirectory = Path(out_directory)
    outdirectory.mkdir(parents=True, exist_ok=True)

    print('Simulate nonuniform TOAs from nothing')
    mytimes = np.linspace(0, Tdays, Nobs*1000)
    indexs = np.sort(np.random.choice(Nobs*1000, Nobs, replace = False))
    mytimes = mytimes[indexs]
    pets0 = mytimes[0]
    tstarts = mytimes*86400
    toa_errors = np.ones(Nobs) * T_error_in
    omgc_0 = omgc_0 + omgc_dot*(pets0 - PEPOCH)*86400

    #Write a new parfile with the frequency at pets0 instead of PEPOCH
    write_par(output_parfile, omgc_0/(2*np.pi), omgc_dot/(2*np.pi), pets0)

    tauc, taus, Nc, Ns = param_map(r, tau, lag, omgc_dot)
    #Set up the two component model for the simulation
    F = np.array([[0, 1/(2*np.pi), 0], [0, -1/tauc, 1/tauc], [0, 1/taus, -1/taus]])
    N = np.array([0, Nc, Ns])
    Q = np.diag([0, sigmac, sigmas])
    def f(x, t):
        return F.dot(x) + N
    def g(x, t):
        return Q

    toa_fracs = []
    toa_ints = []
    phis = []
    omegacs = []
    omegass = []
    skipsize = min(1000, min(tstarts[1:]-tstarts[:-1]))
    print("skipsize =", skipsize)
    print("2*np.pi/omgc_0 =", 2*np.pi/omgc_0)

    #Simulate frequencies and phases of the pulsar to calculate TOAs
    #Set initial state
    omgs_0 = omgc_0 - lag
    p0 = np.asarray([phi_0, omgc_0, omgs_0])
    prev_tstart = tstarts[0]
    #The first time through the loop the while loop is skipped, it just evolves the system to an integer phase
    for next_tstart in tqdm(tstarts):
        #Move from prev_start to next_start
        times = [prev_tstart]
        while next_tstart > times[-1]:
            #Evolve forward to next_tstart in small steps to maintain precision
            times = np.linspace(times[-1], times[-1] + min(skipsize, next_tstart - times[-1]), num=2)
            states = sdeint.itoint(f, g, p0, times)
            #Reset current state
            p0 = states[-1, :]
            #Wrap phase
            p0[0] = p0[0] - np.floor(p0[0])
        #Evolve the system forward to make the phase approximately a whole number
        extra_time = np.longdouble(1 - p0[0]) * 2 * np.pi / np.longdouble(p0[1])
        newtimes = np.linspace(0, extra_time, num=2)
        states_toa = sdeint.itoint(f, g, p0, newtimes)
        #Add the new toas to the list
        toa = times[-1] + extra_time
        toa_fracs.append(toa - np.floor(toa))
        toa_ints.append(np.floor(toa))
        #Add the new omega_c and omega_s values to their lists
        omegacs.append(states_toa[-1, 1])
        omegass.append(states_toa[-1, 2])
        #Update for the next cycle. 
        prev_tstart = toa
        p0 = states_toa[-1, :]

    omegacs = np.asarray(omegacs)
    omegass = np.asarray(omegass)
    #Convert toas from seconds to days for TEMPO2
    toas = np.longdouble(toa_ints) / 86400 + np.longdouble(toa_fracs) / 86400
    #Add measurement noise to TOAs
    toas += np.random.randn(toas.size) * toa_errors / 86400

    #Make sure the TOAs haven't been put out of order (unlikely)
    indexs = toas.argsort()
    toas = toas[indexs]
    toa_errors = toa_errors[indexs]
    omegacs = omegacs[indexs]
    omegass = omegass[indexs]

    #Save the frequency values and times
    freq_errs = np.zeros(len(omegacs))
    write_freqs_file(output_freqfile, toas, omegacs/(2*np.pi), freq_errs)
    write_tim_file(output_timfile, toas, toa_errors)

    #Plot data
    plt.figure(figsize=(20,12))
    plt.subplot(3,2,1)
    plt.errorbar(toas, omegacs/(2*np.pi), fmt='.', color='b')
    plt.xlabel('MJD')
    plt.ylabel('$f_c$ [Hz]')
    plt.title('Crust frequency $f_c$')

    plt.subplot(3,2,2)
    plt.errorbar(toas, omegass/(2*np.pi), fmt='.', color='g')
    plt.xlabel('MJD')
    plt.ylabel('$f_s$ [Hz]')
    plt.title('Core frequency $f_s$')

    plt.subplot(3,2,3)
    plt.errorbar(toas, (omegacs - omgc_0 - omgc_dot*(toas-pets0)*86400)/(2*np.pi), fmt='.', color='b')
    plt.xlabel('MJD')
    plt.ylabel('Frequency $[Hz]$')
    plt.title('$f_c$ residuals')

    plt.subplot(3,2,4)
    plt.errorbar(toas, (omegass - (omgc_0 - lag) - omgc_dot*(toas-pets0)*86400)/(2*np.pi), fmt='.', color='g')
    plt.xlabel('MJD')
    plt.ylabel('Frequency $[Hz]$')
    plt.title('$f_s$ residuals')

    plt.subplot(3,2,5)
    plt.plot(np.arange(0, Nobs), toas, marker='.', color='k')
    plt.xlabel('Number')
    plt.ylabel('Time (days)')
    plt.title('ToAs')

    plt.subplot(3,2,6)
    plt.plot(np.arange(0, Nobs-1), toas[1:]-toas[:-1], marker='.', color='k')
    plt.xlabel('Number')
    plt.ylabel('Timestep (days)')
    plt.title('$\\Delta t$\'s')

    plt.tight_layout()
    plt.savefig(f"{out_directory}/{tag}_freqs_and_toas_true.png")
    plt.close()



#The above function can be called if I want to use it in a python program.
#The section below is called to run it from the terminal.
def run(params):
    simulate_toas_random(params.tau, params.r, params.sigmac, params.sigmas, 
                         params.lag, params.omgc_dot, params.omgc_0, params.phi_0,
                         params.PEPOCH, params.T_error_in, params.Nobs, params.Tdays, 
                         params.output_parfile, params.output_timfile, 
                         params.output_freqfile, params.out_directory, params.tag)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", default=1e6, type=float, help="combined relaxation timescale")
    parser.add_argument("--r", default=1, type=float, help="ratio of timescales, taus/tauc")
    parser.add_argument("--sigmac", default=1e-11, type=float, help="strength of crust noise")
    parser.add_argument("--sigmas", default=1e-11, type=float, help="strength of core noise")
    parser.add_argument("--lag", default=-1e-4, type=float, help="average omegac-omegas")
    parser.add_argument("--omgc_dot", default=-1e-13, type=float, help="average frequency derivative")
    parser.add_argument("--omgc_0", default=10, type=float, help="initial frequency")
    parser.add_argument("--phi_0", default=0, type=float, help="initial phi_c/2*np.pi")
    parser.add_argument("--PEPOCH", default=0, type=float, help="time parfile was measured at")
    parser.add_argument("--T_error_in", default=1e-8, type=float, help="size of toa_errors")
    parser.add_argument("--Nobs", default=1000, type=int, help="number of toas")
    parser.add_argument("--Tdays", default=1000, type=int, help="number of days data is over")
    #The output filenames should start with '{out_directory}/{tag}' except in special cases.
    parser.add_argument("--output_parfile", default="./outdir/test.par", type=str, help="filename for saving simulated parfile")
    parser.add_argument("--output_timfile", default="./outdir/test_true.tim", type=str, help="filename for saving simulated true TOAs")
    parser.add_argument("--output_freqfile", default="./outdir/test_freq.freq", type=str, help="filename for saving simulated true freqs")
    parser.add_argument("--out_directory", default="./outdir/", help="output directory for sampling and plot making", type=str)
    parser.add_argument("--tag", default="test", type=str, help="output tag prefix for file names")
    params = parser.parse_args()
    run(params)

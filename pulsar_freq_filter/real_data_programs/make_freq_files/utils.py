import numpy as np

def write_freqs_file(filename, times, freqs, freq_errs):
    #File name should end in '.freq'
    with open(filename, 'w') as myf:
        for t, freq, fe in zip(times, freqs, freq_errs):
            print(f"{t} {freq} {fe}", file=myf)

def write_tim_file(filename, toas, toa_errors):
    #File name should end in '.tim'
    with open(filename, 'w') as myf:
        print("FORMAT 1", file=myf)
        print("MODE 1", file=myf)
        for toa, terr in zip(toas, toa_errors):
            print(f"fake 1000 {toa} {terr*1e6} BAT", file=myf)

def write_tim_file2(filename, radio_freqs, toas, toa_errors):
    #File name should end in '.tim'
    with open(filename, 'w') as myf:
        print("FORMAT 1", file=myf)
        print("MODE 1", file=myf)
        for radio_freq, toa, terr in zip(radio_freqs, toas, toa_errors):
            print(f"test {radio_freq} {toa} {terr*1e6} BAT", file=myf)

def write_par(filename, F0, F1, PEPOCH, F1err=1e-13, F0err=1e-7, fit_omgdot=True):
    #File name should end in '.par'
    with open(filename, 'w') as myf:
        print(f"{'PSRJ':15}FAKE", file=myf)
        print(f"{'RAJ':15}0", file=myf)
        print(f"{'DECJ':15}0", file=myf)
        print(f"{'F0':15}{F0} 1  {F0err}", file=myf)
        if fit_omgdot:
            fit=1
        else:
            fit=0
        if F1 is not None:
            print(f"{'F1':15}{F1:.10e} {fit} {F1err}", file=myf)
        print(f"{'PEPOCH':15}{PEPOCH}", file=myf)
        print("TRACK -2", file=myf)

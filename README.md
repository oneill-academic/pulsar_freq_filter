# pulsar_freq_filter
A package for applying Kalman filtering to real pulsar data. See **paper link** (to be added).

# Installation

Choose a directory to save the package to. This will be referred to as `download_location` in this documentation.
Download the package with git.

```bash
cd ${download_location}
git clone https://github.com/oneill-academic/pulsar_freq_filter.git
```


# Requirements

* See requirements.txt file for required python packages. The slurm scripts in this package currently assume these are in a python environment called `my_env`. This can be changed to your own environment.

* Pulsar timing data can be found at https://github.com/Molonglo/TimingDataRelease1/.
  Our paper uses data from PSR J1359-6038. To use this data, follow the commands
```bash
cd ${download_location}
git clone https://github.com/Molonglo/TimingDataRelease1.git
cp -r ${download_location}/TimingDataRelease1/pulsars/J1359-6038/ ${download_location}/pulsar_freq_filter/pulsar_freq_filter/
```
  Currently, the scripts assume the real data is saved at this location.

# Usage
Instructions for running the code are given in usage_guide.pdf.
Additional instructions are given in the note.txt files in each major directory.

# VRDL-HW1
Homework in NCTU VRDL

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
- NVIDIA RTX 2070

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended. {envs_name} is the new environment name which you should assigned it.
```
conda create -n {envs_name} python=3.6
source activate {envs_name}
pip install -r requirements.txt
```

### Prepare Images
After downloading, the data directory is structured as:
```
data
  +- training_data
  +- validation_data
```

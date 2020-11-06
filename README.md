# VRDL-HW1
Homework in NCTU VRDL

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
- NVIDIA RTX 2070

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended. {envs_name} is the new environment name which you should assign.
```
conda create -n {envs_name} python=3.6
source activate {envs_name}
pip install -r requirements.txt
```
## Dataset Preparation
The training_label.csv is already in the data directory. You can download the data on the Kaggle website: https://www.kaggle.com/c/cs-t0828-2020-hw1/data

### Prepare Images
After downloading, the data directory is structured as:
```
data
  +- training_data
    +- 000001.jpg
    +- 000002.jpg
    ...
  +- validation_data
    +- 000004.jpg
    +- 000005.jpg
    ...
```

### Data Preprocessing
It is going to split the training data randomly to generate a new training data and valid data in the data directory. The ratio of the training data and valid data is 8 : 2

```
$ python3 preprocessing.py
```

## Training
I provide 2 model for the task. One is ResNet50, and the other is DenseNet201.
You can run the ResNet50 model by following
```
$ python3 ResNet50.py
```
You can run the DenseNet201 model by following
```
$ python3 DenseNet201.py
```

## Make Submission
There are two python file to make different submission
You can run make_submission_ResNet50 to make a submission for ResNet50 model
```
$ python3 make_submission_ResNet50.py
```
You can run make_submission_ResNet50 to make a submission for DenseNet201 model
```
$ python3 make_submission_DenseNet201.py
```

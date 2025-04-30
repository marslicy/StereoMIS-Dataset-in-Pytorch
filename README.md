# StereoMIS-Dataset-in-Pytorch

## Download the Dataset

Download the *StereoMIS* dataset from [here](https://zenodo.org/records/8154924) and unzip it in the data folder.

**Be sure the version is 0.0.1.**

Theoretically, without any preprocessing of the data, you can use the pytorch dataset provided by this repo directly.
BUT! No masks for P1 frame 16801 to 19140 are provided, but they claimed they use those for training. You may change it manually!

## Prepare the Python Environment

Create a python environment with

- torch
- torchvision
- opencv-python
- scipy
- decord

Then you are ready to run `dataset.py`

## Acknowledgement

Code is adapted from https://github.com/aimi-lab/robust-pose-estimator

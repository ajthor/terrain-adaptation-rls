# Zero to Autonomy in Real-Time: Online Adaptation of Dynamics in Unstructured Environments
Online terrain adaptation via function encoders with recursive least squares updates. 

## Installation
1. Create a conda environment.
```
conda create -n rlsenv --no-default-packages
conda activate rlsenv
```
2. Install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).
3. Install Dependencies.
```
pip install --upgrade pip setuptools wheel
pip install git+https://github.com/ajthor/function-encoder.git
pip install git+https://github.com/ajthor/meta-learning.git
```

## Usage
### Process CSV file data.
The `terrain_adaptation_rls/data` directory contains data from different simulated (warty) and hardware (jackal_0770, bluebonnet) robotic platforms. 
For faster data loading, pre-process and split datasets from each platform into train and test sets. This generates shuffled train and test datasets across different terrains. 
for models trained over 10 random seeds. 
```
python3 terrain_adaptation/shuffle_and_split_data.py
```

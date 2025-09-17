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
### Pre-process training data from CSV files.
The `terrain_adaptation_rls/data` directory contains data from different simulated (warty) and hardware (jackal_0770, bluebonnet) robotic platforms. 
For faster data loading, pre-process and split datasets from each platform into train and test sets. This generates shuffled train and test datasets across different terrains. 
for models trained over 10 random seeds. 
```
python3 terrain_adaptation/shuffle_and_split_data.py
```

### Train FE, NODE, and MAML models.
To train individual models, run
```
python terrain_adaptation_rls/train.py
```
Arguments:
- `seed`: sets the random seed for training and chooses the pre-split training data
- `model`: `neural_ode`, `function_encoder`, or `maml`
- `n_basis`: sets the number of basis functions for the FE and part of the hidden layer sizes for NODE and MAML
- `gradsteps`: sets the number of gradient steps during training
- `platform`: sets the robotic platform to train on (`warty` or `jackal_0770`)
- `hidden_size`: parameter in choosing the number of neurons on the model layers
- `inner_lr`: sets the inner learning rate for MAML
- `inner_steps`: sets the number of inner steps that MAML takes during training.

To train FE, NODE, and MAML models on each platform, run
```
./train_all.sh
```

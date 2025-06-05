#!/bin/bash



python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 0
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 1
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 2
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 3
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 4
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 5
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 6
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 7
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 8
python terrain_adaptation_rls/train.py --grad 10000 --model neural_ode --seed 9

python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 0
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 1
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 2
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 3
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 4
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 5
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 6
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 7
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 8
python terrain_adaptation_rls/train.py --grad 10000 --model function_encoder --seed 9

python terrain_adaptation_rls/plot.py
python terrain_adaptation_rls/plot_scene_error_over_seeds.py
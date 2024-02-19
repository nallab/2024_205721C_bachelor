#!/bin/bash
#SBATCH --job-name bart-finetuning
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 24:00:00

date
singularity exec --nv env_train.sif python3 ex3.py
date

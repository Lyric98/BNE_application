#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=30GB

module load Anaconda3/2020.11

python BNE_tune.py --bne_gp_lengthscale 1 --bma_gp_l2_regularizer 0.001


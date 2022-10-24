#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=minecraft_3D_holey_maze_narrow3Dholey_SeqNCA3D_3-scans_lr-5.0e-06_12.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ssssss@nyu.edu
#SBATCH --output=rl_runs/pcgrl_minecraft_3D_holey_maze_narrow3Dholey_SeqNCA3D_3-scans_lr-5.0e-06_12_%j.out

## 你不用手动改这些文件名，只需要改上面的SBATCH参数就行了
python BNE_tune.py --bne_gp_lengthscale 1 --bma_gp_l2_regularizer 0.001


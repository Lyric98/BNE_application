#!/bin/bash
#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=80000
#SBATCH --output=%j.out

python bma_cos.py --ls 1 --l2 0.01 

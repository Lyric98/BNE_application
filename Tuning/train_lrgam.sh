#!/bin/bash
#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --mem=80000
#SBATCH --output=%j.out

python BMA_tune_ref2model.py --ls 1.0 --l2 1.0 

#!/bin/bash
#SBATCH -J rjob
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-08:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test   	    # Partition to submit to
#SBATCH --mem=80000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o .result/pyjob_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e .result/pyjob_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/2020.11

python BMA_tune_cv.py --ls 1 --l2 0.01 


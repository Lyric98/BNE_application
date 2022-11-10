"""
Launch a batch of experiments on a SLURM cluster.
"""
import itertools
import json
import os
import re
from typing import Dict, List

import numpy as np

bma_gp_lengthscale = [.05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4, .425, .45, .475, .5, .525, .55, .575, .6, .625, .65, .675, .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1.0]
bma_gp_l2_regularizer = [.025, .05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4, .425, .45, .475, .5, .525, .55, .575, .6, .625, .65, .675, .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1.0]
# R^2
bma_gp_lengthscale = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
bma_gp_l2_regularizer = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
#bne_gp_lengthscale = [0.2, 0.01]
#bne_gp_l2_regularizer = [.1, .05]

def launch_batch_cluster(bma_gp_lengthscale, bma_gp_l2_regularizer):
    sbatch_name = "train.sh"
    exp_hypers = itertools.product(bma_gp_lengthscale, bma_gp_l2_regularizer)  # 把list里的元素两两组合

    for (bma_gp_lengthscale, bma_gp_l2_regularizer) in exp_hypers: 
        exp_name = f"BMA_{bma_gp_lengthscale}_{bma_gp_l2_regularizer}"
        # Edit the sbatch file to load the correct config 修改sbatch文件里的内容
        with open(sbatch_name, "r") as f:
            content = f.read()
            content = re.sub(
                "python CVCV.py --ls .* --l2 .* ",
                f"python CVCV.py --ls {bma_gp_lengthscale}  --l2 {bma_gp_l2_regularizer} ",
                content,
            )

            # # Replace the job name.  (This is optional.) (if your job is saved in runs/ folder)
            # content = re.sub(
            #     "runs/.*",
            #     f"runs/{config_name}_%j.out",       # %j is the job id
            #     content
            # )

            content = re.sub(
                "--job-name=.*",
                f"--job-name={exp_name}.out",
                content
            )
        with open(sbatch_name, "w") as f:
            f.write(content)
    
        os.system(f"sbatch {sbatch_name}")



# def launch_batch_local(bma_gp_lengthscale, bma_gp_l2_regularizer):
#     py_script_name = "CVCV.py"

#     exp_hypers = itertools.product(bma_gp_lengthscale, bma_gp_l2_regularizer)  # 把list里的元素两两组合

#     for (bma_gp_lengthscale, bma_gp_l2_regularizer) in exp_hypers: 
#         full_cmd = f"python {py_script_name} --ls {bma_gp_lengthscale} --l2 {bma_gp_l2_regularizer}"
#         print(f"Running command:\n{full_cmd}")
#         os.system(full_cmd)


if __name__ == "__main__":
    launch_batch_cluster(bma_gp_lengthscale, bma_gp_l2_regularizer)
    #launch_batch_local(bma_gp_lengthscale, bma_gp_l2_regularizer)
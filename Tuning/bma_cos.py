"""
Launch a batch of experiments on a SLURM cluster.
"""
import itertools
import json
import os
import re
from typing import Dict, List

import numpy as np

bma_gp_lengthscale = [.025, .05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4, .425, .45, .475, .5, .525, .55, .575, .6, .625, .65, .675, .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.225, 1.25, 1.275, 1.3, 1.325, 1.35, 1.375, 1.4, 1.425, 1.45, 1.475, 1.5, 1.525, 1.55, 1.575, 1.6, 1.625, 1.65, 1.675, 1.7, 1.725, 1.75, 1.775, 1.8, 1.825, 1.85, 1.875, 1.9, 1.925, 1.95, 1.975, 2., 2.025, 2.05, 2.075, 2.1, 2.125, 2.15, 2.175, 2.2, 2.225, 2.25, 2.275, 2.3, 2.325, 2.35, 2.375, 2.4, 2.425, 2.45, 2.475, 2.5, 2.525, 2.55, 2.575, 2.6, 2.625, 2.65, 2.675, 2.7, 2.725, 2.75, 2.775, 2.8, 2.825, 2.85, 2.875, 2.9, 2.925, 2.95, 2.975, 3.0, 3.025, 3.05, 3.075, 3.1, 3.125, 3.15, 3.175, 3.2, 3.225, 3.25, 3.275, 3.3, 3.325, 3.35, 3.375, 3.4, 3.425, 3.45, 3.475, 3.5, 3.525, 3.55, 3.575, 3.6, 3.625, 3.65, 3.675, 3.7, 3.725, 3.75, 3.775, 3.8, 3.825, 3.85, 3.875, 3.9, 3.925, 3.95, 3.975, 4.0, 4.025, 4.05, 4.075, 4.1, 4.125, 4.15, 4.175, 4.2, 4.225, 4.25, 4.275, 4.3, 4.325, 4.35, 4.375, 4.4, 4.425, 4.45, 4.475, 4.5, 4.525]
bma_gp_l2_regularizer = [.025, .05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4, .425, .45, .475, .5, .525, .55, .575, .6, .625, .65, .675, .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1.0]

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
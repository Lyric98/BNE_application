"""
Launch a batch of experiments on a SLURM cluster.
"""
import itertools
import json
import os
import re
from typing import Dict, List

import numpy as np

#bma_gp_lengthscale = [.05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4, .425, .45, .475, .5, .525, .55, .575, .6, .625, .65, .675, .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1.0]
#bma_gp_l2_regularizer = [.025, .05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4, .425, .45, .475, .5, .525, .55, .575, .6, .625, .65, .675, .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1.0]
bne_gp_lengthscale = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14, 15, 16, 17, 18, 19, 20]
bne_gp_l2_regularizer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14, 15, 16, 17, 18, 19, 20]

def launch_batch_cluster(bne_gp_lengthscale, bne_gp_l2_regularizer):
    sbatch_name = "train.sh"
    exp_hypers = itertools.product(bne_gp_lengthscale, bne_gp_l2_regularizer)  # 把list里的元素两两组合

    for (bne_gp_lengthscale, bne_gp_l2_regularizer) in exp_hypers: 
        exp_name = f"BMA_{bne_gp_lengthscale}_{bne_gp_l2_regularizer}"
        # Edit the sbatch file to load the correct config 修改sbatch文件里的内容
        with open(sbatch_name, "r") as f:
            content = f.read()
            content = re.sub(
                "python BNE_1213.py --ls .* --l2 .* ",
                f"python BNE_1213.py --ls {bne_gp_lengthscale}  --l2 {bne_gp_l2_regularizer} ",
                content,
            )


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
    launch_batch_cluster(bne_gp_lengthscale, bne_gp_l2_regularizer)
    #launch_batch_local(bma_gp_lengthscale, bma_gp_l2_regularizer)
"""
Launch a batch of experiments on a SLURM cluster.
"""
import itertools
import json
import os
import re
from typing import Dict, List

import numpy as np


bma_gp_lengthscale = [0.1, 0.15]
bma_gp_l2_regularizer = [.1, .09]
bne_gp_lengthscale = [0.2, 0.01]
bne_gp_l2_regularizer = [.1, .05]

def launch_batch_cluster(bma_gp_lengthscale, bma_gp_l2_regularizer, bne_gp_lengthscale, bne_gp_l2_regularizer):
    sbatch_name = "train.sh"
    exp_hypers = itertools.product(bma_gp_lengthscale, bne_gp_lengthscale, bma_gp_l2_regularizer, bne_gp_l2_regularizer)  # 把list里的元素两两组合

    for (bma_gp_lengthscale, bne_gp_lengthscale, bma_gp_l2_regularizer, bne_gp_l2_regularizer) in exp_hypers: 
        exp_name = f"BMA_{bma_gp_lengthscale}_{bma_gp_l2_regularizer}_BNE_{bne_gp_lengthscale}_{bne_gp_l2_regularizer}"
        # Edit the sbatch file to load the correct config 修改sbatch文件里的内容
        with open(sbatch_name, "r") as f:
            content = f.read()
            content = re.sub(
                "python BNE_tune.py --bma_gp_lengthscale .* --bne_gp_lengthscale .* --bma_gp_l2_regularizer .* --bne_gp_l2_regularizer .*",
                f"python BNE_tune.py --bma_gp_lengthscale {bma_gp_lengthscale} --bne_gp_lengthscale {bne_gp_lengthscale} --bma_gp_l2_regularizer {bma_gp_l2_regularizer} --bne_gp_l2_regularizer {bne_gp_l2_regularizer}",
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



def launch_batch_local(bma_gp_lengthscale, bne_gp_lengthscale, bma_gp_l2_regularizer, bne_gp_l2_regularizer):
    py_script_name = "BNE_tune.py"

    exp_hypers = itertools.product(bma_gp_lengthscale, bne_gp_lengthscale, bma_gp_l2_regularizer, bne_gp_l2_regularizer)  # 把list里的元素两两组合

    for (bma_gp_lengthscale, bne_gp_lengthscale, bma_gp_l2_regularizer, bne_gp_l2_regularizer) in exp_hypers: 
        full_cmd = f"python {py_script_name} --bma_gp_lengthscale {bma_gp_lengthscale} --bne_gp_lengthscale {bne_gp_lengthscale} --bma_gp_l2_regularizer {bma_gp_l2_regularizer} --bne_gp_l2_regularizer {bne_gp_l2_regularizer}"
        print(f"Running command:\n{full_cmd}")
        os.system(full_cmd)


if __name__ == "__main__":
    # launch_batch_cluster(bne_gp_lengthscale, bma_gp_l2_regularizer)
    launch_batch_local(bma_gp_lengthscale, bne_gp_lengthscale, bma_gp_l2_regularizer, bne_gp_l2_regularizer)
# BNE_application

## Installation

To install requirements:
```
$ conda env update -f environment.yml
$ conda activate BNE
```
This environment includes all necessary dependencies.

To create an `BNE` executable to run experiments, run `pip install -e .`.


## Instructions for Use

### Reproducibility 
Each notebook can be run independently to generate the examples, simulation experiment, or the case study. The R code can be run to prepare the data for the case study. We recommend using the Unix code in convert_nb_to_py.txt to convert the notebooks starting with "0_" to python scripts. These scripts can then be sourced at the start of each notebook. Note that base data refers to data that the base models are trained on; train data refers to data that the ensemble is trained on, and test data refers to data that the ensemble is evaluated on. The R code in the eastMA_case_study folder should be ran prior to the 3_experiment since the R code is used to generate the data used in 3_experiment. The other notebooks can be run independently.




## Tuning Hyperparameters
```
python BNE_tune.py --bma_gp_lengthscale .05 --bma_gp_l2_regularizer 0.09 --bne_gp_lengthscale 4 --bne_gp_l2_regularizer 5
```


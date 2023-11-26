# BNE_application

## Installation

To install requirements:
```
$ conda env update -f environment.yml
$ conda activate BNE
```
This environment includes all necessary dependencies.

To create an `BNE` executable to run experiments, run `pip install -e .`.




## Tuning Hyperparameters
```
python BNE_tune.py --bma_gp_lengthscale .05 --bma_gp_l2_regularizer 0.09 --bne_gp_lengthscale 4 --bne_gp_l2_regularizer 5
```

trace plot: https://www.tensorflow.org/probability/examples/A_Tour_of_TensorFlow_Probability#outline 


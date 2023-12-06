import argparse

def test_parse(opts):
    print("#########################")
    print("Testing here:")
    print("BMA GP lengthscale is: ", opts.ls, 
          " and BMA GP L2 regularizer is: ", opts.l2)
    print("Thanks for testing!")
    print("#########################")


if __name__ == '__main__':
    opts = argparse.ArgumentParser(
        description="Launch a batch of experiments"
    )

    opts.add_argument(
        "--ls",
        help="the first parameter to tune",
        type=float,
        default=1.,
    )
    opts.add_argument(
        "--l2",
        help="the second parameter to tune",
        type=float,
        default=1e-1,
    )
    opts = opts.parse_args()

    test_parse(opts)


from wrapper_functions import *

# BMA parameters.
y_noise_std = 0.01  # Note: Changed from 0.1 # @param
bma_gp_lengthscale = opts.ls # @param
bma_gp_l2_regularizer = opts.l2 # @param

bma_n_samples_train = 100 # @param
bma_n_samples_eval = 1000 # @param
bma_n_samples_test = 250 # @param
bma_seed = 0 # @param

# Optimization configs. 
# Consider reduce below parameters / set to `False` if MCMC is taking too long:
# mcmc_num_steps, mcmc_burnin, mcmc_nchain, mcmc_initialize_from_map.
map_step_size=5e-4   # @param
map_num_steps=10_000  # @param

mcmc_step_size=1e-4 # @param
mcmc_num_steps=10_000 # @param

mcmc_nchain=10 # @param
mcmc_burnin=2500 # @param
bne_mcmc_initialize_from_map="True" # @param ["False", "True"]

bne_mcmc_initialize_from_map = eval(bne_mcmc_initialize_from_map)

      # Assemble into configs.
bma_model_config = DEFAULT_GP_CONFIG.copy()
map_config = DEFAULT_MAP_CONFIG.copy()
mcmc_config = DEFAULT_MCMC_CONFIG.copy()

bma_model_config.update(dict(lengthscale=bma_gp_lengthscale,
                             l2_regularizer=bma_gp_l2_regularizer,
                             y_noise_std=y_noise_std,
                             activation=None))

map_config.update(dict(learning_rate=map_step_size,
                       num_steps=map_num_steps))

mcmc_config.update(dict(step_size=mcmc_step_size, 
                        num_steps=mcmc_num_steps,
                       burnin=mcmc_burnin,
                       nchain=mcmc_nchain,
                       debug_mode=False))

# BNE parameters.
bne_gp_lengthscale = 4 # 5. # @param
bne_gp_l2_regularizer = 5 # 15 # @param
bne_variance_prior_mean = -2.5 # @param
bne_skewness_prior_mean = -2.5 # @param
bne_seed = 0 # @param

estimate_mean = "True" # @param ["True", "False"]
variance_prior_mean=0. # @param
# MAP and MCMC configs

bne_gp_config = DEFAULT_GP_CONFIG.copy()
bne_model_config = DEFAULT_BNE_CONFIG.copy()



bne_gp_config.update(dict(lengthscale=bne_gp_lengthscale, 
                          l2_regularizer=bne_gp_l2_regularizer))
bne_model_config.update(dict(estimate_mean=eval(estimate_mean),
                             variance_prior_mean=variance_prior_mean,
                             **bne_gp_config))
print("BMA model config:", bma_model_config, "\n", "BNE model config:", bne_model_config, "\n", "MAP config:", map_config, "\n", "MCMC config:", mcmc_config)



training_eastMA = pd.read_csv('../data/training_dataset/training_eastMA.csv')
training_eastMA_noMI = training_eastMA[:51]
training_eastMA_folds = pd.read_csv('../data/training_dataset/training_eastMA_folds.csv')
base_model_predictions_eastMA = pd.read_csv('../data/prediction_dataset/base_model_predictions_eastMA.csv')

print("pred longitude max and min", base_model_predictions_eastMA["lon"].max(),base_model_predictions_eastMA["lon"].min())
print("pred latitude max and min", base_model_predictions_eastMA["lat"].max(),base_model_predictions_eastMA["lat"].min())
#list(base_model_predictions_eastMA.columns)
print("train longitude max and min", training_eastMA["lon"].max(),training_eastMA["lon"].min())
print("train latitude max and min", training_eastMA["lat"].max(),training_eastMA["lat"].min())


training51= pd.read_csv('../data/training_dataset/training51.csv')



# standardize
X_train1 = np.asarray(training_eastMA_noMI[["lon", "lat"]].values.tolist()).astype(np.float32)
X_test1 = np.asarray(base_model_predictions_eastMA[["lon", "lat"]].values.tolist()).astype(np.float32)
X_valid = np.concatenate((X_train1, X_test1), axis=0)
X_centr = np.mean(X_valid, axis=0)
X_scale = np.max(X_valid, axis=0) - np.min(X_valid, axis=0)

X_train1 = (X_train1 - X_centr) / X_scale
X_test1 = (X_test1 - X_centr) / X_scale

Y_train = np.expand_dims(training_eastMA_noMI["aqs"], 1).astype(np.float32)
#Y_test = np.expand_dims(base_model_predictions_eastMA["pred_av"], 1).astype(np.float32)



base_model_names = ["pred_av", "pred_gs", "pred_caces"]
base_preds_train = tf.stack([training_eastMA_noMI[base_model_name].astype(np.float32) for base_model_name in base_model_names], axis=-1)
base_preds_test = tf.stack([base_model_predictions_eastMA[base_model_name].astype(np.float32) for base_model_name in base_model_names], axis=-1)
#base_preds_test

coverage_lr = 0
coverage_gam = 0
coverage_bma = 0


import rpy2
#from rpy2.robjects import pandas2ri
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter
# import R's "base" package
base = importr('base')
#ms = importr('MSGARCH')
# import R's "utils" package
utils = importr('utils')


with localconverter(ro.default_converter + pandas2ri.converter):
  r_from_pd_df = ro.conversion.py2rpy(training_eastMA_noMI)


mgcv  = importr('mgcv')
stats = importr('stats')
ciTools = importr('ciTools')


#ref_model = LinearRegression()
kf = KFold(n_splits=10, random_state=bma_seed, shuffle=True) 

rmse_lr = []
rmse_bma = []
rmse_gam = []

for train_index, test_index in kf.split(X_train1):
      #print("Train:", train_index, "Validation:",test_index)
      X_tr, X_te = X_train1[train_index], X_train1[test_index] 
      Y_tr, Y_te = Y_train[train_index], Y_train[test_index]
      
      base_preds_tr, base_preds_te = base_preds_train.numpy()[train_index], base_preds_train.numpy()[test_index]
      print(X_tr.shape, X_te.shape, Y_tr.shape, Y_te.shape, base_preds_tr.shape, base_preds_te.shape)
    
      r_dat_py = training_eastMA_noMI
      #r_dat_py[['lon', 'lat']] = X_train1
      
      with localconverter(ro.default_converter + pandas2ri.converter):
            r_tr = ro.conversion.py2rpy(r_dat_py.iloc[train_index])
            r_te = ro.conversion.py2rpy(r_dat_py.iloc[test_index])

      # Ref: lr
      lr_model = stats.lm(ro.Formula('aqs~pred_av+pred_gs+pred_caces'), data=r_tr)
      #l = stats.predict(lr_model, newdata =r_te, interval = 'prediction')
      #py_l = np.asanyarray(l).reshape(-1, 3)
      #py_l = pd.DataFrame(py_l, columns=['pred', 'l', 'u'])
      #lr_ci_l, lr_ci_u = py_l['l'], py_l['u']
      l = ciTools.add_pi(r_te, lr_model)
      lr_pred = l[7]
      lr_ci_l, lr_ci_u = l[8], l[9]
      coverage_lr += np.sum([(Y_te[i] > lr_ci_l[i]) & (Y_te[i] < lr_ci_u[i]) for i in range(len(Y_te))])
      rmse_lr.append(rmse(Y_te, np.asanyarray(lr_pred).reshape(-1,1)))

      # Ref: GAM
      #df = training_eastMA_noMI.iloc[train_index]
      gam_model = mgcv.gam(ro.Formula('aqs ~ s(lon, lat, by=pred_av, k=4) + s(lon, lat,by=pred_gs, k=4) +s(lon, lat, by=pred_caces, k=4)'), data=r_tr)
      a= ciTools.add_pi(r_te, gam_model)
      Y_pred = a[7]
      gam_ci_l, gam_ci_u = a[8], a[9]
      coverage_gam += np.sum([(Y_te[i] > gam_ci_l[i]) & (Y_te[i] < gam_ci_u[i]) for i in range(len(Y_te))])
      rmse_gam.append(rmse(Y_te, np.asanyarray(Y_pred).reshape(-1,1)))
      #print(rmse_gam)




print("rmse_lr:", rmse_lr, "rmse_gam:", rmse_gam)
print("rmse_lr_mean:", np.mean(rmse_lr),"\n", "rmse_lr_std",  np.std(rmse_lr),"\n", "rmse_gam_mean:", np.mean(rmse_gam), "\n", "rmse_gam_std:", np.std(rmse_gam))
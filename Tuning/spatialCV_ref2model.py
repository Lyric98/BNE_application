import argparse

def test_parse(opts):
    print("#########################")
    print("Testing here:")
    print("BMA GP lengthscale is: ", opts.ls, 
          " and BMA GP L2 regularizer is: ", opts.l2,
          " and activation function is: ", opts.activation)
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
    opts.add_argument(
        "--activation",
        help="the activation function to use",
        type=str,
        default="softmax",
    )
    opts = opts.parse_args()

    test_parse(opts)


from wrapper_functions import *


# Optimization configs. 
# Consider reduce below parameters / set to `False` if MCMC is taking too long:
# mcmc_num_steps, mcmc_burnin, mcmc_nchain, mcmc_initialize_from_map.
map_step_size=5e-4   # @param
map_num_steps=10_000  # @param

mcmc_step_size=1e-4 # @param
mcmc_num_steps=1000 # @param

mcmc_nchain=10 # @param
mcmc_burnin=2_500 # @param
bne_mcmc_initialize_from_map="True" # @param ["False", "True"]

bne_mcmc_initialize_from_map = eval(bne_mcmc_initialize_from_map)


# BMA parameters.
y_noise_std = 0.01  # Note: Changed from 0.1 # @param
bma_gp_lengthscale = opts.ls # @param
bma_gp_l2_regularizer = opts.l2 # @param
activation_func = opts.activation

bma_n_samples_train = 100 # @param
bma_n_samples_eval = 250 # @param
bma_n_samples_test = 250 # @param
bma_seed = 0 # @param

# BNE parameters.
bne_gp_lengthscale = .08 # 5. # @param
bne_gp_l2_regularizer = 1. # 15 # @param
bne_variance_prior_mean = -2.5 # @param
bne_skewness_prior_mean = -2.5 # @param
bne_seed = 0 # @param


# ### Read training/prediction data

training_eastMA = pd.read_csv('../data/training_dataset/training51_new.csv')
training_eastMA_noMI = training_eastMA[:51]
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

print("2011 center and scale: ", X_centr, X_scale)


base_model_names = ["pred_av", "pred_gs", "pred_caces"]
base_preds_train = tf.stack([training_eastMA_noMI[base_model_name].astype(np.float32) for base_model_name in base_model_names], axis=-1)
base_preds_test = tf.stack([base_model_predictions_eastMA[base_model_name].astype(np.float32) for base_model_name in base_model_names], axis=-1)


data_dicts = dict(X_train = X_train1,
                  X_test=X_test1,
                  Y_train = Y_train,
                  base_preds_train = base_preds_train,
                  base_preds_test = base_preds_test)

bma_config = dict(gp_lengthscale=bma_gp_lengthscale,
                  gp_l2_regularizer=bma_gp_l2_regularizer,
                  y_noise_std=y_noise_std,
                  map_step_size=map_step_size,
                  map_num_steps=map_num_steps,
                  mcmc_step_size=mcmc_step_size,
                  mcmc_num_steps=mcmc_num_steps,
                  mcmc_initialize_from_map=False,
                  n_samples_eval=bma_n_samples_eval,
                  n_samples_train=bma_n_samples_train,
                  n_samples_test=bma_n_samples_test,
                  seed=bma_seed)

bne_config = dict(gp_lengthscale=bne_gp_lengthscale,
                  gp_l2_regularizer=bne_gp_l2_regularizer,
                  variance_prior_mean=bne_variance_prior_mean,
                  skewness_prior_mean=bne_skewness_prior_mean,
                  map_step_size=map_step_size,
                  map_num_steps=map_num_steps,
                  mcmc_step_size=mcmc_step_size,
                  mcmc_num_steps=mcmc_num_steps,
                  mcmc_nchain=mcmc_nchain,
                  mcmc_burnin=mcmc_burnin,
                  mcmc_initialize_from_map=bne_mcmc_initialize_from_map,
                  seed=bne_seed)


def get_bne_result(data_dict, moment_mode, bne_config):
  """Trains Bayesian nonparametric ensemble."""
  mode_to_name_map = {'none': 'bma', 'mean': 'bae',
                      'variance': 'bne_var', 'skewness': 'bne_skew'}
  model_name = mode_to_name_map[moment_mode]

  joint_samples = run_bne_model(X_train=data_dict['X_train_mcmc'],
                                Y_train=data_dict['Y_train_mcmc'],
                                X_test=data_dict['X_test'],
                                base_model_samples_train=data_dict['means_train_mcmc'],
                                base_model_samples_test=data_dict['means_test_mcmc'],
                                moment_mode=moment_mode,
                                **bne_config)

  #data_dict[f'{model_name}_samples'] = joint_samples['y']
  #return data_dict
  return joint_samples

# @title Simulation: get_bma_result


def get_bma_result(data_dict, bma_config):
  """Trains Adaptive Bayesian model averaging."""
  (bma_joint_samples, X_train_mcmc, Y_train_mcmc,
   means_train_mcmc, means_test_mcmc) = run_bma_model(
       X_train=data_dict["X_train"],
       X_test=data_dict["X_test"],
       Y_train=data_dict["Y_train"],
       base_preds_train=data_dict["base_preds_train"],
       base_preds_test=data_dict["base_preds_test"],
       return_mcmc_examples=True,
       **bma_config)

  data_dict['X_train_mcmc'] = X_train_mcmc
  data_dict['Y_train_mcmc'] = Y_train_mcmc
  data_dict['means_train_mcmc'] = means_train_mcmc
  data_dict['means_test_mcmc'] = means_test_mcmc
  data_dict['bma_mean_samples'] = bma_joint_samples['y']
  # data_dict['bma_mean_samples_original'] = bma_joint_samples['mean_original']
  # data_dict['bma_mean_samples_resid'] = bma_joint_samples['resid']
  # data_dict['bma_weight_samples'] = bma_joint_samples['weights']

  return data_dict, bma_joint_samples


def calc_prediction_std(y_pred, y):
  """
  This function takes two arguments:
  y_pred: a TensorFlow tensor containing the predicted values of the response variable
  y: a TensorFlow tensor containing the observed values of the response variable
  The function calculates the residuals, mean, variance, and standard deviation of the residuals using TensorFlow operations and returns the standard deviation as a TensorFlow tensor.
  To use this function, you will need to pass in the appropriate tensors as arguments and execute the TensorFlow graph to calculate the standard deviation.
  """
  # Calculate the residuals
  residuals = y - y_pred

  # Calculate the mean of the residuals
  mean = tf.reduce_mean(residuals)
  # change tf.int32 to tf.float32
  mean = tf.cast(mean, tf.float32)
  df = tf.cast(tf.size(residuals) - 1, tf.float32)
  # Calculate the variance of the residuals
  variance = tf.reduce_sum(tf.square(residuals - mean)) / df

  # Calculate the standard deviation of the residuals
  std = tf.sqrt(variance)

  return std

def posterior_heatmap_2d_tr(plot_data, X,
                         X_monitor=None,
                         cmap='inferno_r',
                         norm=None, 
                         norm_method="percentile",
                         save_addr=''):
    
    plt.scatter(x=X[:, 0], y=X[:, 1],
                s=3,
                c=plot_data, cmap=cmap, norm=norm)
    cbar = plt.colorbar()

    plt.scatter(X_tr[:, 0] * X_scale[0] + X_centr[0], X_tr[:, 1] * X_scale[1] +
                X_centr[1], c="black", s=10, alpha=0.5, marker="s")  # different shape
    plt.scatter(X_te[:, 0] * X_scale[0] + X_centr[0], X_te[:, 1] * X_scale[1] + X_centr[1], c="blue",
                s=abs(nll_bae_each)*50)
                #, cmap='inferno_r', norm=color_norm_nll_each)

    # adjust plot window
    plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    plt.ylim((np.min(X[:, 1]), np.max(X[:, 1])))

    return norm

spcv_id = training_eastMA_noMI["fold"]

BMA_lenthscale = bma_gp_lengthscale
BNE_lenthscale = bne_gp_lengthscale
BMA_L2 = bma_gp_l2_regularizer
BNE_L2 = bne_gp_l2_regularizer

from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
import rpy2

rmse_lr = []
rmse_gam = []
rmse_bma = []
rmse_bae = []
rmse_bma2 = []
rmse_bma_mean = []


nll_lr, nll_gam, nll_bma_mean, nll_bma, nll_bae = [], [], [], [], []
nll_bma2 = []
# initialize a dataframe to store lon, lat and raw error
error_df = pd.DataFrame(columns=['lon', 'lat', 'raw_error'])

coverage_lr, coverage_gam, coverage_bma_mean, coverage_bma, coverage_bae = 0, 0, 0, 0, 0
coverage_bma2 = 0
#from rpy2.robjects import pandas2ri
# import R's "base" package
base = importr('base')
#ms = importr('MSGARCH')
# import R's "utils" package
utils = importr('utils')

with localconverter(ro.default_converter + pandas2ri.converter):
  # convert "lon" and "lat" in training_eastMA_noMI into scaled X_train1 values
  training_eastMA_noMI["lon"] = (
      training_eastMA_noMI["lon"] - X_centr[0]) / X_scale[0]
  training_eastMA_noMI["lat"] = (
      training_eastMA_noMI["lat"] - X_centr[1]) / X_scale[1]
  r_from_pd_df = ro.conversion.py2rpy(training_eastMA_noMI)


mgcv = importr('mgcv')
stats = importr('stats')
ciTools = importr('ciTools')

rmse_bma_mean, rmse_bma2, rmse_bae = [], [], []
for fold_id in range(1,11):
    print(fold_id)
    X_tr, X_te = X_train1[spcv_id!=fold_id], X_train1[spcv_id==fold_id]
    Y_tr, Y_te = Y_train[spcv_id!=fold_id], Y_train[spcv_id==fold_id]

    base_preds_tr, base_preds_te = base_preds_train.numpy()[spcv_id!=fold_id], base_preds_train.numpy()[spcv_id==fold_id]
    X_test_long = np.vstack((X_test1, X_te))
    base_preds_test_long = np.vstack((base_preds_test.numpy(), base_preds_te))

    print(X_tr.shape, X_te.shape, Y_tr.shape, Y_te.shape,
          base_preds_tr.shape, base_preds_te.shape)


    r_dat_py = training_eastMA_noMI

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_tr = ro.conversion.py2rpy(r_dat_py[spcv_id!=fold_id])
        r_te = ro.conversion.py2rpy(r_dat_py[spcv_id==fold_id])

    # Ref: lr
    lr_model = stats.lm(ro.Formula(
        'aqs~pred_av+pred_gs+pred_caces'), data=r_tr)
    l = ciTools.add_pi(r_te, lr_model)
    lr_pred = l[10]
    lr_ci_l, lr_ci_u = l[11], l[12]
    coverage_lr += np.sum([(Y_te[i] > lr_ci_l[i]) &
                          (Y_te[i] < lr_ci_u[i]) for i in range(len(Y_te))])
    rmse_lr.append(rmse(Y_te, np.asanyarray(lr_pred).reshape(-1, 1)))
    nll_lr.append(nll(Y_te, np.asanyarray(lr_pred).reshape(-1, 1)))
    print(rmse_lr)

    # Ref: GAM
    gam_model = mgcv.gam(ro.Formula(
        'aqs ~ s(lon, lat, by=pred_av, k=4) + s(lon, lat,by=pred_gs, k=4) +s(lon, lat, by=pred_caces, k=4)'), data=r_tr)
    a = ciTools.add_pi(r_te, gam_model)
    gam_pred = a[10]
    gam_ci_l, gam_ci_u = a[11], a[12]
    coverage_gam += np.sum([(Y_te[i] > gam_ci_l[i]) &
                           (Y_te[i] < gam_ci_u[i]) for i in range(len(Y_te))])
    rmse_gam.append(rmse(Y_te, np.asanyarray(gam_pred).reshape(-1, 1)))
    nll_gam.append(nll(Y_te, np.asanyarray(gam_pred).reshape(-1, 1)))
    print(rmse_gam)

    print("LR prediction", lr_pred)
    print("GAM prediction", gam_pred)

    data_dicts = dict(X_train=X_tr,
                      X_test=X_te,
                      Y_train=Y_tr,
                      base_preds_train=base_preds_tr,
                      base_preds_test=base_preds_te)

    print(Y_te)
    # BMA-mean.
    print('BMA-mean:', flush=True)
    data_dict, bma_mean_joint_samples = get_bma_result(
        data_dicts, bma_config=bma_config)
    y_pred_bma_mean = np.mean(np.nan_to_num(
        bma_mean_joint_samples['y']), axis=0)
    pred_std = calc_prediction_std(y_pred_bma_mean, Y_te)
    bma_mean_pi = np.array(
        [(y_pred_bma_mean - 1.96*pred_std).numpy(), (y_pred_bma_mean + 1.96*pred_std).numpy()])
    print(y_pred_bma_mean)

    # BMA.
    bma_var_config = bne_config.copy()
    bma_var_config['mcmc_initialize_from_map'] = bma_config['mcmc_initialize_from_map']
    bma_joint_samples = get_bne_result(data_dict, moment_mode='none',
                                       bne_config=bma_var_config)
    y_pred_bma = np.mean(np.nan_to_num(bma_joint_samples['y']), axis=0)
    print(y_pred_bma)
    pred_std = calc_prediction_std(y_pred_bma, Y_te)
    bma_pi2 = np.array([(y_pred_bma - 1.96*pred_std).numpy(),
                       (y_pred_bma + 1.96*pred_std).numpy()])

    # BAE.
    bae_joint_samples = get_bne_result(data_dict, moment_mode='mean',
                                       bne_config=bne_config)
    y_pred_bae = np.mean(np.nan_to_num(bae_joint_samples['y']), axis=0)
    print(y_pred_bae)
    pred_std = calc_prediction_std(y_pred_bae, Y_te)
    bae_pi = np.array([(y_pred_bae - 1.96*pred_std).numpy(),
                      (y_pred_bae + 1.96*pred_std).numpy()])

    # save the rmse & nll for each fold
    rmse_bma_mean.append(rmse(Y_te, y_pred_bma_mean))
    nll_bma_mean.append(nll(Y_te, y_pred_bma_mean))
    rmse_bma2.append(rmse(Y_te, y_pred_bma))
    nll_bma.append(nll(Y_te, y_pred_bma))
    rmse_bae.append(rmse(Y_te, y_pred_bae))
    nll_bae.append(nll(Y_te, y_pred_bae))

    # save the coverage for each fold
    coverage_bma_mean += np.sum([(Y_te[i] > bma_mean_pi[0][i])
                                & (Y_te[i] < bma_mean_pi[1][i]) for i in range(len(Y_te))])
    coverage_bma += np.sum([(Y_te[i] > bma_pi2[0][i]) &
                           (Y_te[i] < bma_pi2[1][i]) for i in range(len(Y_te))])
    coverage_bae += np.sum([(Y_te[i] > bae_pi[0][i]) &
                           (Y_te[i] < bae_pi[1][i]) for i in range(len(Y_te))])

    raw_error = pd.DataFrame(columns=['lon', 'lat', 'raw_error'])
    raw_error["lon"] = X_te[:, 0]
    raw_error["lat"] = X_te[:, 1]
    raw_error["raw_error"] = (y_pred_bae - Y_te).reshape(-1)
    error_df = error_df.append(raw_error)
    print("rmse:", flush=True)
    print(rmse_bma_mean, rmse_bma2, rmse_bae)
    print("nll:", flush=True)
    print(nll_bma_mean, nll_bma, nll_bae)

    

print("RMSE LR: ", np.mean(rmse_lr), np.median(rmse_lr), np.std(rmse_lr))
print("RMSE GAM: ", np.mean(rmse_gam), np.median(rmse_gam), np.std(rmse_gam))
print("RMSE BMA: ", np.mean(rmse_bma), np.median(rmse_bma), np.std(rmse_bma))

print("NLL LR: ", np.mean(nll_lr), np.median(nll_lr), np.std(nll_lr))
print("NLL GAM: ", np.mean(nll_gam), np.median(nll_gam), np.std(nll_gam))
print("NLL BMA: ", np.mean(nll_bma), np.median(nll_bma), np.std(nll_bma))

print("Coverage LR: ", coverage_lr/len(Y_train))
print("Coverage GAM: ", coverage_gam/len(Y_train))
print("Coverage BMA: ", coverage_bma/len(Y_train))

# lr_s = ['', str(np.mean(rmse_lr)), str(np.std(rmse_lr))]
# bma_s = [ str(np.mean(rmse_bma)), str(np.std(rmse_bma))]


with open('spcv0406.txt', 'a') as f:
    f.write('\n')
    f.write(''.join(str(bma_gp_lengthscale)+ " "+str(bma_gp_l2_regularizer) + " "+ 
                    str(np.mean(rmse_bma_mean)) + " "+ str(np.median(rmse_bma_mean))+ " "+str(np.std(rmse_bma_mean))+ " "+str(coverage_bma_mean/len(Y_train))+ " "+
                    str(np.mean(rmse_bma2)) + " "+ str(np.median(rmse_bma2))+ " "+str(np.std(rmse_bma2))+ " "+str(coverage_bma/len(Y_train))+ " "+
                    str(np.mean(rmse_bae)) + " "+ str(np.median(rmse_bae))+ " "+str(np.std(rmse_bae))+ " "+str(coverage_bae/len(Y_train))
                    ))
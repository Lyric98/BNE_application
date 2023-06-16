from wrapper_functions import *
import argparse


def test_parse(opts):
    print("#########################")
    print("Testing here:")
    print("BNE GP lengthscale is: ", opts.ls,
          " and BNE GP L2 regularizer is: ", opts.l2)
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


# Optimization configs.
# Consider reduce below parameters / set to `False` if MCMC is taking too long:
# mcmc_num_steps, mcmc_burnin, mcmc_nchain, mcmc_initialize_from_map.
map_step_size = 5e-4   # @param
map_num_steps = 10_000  # @param

mcmc_step_size = 1e-4  # @param
mcmc_num_steps = 1_000  # @param

mcmc_nchain = 10  # @param
mcmc_burnin = 2_500  # @param
bne_mcmc_initialize_from_map = "True"  # @param ["False", "True"]

bne_mcmc_initialize_from_map = eval(bne_mcmc_initialize_from_map)


# BMA parameters.
y_noise_std = 0.01  # Note: Changed from 0.1 # @param
bma_gp_lengthscale =  0.725 # @param
bma_gp_l2_regularizer = 0.05  # @param
activation_func = "softmax"

bma_n_samples_train = 100  # @param
bma_n_samples_eval = 250  # @param
bma_n_samples_test = 250  # @param
bma_seed = 0  # @param

# BNE parameters.
bne_gp_lengthscale = opts.ls  # 5. # @param
bne_gp_l2_regularizer = opts.l2  # 15 # @param
bne_variance_prior_mean = -2.5  # @param
bne_skewness_prior_mean = -2.5  # @param
bne_seed = 0  # @param


# ### Read training/prediction data

training_eastMA = pd.read_csv(
    '../data/training_dataset/training51_kmeans6.csv')
training_eastMA_noMI = training_eastMA[:51]
base_model_predictions_eastMA = pd.read_csv(
    '../data/prediction_dataset/base_model_predictions_eastMA.csv')

print("pred longitude max and min", base_model_predictions_eastMA["lon"].max(
), base_model_predictions_eastMA["lon"].min())
print("pred latitude max and min", base_model_predictions_eastMA["lat"].max(
), base_model_predictions_eastMA["lat"].min())
#list(base_model_predictions_eastMA.columns)
print("train longitude max and min",
      training_eastMA["lon"].max(), training_eastMA["lon"].min())
print("train latitude max and min",
      training_eastMA["lat"].max(), training_eastMA["lat"].min())


training51 = pd.read_csv('../data/training_dataset/training51.csv')

# standardize
X_train1 = np.asarray(
    training_eastMA_noMI[["lon", "lat"]].values.tolist()).astype(np.float32)
X_test1 = np.asarray(base_model_predictions_eastMA[[
                     "lon", "lat"]].values.tolist()).astype(np.float32)
X_valid = np.concatenate((X_train1, X_test1), axis=0)
X_centr = np.mean(X_valid, axis=0)
X_scale = np.max(X_valid, axis=0) - np.min(X_valid, axis=0)

X_train1 = (X_train1 - X_centr) / X_scale
X_test1 = (X_test1 - X_centr) / X_scale

Y_train = np.expand_dims(training_eastMA_noMI["aqs"], 1).astype(np.float32)
#Y_test = np.expand_dims(base_model_predictions_eastMA["pred_av"], 1).astype(np.float32)

print("2011 center and scale: ", X_centr, X_scale)


base_model_names = ["pred_av", "pred_gs", "pred_caces"]
base_preds_train = tf.stack([training_eastMA_noMI[base_model_name].astype(
    np.float32) for base_model_name in base_model_names], axis=-1)
base_preds_test = tf.stack([base_model_predictions_eastMA[base_model_name].astype(
    np.float32) for base_model_name in base_model_names], axis=-1)


data_dicts = dict(X_train=X_train1,
                  X_test=X_test1,
                  Y_train=Y_train,
                  base_preds_train=base_preds_train,
                  base_preds_test=base_preds_test)

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


spcv_id = training_eastMA_noMI["fold"]

BMA_lenthscale = bma_gp_lengthscale
BNE_lenthscale = bne_gp_lengthscale
BMA_L2 = bma_gp_l2_regularizer
BNE_L2 = bne_gp_l2_regularizer

rmse_bne = []
nll_bne = []
coverage_bne = 0

for fold_id in range(1, 7):
    print(fold_id)
    X_tr, X_te = X_train1[spcv_id != fold_id], X_train1[spcv_id == fold_id]
    Y_tr, Y_te = Y_train[spcv_id != fold_id], Y_train[spcv_id == fold_id]

    base_preds_tr, base_preds_te = base_preds_train.numpy(
    )[spcv_id != fold_id], base_preds_train.numpy()[spcv_id == fold_id]
    X_test_long = np.vstack((X_test1, X_te))
    base_preds_test_long = np.vstack((base_preds_test.numpy(), base_preds_te))

    print(X_tr.shape, X_te.shape, Y_tr.shape, Y_te.shape,
          base_preds_tr.shape, base_preds_te.shape)

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

    # BNE variance + skewness.
    print('BNE config:', flush=True)
    print(bne_config)
    bne_joint_samples = get_bne_result(data_dict, moment_mode='skewness',
                                       bne_config=bne_config)
    y_pred_bne = np.mean(np.nan_to_num(bne_joint_samples['y']), axis=0)
    print(y_pred_bne)
    pred_std = calc_prediction_std(y_pred_bne, Y_te)
    bne_pi = np.array([(y_pred_bne - 1.96*pred_std).numpy(),
                      (y_pred_bne + 1.96*pred_std).numpy()])

    rmse_bne.append(rmse(Y_te, y_pred_bne))
    nll_bne.append(nll(Y_te, y_pred_bne))

    coverage_bne += np.sum([(Y_te[i] > bne_pi[0][i]) &
                           (Y_te[i] < bne_pi[1][i]) for i in range(len(Y_te))])


with open('spcv0609.txt', 'a') as f:
    f.write('\n')
    f.write(''.join(str(bne_gp_lengthscale) + " "+str(bne_gp_l2_regularizer) + " " +
                    str(np.mean(rmse_bne)) + " " + str(np.median(rmse_bne)) + " "+str(np.std(rmse_bne)) + " "+str(coverage_bne/len(Y_train)) + " " +
                    str(np.mean(nll_bne)) + " " + str(np.median(nll_bne))))

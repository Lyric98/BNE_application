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


from wrapper_functions import *


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
bma_gp_lengthscale = .34 # @param
bma_gp_l2_regularizer = 0.25 # @param

bma_n_samples_train = 100 # @param
bma_n_samples_eval = 250 # @param
bma_n_samples_test = 250 # @param
bma_seed = 0 # @param

# BNE parameters.
bne_gp_lengthscale = opts.ls # 5. # @param
bne_gp_l2_regularizer = opts.l2 # 15 # @param
bne_variance_prior_mean = -2.5 # @param
bne_skewness_prior_mean = -2.5 # @param
bne_seed = 0 # @param


# ### Read training/prediction data


training_eastMA = pd.read_csv('../data/training_dataset/training_eastMA.csv')
training_eastMA_noMI = training_eastMA[:51]
training_eastMA_folds = pd.read_csv('../data/training_dataset/training_eastMA_folds.csv')
base_model_predictions_eastMA = pd.read_csv('../data/prediction_dataset/base_model_predictions_eastMA.csv')

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

base_model_names = ["pred_av", "pred_gs", "pred_caces"]
base_preds_train = tf.stack([training_eastMA_noMI[base_model_name].astype(np.float32) for base_model_name in base_model_names], axis=-1)
base_preds_test = tf.stack([base_model_predictions_eastMA[base_model_name].astype(np.float32) for base_model_name in base_model_names], axis=-1)

bma_config=dict(gp_lengthscale=bma_gp_lengthscale,
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

print('BMA config:', bma_config, 'BNE config:', bne_config, flush=True)


# Start cross-validation
kf = KFold(n_splits=10, random_state=bma_seed, shuffle=True) 
rmse_bma_mean, rmse_bma, rmse_bae, rmse_bne_vo, rmse_bne_vs = [], [], [], [], []
for train_index, test_index in kf.split(X_train1):
    X_tr, X_te = X_train1[train_index], X_train1[test_index] 
    Y_tr, Y_te = Y_train[train_index], Y_train[test_index]
      
    base_preds_tr, base_preds_te = base_preds_train.numpy()[train_index], base_preds_train.numpy()[test_index]
    print(X_tr.shape, X_te.shape, Y_tr.shape, Y_te.shape, base_preds_tr.shape, base_preds_te.shape)

    # Create data dictionary.
    data_dicts = dict(X_train = X_tr,
                  X_test=X_te,
                  Y_train = Y_tr,
                  base_preds_train = base_preds_tr,
                  base_preds_test = base_preds_te)

    print(Y_te)
    # BMA-mean.
    print('BMA-mean:', flush=True)
    data_dict, bma_mean_joint_samples = get_bma_result(data_dicts, bma_config=bma_config)
    y_pred_bma_mean = np.mean(np.nan_to_num(bma_mean_joint_samples['y']), axis=0)
    print(y_pred_bma_mean)

    # BMA.
    bma_var_config = bne_config.copy()
    bma_var_config['mcmc_initialize_from_map'] = bma_config['mcmc_initialize_from_map']
    bma_joint_samples = get_bne_result(data_dict, moment_mode='none',
                                      bne_config=bma_var_config)
    y_pred_bma = np.mean(np.nan_to_num(bma_mean_joint_samples['y']), axis=0)
    print(y_pred_bma)
      
    # BAE.
    bae_joint_samples = get_bne_result(data_dict, moment_mode='mean',
                                      bne_config=bne_config)
    y_pred_bae = np.mean(np.nan_to_num(bae_joint_samples['y']), axis=0)
    print(y_pred_bae)
    
    # BNE-Variance.
    bne_vo_joint_samples =get_bne_result(data_dict, moment_mode='variance',
                                      bne_config=bne_config)
    y_pred_bne_vo = np.mean(np.nan_to_num(bne_vo_joint_samples['y']), axis=0) 
    print(y_pred_bne_vo)              
    
    # BNE-Skewness.
    bne_vs_joint_samples = get_bne_result(data_dict, moment_mode='skewness',
                                      bne_config=bne_config)
    y_pred_bne_vs = np.mean(np.nan_to_num(bne_vs_joint_samples['y']), axis=0)
    print(y_pred_bne_vs)

    # save the rmse for each fold
    rmse_bma_mean.append(rmse(Y_te, y_pred_bma_mean))
    rmse_bma.append(rmse(Y_te, y_pred_bma))
    rmse_bae.append(rmse(Y_te, y_pred_bae))
    rmse_bne_vo.append(rmse(Y_te, y_pred_bne_vo))
    rmse_bne_vs.append(rmse(Y_te, y_pred_bne_vs))


# save rmse average among folds
average_metrics = np.mean([rmse_bma_mean, rmse_bma, rmse_bae, rmse_bne_vo, rmse_bne_vs], axis=1)
print(average_metrics)

with open('rmse_bne.txt', 'a') as f:
    f.write('\n')
    #f.write(''.join(str(bma_gp_lengthscale)+ " "+str(bma_gp_l2_regularizer) + " "+ str(np.mean(rmse_bma))+ " "+str(np.std(rmse_bma))))
    f.write(''.join(str(bne_gp_lengthscale)+ " "+str(bne_gp_l2_regularizer) + " "+ str(np.mean(rmse_bma_mean))+ " "+ str(np.mean(rmse_bma))+ " "+ str(np.mean(rmse_bae))+ " "+ str(np.mean(rmse_bne_vo))+ " "+ str(np.mean(rmse_bne_vs))))
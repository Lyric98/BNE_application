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


# In[3]:


# BMA parameters.
y_noise_std = 0.01  # Note: Changed from 0.1 # @param
bma_gp_lengthscale = opts.ls # @param
bma_gp_l2_regularizer = opts.l2 # @param

bma_n_samples_train = 100 # @param
bma_n_samples_eval = 250 # @param
bma_n_samples_test = 250 # @param
bma_seed = 0 # @param


# In[4]:


# BNE parameters.
bne_gp_lengthscale = .05 # 5. # @param
bne_gp_l2_regularizer = 1. # 15 # @param
bne_variance_prior_mean = -2.5 # @param
bne_skewness_prior_mean = -2.5 # @param
bne_seed = 0 # @param


# ### Read training/prediction data

# In[5]:


training_eastMA = pd.read_csv('../data/training_dataset/training_eastMA.csv')
training_eastMA_noMI = training_eastMA[:51]
training_eastMA_folds = pd.read_csv('../data/training_dataset/training_eastMA_folds.csv')
base_model_predictions_eastMA = pd.read_csv('../data/prediction_dataset/base_model_predictions_eastMA.csv')

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
#base_preds_test


# Assemble into configs.
bma_model_config = DEFAULT_GP_CONFIG.copy()
map_config = DEFAULT_MAP_CONFIG.copy()
mcmc_config = DEFAULT_MCMC_CONFIG.copy()

bma_model_config.update(dict(lengthscale=bma_gp_lengthscale,
                             l2_regularizer=bma_gp_l2_regularizer,
                             y_noise_std=y_noise_std))

map_config.update(dict(learning_rate=map_step_size,
                       num_steps=map_num_steps))

mcmc_config.update(dict(step_size=mcmc_step_size, 
                        num_steps=mcmc_num_steps,
                       burnin=mcmc_burnin,
                       nchain=mcmc_nchain,
                       debug_mode=False))


# ### Check 10-fold Random Cross Validation RMSE


ref_model = LinearRegression()
kf = KFold(n_splits=10, random_state=bma_seed, shuffle=True) 

rmse_lr = []
rmse_bma = []

for train_index, test_index in kf.split(X_train1):
      #print("Train:", train_index, "Validation:",test_index)
      X_tr, X_te = X_train1[train_index], X_train1[test_index] 
      Y_tr, Y_te = Y_train[train_index], Y_train[test_index]
      
      # Ref: linear regression
      ref_model.fit(X_tr, Y_tr)
      Y_pred = ref_model.predict(X_te)
      rmse_lr.append(rmse(Y_te, Y_pred))
      

      base_preds_tr, base_preds_te = base_preds_train.numpy()[train_index], base_preds_train.numpy()[test_index]
      print(X_tr.shape, X_te.shape, Y_tr.shape, Y_te.shape, base_preds_tr.shape, base_preds_te.shape)

      # build model & run MCMC
      bma_prior, bma_gp_config = bma_dist(X_tr, 
                                    base_preds_tr, 
                                    **bma_model_config)

      bma_model_config.update(bma_gp_config)


      bma_gp_w_samples = run_posterior_inference(model_dist=bma_prior, 
                                           model_config=bma_model_config,
                                           Y=Y_tr, 
                                           map_config=map_config,
                                           mcmc_config=mcmc_config)


      bma_joint_samples = make_bma_samples(X_te, None, base_preds_te, 
                                     bma_weight_samples=bma_gp_w_samples[0],
                                     bma_model_config=bma_model_config,
                                     n_samples=bma_n_samples_eval, 
                                     seed=bne_seed,
                                     y_samples_only=False)
      y_pred = bma_joint_samples['y']
      y_pred = tf.reduce_mean(y_pred, axis=0)

      rmse_bma.append(rmse(Y_te, y_pred))

print(rmse_lr, rmse_bma)

print("RMSE LR: ", np.mean(rmse_lr), np.std(rmse_lr))
print("RMSE BMA: ", np.mean(rmse_bma), np.std(rmse_bma))


print(bma_model_config)


# ### Check 10-fold Stratified Cross Validation RMSE

print("RMSE LR: ", np.mean(rmse_lr), np.std(rmse_lr))
print("RMSE BMA: ", np.mean(rmse_bma), np.std(rmse_bma))


lr_s = ['', str(np.mean(rmse_lr)), str(np.std(rmse_lr))]
bma_s = [ str(np.mean(rmse_bma)), str(np.std(rmse_bma))]


with open('bma_cos_rmse.txt', 'a') as f:
    f.write('\n')
    f.write(''.join(str(bma_gp_lengthscale)+ " "+str(bma_gp_l2_regularizer) + " "+ str(np.mean(rmse_bma))+ " "+str(np.std(rmse_bma))))
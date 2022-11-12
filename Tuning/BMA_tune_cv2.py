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




ref_model = LinearRegression()
kf = KFold(n_splits=10, random_state=bma_seed, shuffle=True) 

rmse_lr = []
rmse_bma = []

ref_model = LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=0)

rmse_lr = []
rmse_bma = []
rmse_gam = []

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

      means_tr_mcmc, X_tr_mcmc, Y_tr_mcmc = make_bma_samples(
        X_tr, Y_tr, base_preds_tr, 
        bma_weight_samples=bma_gp_w_samples[0],
        bma_model_config=bma_model_config,
        n_samples=bma_n_samples_train,
        seed=bma_seed, 
        prepare_mcmc_training=True)

      means_te_mcmc = make_bma_samples(
        X_te, None, base_preds_te, 
        bma_weight_samples=bma_gp_w_samples[0],
        bma_model_config=bma_model_config,
        n_samples=bma_n_samples_test,
        seed=bma_seed)

    # Construct posterior sampler.
      bne_prior, bne_gp_config = bne_model_dist(
          inputs=X_tr_mcmc,
          mean_preds=means_tr_mcmc,
          **bne_model_config)

      bne_model_config.update(bne_gp_config)
      print(f'prior model graph: {bne_prior.resolve_graph()}')

      #Estimates GP weight posterior using MCMC.
      bne_gp_w_samples = run_posterior_inference(model_dist=bne_prior,
                                           model_config=bne_gp_config,
                                           Y=Y_tr_mcmc,
                                           map_config=map_config,
                                           mcmc_config=mcmc_config,
                                           initialize_from_map=True)
# Generates the posterior sample for all model parameters. 
      bne_joint_samples = make_bne_samples(X_te,
                                     mean_preds=means_te_mcmc,
                                     bne_model_config=bne_model_config,
                                     bne_weight_samples=bne_gp_w_samples[0],
                                     seed=bne_seed)


    #   bma_joint_samples = make_bma_samples(X_te, None, base_preds_te, 
    #                                  bma_weight_samples=bma_gp_w_samples[0],
    #                                  bma_model_config=bma_model_config,
    #                                  n_samples=bma_n_samples_eval, 
    #                                  seed=bne_seed,
    #                                  y_samples_only=False)
      y_pred = bne_joint_samples['y']
      print(y_pred.shape)
      y_pred = tf.reduce_mean(y_pred, axis=0)
      print(y_pred, Y_te)
      print(rmse(Y_te, y_pred))

      rmse_bma.append(rmse(Y_te, y_pred))

print(rmse_lr, rmse_bma)

print("RMSE LR: ", np.mean(rmse_lr), np.std(rmse_lr))
print("RMSE BMA: ", np.mean(rmse_bma), np.std(rmse_bma))


# bma_prior, bma_gp_config = bma_dist(X_train1, 
#                                     base_preds_train, 
#                                     **bma_model_config)

# bma_model_config.update(bma_gp_config)

# # Check if the model graph is specified correctly.
# bma_prior.resolve_graph()


# print(bma_model_config)


# bma_gp_w_samples = run_posterior_inference(model_dist=bma_prior, 
#                                            model_config=bma_model_config,
#                                            Y=Y_train, 
#                                            map_config=map_config,
#                                            mcmc_config=mcmc_config)


# bma_joint_samples = make_bma_samples(X_test1, None, base_preds_test, 
#                                      bma_weight_samples=bma_gp_w_samples[0],
#                                      bma_model_config=bma_model_config,
#                                      n_samples=bma_n_samples_eval, 
#                                      seed=bne_seed,
#                                      y_samples_only=False)



# # Construct data from BMA samples, shapes (num_samples * num_data, ...)
# means_train_mcmc, X_train_mcmc, Y_train_mcmc = make_bma_samples(
#     X_train1, Y_train, base_preds_train, 
#     bma_weight_samples=bma_gp_w_samples[0],
#     bma_model_config=bma_model_config,
#     n_samples=bma_n_samples_train,
#     seed=bma_seed, 
#     prepare_mcmc_training=True)

# # Mean samples based on test data, shape (num_samples, num_data, num_output).
# # It is used to generate final examples in `make_bne_samples()`.
# means_test_mcmc = make_bma_samples(
#     X_test1, None, base_preds_test, 
#     bma_weight_samples=bma_gp_w_samples[0],
#     bma_model_config=bma_model_config,
#     n_samples=bma_n_samples_test,
#     seed=bma_seed)


### Define Model & Run MCMC

# # # BNE GP Configs.
# # lengthscale = 1. # @param
# # l2_regularizer = 10. # @param






# surface_pred_bne_vo = {k: np.mean(np.nan_to_num(bne_joint_samples[k]), axis=0) for k in ('y', 'mean_original', 'resid')}
# surface_var_bne_vo = {k: np.var(np.nan_to_num(bne_joint_samples[k]), axis=0) for k in ('y', 'mean_original', 'resid')}


# ## Basic Plots


#_DATA_ADDR_PREFIX = "./example/data"
# Tuning Parameters
# BMA_lenthscale = bma_gp_lengthscale
# #BNE_lenthscale = bne_gp_lengthscale
# BMA_L2 = bma_gp_l2_regularizer
# #BNE_L2 = bne_gp_l2_regularizer
# _SAVE_ADDR_PREFIX = "./pic/BMA_lenthscale_{}_L2_{}".format(BMA_lenthscale, BMA_L2)

# path=_SAVE_ADDR_PREFIX
# isExists=os.path.exists(path) #判断路径是否存在，存在则返回true

# if not isExists:
#     os.makedirs(path)




# coordinate = np.asarray(base_model_predictions_eastMA[["lon", "lat"]].values.tolist()).astype(np.float32)
# monitors = np.asarray(training_eastMA_noMI[["lon", "lat"]].values.tolist()).astype(np.float32)
# base_model_names = ["pred_av", "pred_gs", "pred_caces"]

# base_model_predictions_eastMA[["pred_av", "pred_gs", "pred_caces"]] = np.where(np.isnan(base_model_predictions_eastMA[["pred_av", "pred_gs", "pred_caces"]]), 0, base_model_predictions_eastMA[["pred_av", "pred_gs", "pred_caces"]])
# color_norm_base = make_color_norm(
#     base_model_predictions_eastMA[["pred_av", "pred_gs", "pred_caces"]],   
#     method="percentile")



# ### 2. The predictive surface of individual BNE gp weights


# bma_ensemble_weights = bma_joint_samples['ensemble_weights']
# ensemble_weights_val = tf.reduce_mean(bma_ensemble_weights, axis=0)

# weights_dict = {
#     "AV": ensemble_weights_val[:, 0],
#     "GS": ensemble_weights_val[:,1],
#     "CACES": ensemble_weights_val[:,2],
# }
# #weights_dict
# color_norm_weights = make_color_norm(
#     list(weights_dict.values()),#[2],   
#     method="percentile")



# ensemble_weights_var = np.var(bma_ensemble_weights, axis=0)
# weights_var_dict = {
#     "AV": ensemble_weights_var[:, 0],
#     "GS": ensemble_weights_var[:,1],
#     "CACES": ensemble_weights_var[:,2],
# }
# #weights_dict
# color_norm_weights_var = make_color_norm(
#     list(weights_var_dict.values()),#[0],   
#     method="percentile")


# base_model_names = ["AV", "GS", "CACES"]
# for base_model_name in base_model_names:
#     save_name = os.path.join(_SAVE_ADDR_PREFIX,
#                              'base_weights_{}_bmals_{}_r_{}.png'.format(
#                                  base_model_name, bma_gp_lengthscale,  bma_gp_l2_regularizer))
    
#     posterior_heatmap_2d(weights_dict[base_model_name], coordinate,
#                          monitors,
#                          cmap='viridis',
#                          norm=color_norm_weights, 
#                          #norm_method="percentile",
#                          #save_addr='')
#                          save_addr=save_name)


# # plot weights' variance
# for base_model_name in base_model_names:
#     save_name = os.path.join(_SAVE_ADDR_PREFIX,
#                              'base_wvar_{}_bmals_{}_r_{}.png'.format(
#                                  base_model_name, bma_gp_lengthscale,  bma_gp_l2_regularizer))
    
#     posterior_heatmap_2d(weights_var_dict[base_model_name], coordinate,
#                          monitors,
#                          cmap='viridis',
#                          norm=color_norm_weights_var, 
#                          #norm_method="percentile",
#                          #save_addr='')
#                          save_addr=save_name)


# ### 3. The predictive surface of Y_mean, residual process, and Y_mean + residual process.

    

# # BNE vo
# color_norm_pred = make_color_norm(
#     #np.nan_to_num(list(surface_pred_bae.values())[:2][0]),
#     list(surface_pred_bne_vo.values())[:2],  
#     method="percentile")

# color_norm_pred_r = make_color_norm(
#     #np.nan_to_num(list(surface_pred_bae.values())[2:]),
#     list(surface_pred_bne_vo.values())[2],  
#     method="residual_percentile")


# for name, value in surface_pred_bne_vo.items():
#     save_name = os.path.join(_SAVE_ADDR_PREFIX,
#                             'BNEvo_{}_bma:ls_{}_r_{}_bne:ls_{}_r_{}.png'.format(
#                                 name, bma_gp_lengthscale,  bma_gp_l2_regularizer,
#                                 bne_gp_lengthscale, bne_gp_l2_regularizer))

#     value = np.where(np.isnan(value), 0, value)
#     color_norm = posterior_heatmap_2d(value, X=coordinate, X_monitor=monitors,
#                                                   cmap='RdYlGn_r',
#                     norm= color_norm_pred_r if name=='resid' else color_norm_pred,
#                                       save_addr=save_name)
                    #norm_method="percentile",
                    #save_addr='')
    


# In[ ]:


# # BNE v+s
# color_norm_pred = make_color_norm(
#     #np.nan_to_num(list(surface_pred_bae.values())[:2][0]),
#     list(surface_pred_bne_vs.values())[:2],  
#     method="percentile")

# color_norm_pred_r = make_color_norm(
#     #np.nan_to_num(list(surface_pred_bae.values())[2:]),
#     list(surface_pred_bne_vs.values())[2],  
#     method="residual_percentile")


# for name, value in surface_pred_bne_vs.items():
#     save_name = os.path.join(_SAVE_ADDR_PREFIX,
#                             'BNEvs_{}_bma:ls_{}_r_{}_bne:ls_{}_r_{}.png'.format(name, bma_gp_lengthscale, 
#                                 bma_gp_l2_regularizer, bne_gp_lengthscale, bne_gp_l2_regularizer))

#     value = np.where(np.isnan(value), 0, value)
#     color_norm = posterior_heatmap_2d(value, X=coordinate, X_monitor=monitors,
#                                                   cmap='RdYlGn_r',
#                     norm= color_norm_pred_r if name=='resid' else color_norm_pred,
#                     #norm_method="percentile",
#                     #save_addr='')
#                 save_addr=save_name)
    


# ### 4.The predictive variance of Y_mean, residual process, and Y.

# In[ ]:


# # BAE
# color_norm_var = make_color_norm(
#     list(surface_var_bae.values())[:2], 
#     method="percentile")

# color_norm_var_r = make_color_norm(
#     list(surface_var_bae.values())[2], 
#     method="percentile")


# for name, value in surface_var_bae.items():
#     save_name = os.path.join(_SAVE_ADDR_PREFIX,
#                             'var_BAE_{}_bma:ls_{}_r_{}_bne:ls_{}_r_{}.png'.format(
#                                 name, bma_gp_lengthscale,  bma_gp_l2_regularizer,
#                                 bne_gp_lengthscale, bne_gp_l2_regularizer))
#     #value = np.where(np.isnan(value), 0, value)
#     color_norm = posterior_heatmap_2d(value, X=coordinate, X_monitor=monitors,
#                                 cmap='inferno_r',
#                                 norm= color_norm_var_r if name=='resid' else color_norm_var,
#                                 #norm_method="percentile",
#                                 save_addr=save_name)


# In[ ]:


# # BNE vo
# color_norm_var = make_color_norm(
#     list(surface_var_bne_vo.values())[:2], 
#     method="percentile")

# color_norm_var_r = make_color_norm(
#     list(surface_var_bne_vo.values())[2], 
#     method="percentile")


# for name, value in surface_var_bne_vo.items():
#     save_name = os.path.join(_SAVE_ADDR_PREFIX,
#                             'var_BNEvo_{}_bma:ls_{}_r_{}_bne:ls_{}_r_{}.png'.format(
#                                 name, bma_gp_lengthscale,  bma_gp_l2_regularizer,
#                                 bne_gp_lengthscale, bne_gp_l2_regularizer))
#     #value = np.where(np.isnan(value), 0, value)
#     color_norm = posterior_heatmap_2d(value, X=coordinate, X_monitor=monitors,
#                                 cmap='inferno_r',
#                                 norm= color_norm_var_r if name=='resid' else color_norm_var,
#                                 #norm_method="percentile",
#                                 save_addr=save_name)


# In[ ]:


# # BNE v+s
# color_norm_var = make_color_norm(
#     list(surface_var_bne_vs.values())[:2], 
#     method="percentile")

# color_norm_var_r = make_color_norm(
#     list(surface_var_bne_vs.values())[2], 
#     method="percentile")


# for name, value in surface_var_bne_vs.items():
#     save_name = os.path.join(_SAVE_ADDR_PREFIX,
#                             'var_BNEvs_{}_bma:ls_{}_r_{}_bne:ls_{}_r_{}.png'.format(
#                                 name, bma_gp_lengthscale,  bma_gp_l2_regularizer,
#                                 bne_gp_lengthscale, bne_gp_l2_regularizer))
#     #value = np.where(np.isnan(value), 0, value)
#     color_norm = posterior_heatmap_2d(value, X=coordinate, X_monitor=monitors,
#                                 cmap='inferno_r',
#                                 norm= color_norm_var_r if name=='resid' else color_norm_var,
#                                 #norm_method="percentile",
#                                 save_addr=save_name)


with open('rmse_cos.txt', 'a') as f:
    f.write('\n')
    f.write(''.join(str(bma_gp_lengthscale)+ " "+str(bma_gp_l2_regularizer) + " "+ str(np.mean(rmse_bma))+ " "+str(np.std(rmse_bma))))
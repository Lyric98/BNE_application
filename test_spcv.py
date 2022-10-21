import numpy as np

def rmse(y_obs, y_pred):
    return np.sqrt(np.mean((y_obs - y_pred) ** 2))
#bma_mcmc_rmse = rmse(means_train_mcmc, Y_train_mcmc)


def compute_cv_rmse(X_train_mcmc, Y_train_mcmc, X_test_mcmc, Y_test_mcmc):

    # Construct posterior sampler.
    bne_prior, bne_gp_config = bne_model_dist(
        inputs=X_train_mcmc,
        mean_preds=means_train_mcmc,
        **bne_model_config)

    bne_model_config.update(bne_gp_config)

    # Estimates GP weight posterior using MCMC.
    bne_gp_w_samples = run_posterior_inference(model_dist=bne_prior,
                                           model_config=bne_gp_config,
                                           Y=Y_train_mcmc,
                                           map_config=map_config,
                                           mcmc_config=mcmc_config,
                                           initialize_from_map=True)

    # Generates the posterior sample for all model parameters. 
    bne_joint_samples = make_bne_samples(X_test_mcmc,
                                     mean_preds=Y_test_mcmc,
                                     bne_model_config=bne_model_config,
                                     bne_weight_samples=bne_gp_w_samples[0],
                                     seed=bne_seed)

    y_pred = bne_joint_samples['y']

    means_pred = np.mean(y_pred, axis=0)
    rmse = np.sqrt(np.mean((means_pred - Y_test_mcmc)**2))

    return rmse






## @title Simulation: compute_metrics
def compute_metrics(data_dict, model_name, q_true=None, ind_ids=None, num_sample=None):
  if q_true is None:
    q_true = np.array(
        [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
         0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975])

  if ind_ids is None:
    # Find IDs of in-domain test data via range comparison 
    # between X_train and X_test.
    X_train_min = np.min(data_dict['X_train'], axis=0)
    X_train_max = np.max(data_dict['X_train'], axis=0)

    test_ids_greater_than_min = np.all(
        data_dict['X_test'] > X_train_min, axis=-1)
    test_ids_less_than_max = np.all(
        data_dict['X_test'] < X_train_max, axis=-1)

    ind_ids = np.where(
        np.logical_and(test_ids_greater_than_min, test_ids_less_than_max))[0]

  samples = data_dict[f'{model_name}_samples']
  means_true = data_dict['mean_test']
  y_test = data_dict['Y_test']

  if num_sample is not None:
    samples = samples[:num_sample]

  means_pred = np.mean(samples, axis=0)
  stds_pred = np.std(samples, axis=0)
  quantile_pred = np.quantile(samples, q=q_true, axis=0)

  # Compute in-domain metrics.
  nll_ind = np.mean(
      ((means_pred[ind_ids] - means_true[ind_ids])/stds_pred[ind_ids])**2 + 
      np.log(stds_pred[ind_ids]))
  clb_ind = np.mean(
      ((means_pred[ind_ids] - means_true[ind_ids])/stds_pred[ind_ids])**2)
  shp_ind = np.mean(np.log(stds_pred[ind_ids]))
  mse_ind = np.mean(
      (means_pred[ind_ids] - means_true[ind_ids])**2) / np.var(means_true[ind_ids])

  q_pred_ind = np.mean(y_test[ind_ids] < quantile_pred[:, ind_ids], axis=(1, 2))
  ece_ind = np.mean((q_pred_ind - q_true)**2)
  cov_prob_95_ind = q_pred_ind[-1] - q_pred_ind[0]
  cov_prob_90_ind = q_pred_ind[-2] - q_pred_ind[1]
  cov_prob_85_ind = q_pred_ind[-3] - q_pred_ind[2]
  cov_prob_80_ind = q_pred_ind[-4] - q_pred_ind[3]

  # Compute all-domain (ind + ood) metrics.
  nll_all = np.mean(((means_pred - means_true)/stds_pred)**2 + np.log(stds_pred))
  clb_all = np.mean(((means_pred - means_true)/stds_pred)**2)
  shp_all = np.mean(np.log(stds_pred))
  mse_all = np.mean((means_pred - means_true)**2) / np.var(means_true)

  q_pred_all = np.mean(y_test < quantile_pred, axis=(1, 2))
  ece_all = np.mean((q_pred_all - q_true)**2)
  cov_prob_95_all = q_pred_all[-1] - q_pred_all[0]
  cov_prob_90_all = q_pred_all[-2] - q_pred_all[1]
  cov_prob_85_all = q_pred_all[-3] - q_pred_all[2]
  cov_prob_80_all = q_pred_all[-4] - q_pred_all[3]

  return (mse_ind, nll_ind, clb_ind, shp_ind, ece_ind, cov_prob_95_ind, cov_prob_90_ind, cov_prob_85_ind, cov_prob_80_ind,
          mse_all, nll_all, clb_all, shp_all, ece_all, cov_prob_95_all, cov_prob_90_all, cov_prob_85_all, cov_prob_80_all)
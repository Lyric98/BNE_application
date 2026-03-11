"""Core conformal prediction methods for BNE.

All methods are post-processing on BNE posterior samples — no changes
to the BNE training pipeline are needed.
"""

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# 1. BNE Credible Intervals (baseline — no conformal correction)
# ---------------------------------------------------------------------------

def bne_credible_interval(bne_samples, alpha=0.05):
    """Extract posterior credible intervals from BNE MCMC samples.

    Args:
        bne_samples: (num_mcmc, n) array of posterior predictive samples.
            Typically obtained from make_bne_samples()['y'] squeezed.
        alpha: Miscoverage level (default 0.05 for 95% intervals).

    Returns:
        lower: (n,) array of lower bounds.
        upper: (n,) array of upper bounds.
    """
    bne_samples = np.asarray(bne_samples)
    lower = np.percentile(bne_samples, 100 * (alpha / 2), axis=0)
    upper = np.percentile(bne_samples, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


# ---------------------------------------------------------------------------
# 2. Split Conformal — standard baseline
# ---------------------------------------------------------------------------

def split_conformal(y_cal, y_pred_cal, y_pred_test, alpha=0.05):
    """Standard split conformal with absolute residual scores.

    Uses the BNE posterior mean as point prediction, then conformalize
    with absolute residual nonconformity scores.

    Args:
        y_cal: (n_cal,) array of calibration true values.
        y_pred_cal: (n_cal,) array of point predictions on calibration set.
        y_pred_test: (n_test,) array of point predictions on test set.
        alpha: Miscoverage level.

    Returns:
        lower: (n_test,) array of lower bounds.
        upper: (n_test,) array of upper bounds.
    """
    y_cal = np.asarray(y_cal).ravel()
    y_pred_cal = np.asarray(y_pred_cal).ravel()
    y_pred_test = np.asarray(y_pred_test).ravel()

    # Nonconformity scores: absolute residuals
    scores = np.abs(y_cal - y_pred_cal)

    # Conformal quantile with finite-sample correction
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(scores, q_level)

    lower = y_pred_test - q_hat
    upper = y_pred_test + q_hat

    return lower, upper


# ---------------------------------------------------------------------------
# 3. Conformalized Quantile Regression (CQR)
# ---------------------------------------------------------------------------

def conformalized_quantile_regression(y_cal, quantiles_cal, quantiles_test,
                                      alpha=0.05):
    """CQR using BNE posterior quantiles as the base quantile estimator.

    Following Romano, Patterson, Candes (2019). Uses BNE's posterior
    alpha/2 and 1-alpha/2 quantiles, then conformalize with CQR scores.

    Args:
        y_cal: (n_cal,) array of calibration true values.
        quantiles_cal: (n_cal, 2) array of [lower, upper] quantiles on cal set.
        quantiles_test: (n_test, 2) array of [lower, upper] quantiles on test set.
        alpha: Miscoverage level.

    Returns:
        lower: (n_test,) array of conformalized lower bounds.
        upper: (n_test,) array of conformalized upper bounds.
    """
    y_cal = np.asarray(y_cal).ravel()
    quantiles_cal = np.asarray(quantiles_cal)
    quantiles_test = np.asarray(quantiles_test)

    q_lo_cal = quantiles_cal[:, 0]
    q_hi_cal = quantiles_cal[:, 1]

    # CQR nonconformity scores: max(q_lo - y, y - q_hi)
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

    # Conformal quantile
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    Q = np.quantile(scores, q_level)

    lower = quantiles_test[:, 0] - Q
    upper = quantiles_test[:, 1] + Q

    return lower, upper


# ---------------------------------------------------------------------------
# 4. Spatially-Weighted CQR — KEY NOVELTY
# ---------------------------------------------------------------------------

def _matern32_kernel(dists, lengthscale):
    """Matern-3/2 kernel: k(d) = (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)."""
    r = np.sqrt(3) * dists / lengthscale
    return (1 + r) * np.exp(-r)


def _rbf_kernel(dists, lengthscale):
    """RBF (squared exponential) kernel."""
    return np.exp(-0.5 * (dists / lengthscale) ** 2)


def _exponential_kernel(dists, lengthscale):
    """Exponential (Matern-1/2) kernel."""
    return np.exp(-dists / lengthscale)


_KERNELS = {
    'matern32': _matern32_kernel,
    'rbf': _rbf_kernel,
    'exponential': _exponential_kernel,
}


def spatial_cqr(y_cal, quantiles_cal, quantiles_test, X_cal, X_test,
                alpha=0.05, kernel='matern32', lengthscale=1.0):
    """Kernel-weighted conformalized quantile regression for spatial data.

    Key idea: instead of a single global conformal quantile Q, compute a
    spatially varying Q(x) by weighting calibration scores by their
    spatial proximity to the test point. This gives tighter intervals
    in well-observed regions and wider intervals in sparse/OOD regions.

    Following the locally-weighted conformal framework (Lei & Wasserman, 2014;
    Barber et al., 2023), with spatial kernel weighting.

    Args:
        y_cal: (n_cal,) array of calibration true values.
        quantiles_cal: (n_cal, 2) array of [lower, upper] quantiles on cal set.
        quantiles_test: (n_test, 2) array of [lower, upper] quantiles on test set.
        X_cal: (n_cal, d) array of calibration features/coordinates.
        X_test: (n_test, d) array of test features/coordinates.
        alpha: Miscoverage level.
        kernel: Kernel function name ('matern32', 'rbf', 'exponential').
        lengthscale: Kernel lengthscale parameter.

    Returns:
        lower: (n_test,) array of spatially-adaptive lower bounds.
        upper: (n_test,) array of spatially-adaptive upper bounds.
    """
    y_cal = np.asarray(y_cal).ravel()
    quantiles_cal = np.asarray(quantiles_cal)
    quantiles_test = np.asarray(quantiles_test)
    X_cal = np.asarray(X_cal)
    X_test = np.asarray(X_test)

    if kernel not in _KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {list(_KERNELS.keys())}")
    kernel_fn = _KERNELS[kernel]

    # CQR nonconformity scores on calibration set
    q_lo_cal = quantiles_cal[:, 0]
    q_hi_cal = quantiles_cal[:, 1]
    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)

    n_cal = len(scores)
    n_test = len(X_test)

    # Pairwise distances: (n_test, n_cal)
    dists = cdist(X_test, X_cal, metric='euclidean')
    weights = kernel_fn(dists, lengthscale)  # (n_test, n_cal)

    # For each test point, compute weighted conformal quantile
    lower = np.zeros(n_test)
    upper = np.zeros(n_test)

    for i in range(n_test):
        w = weights[i]
        # Add weight for the "infinity" score of the test point itself
        # (finite-sample correction from Barber et al., 2023)
        w_test = kernel_fn(np.array([0.0]), lengthscale)[0]
        total_weight = w.sum() + w_test

        if total_weight < 1e-12:
            # Fallback: no nearby calibration points -> infinite interval
            lower[i] = -np.inf
            upper[i] = np.inf
            continue

        # Normalize weights
        p = w / total_weight
        p_test = w_test / total_weight

        # Weighted quantile: find smallest Q s.t. sum of weights with score <= Q >= 1-alpha
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_p = p[sorted_idx]
        cumulative_weight = np.cumsum(sorted_p)

        # The test point contributes p_test at score = +inf
        # We need cumulative_weight + p_test >= 1 - alpha at some score
        target = 1 - alpha - p_test
        idx = np.searchsorted(cumulative_weight, target, side='right')
        if idx >= n_cal:
            Q_i = sorted_scores[-1]
        else:
            Q_i = sorted_scores[idx]

        lower[i] = quantiles_test[i, 0] - Q_i
        upper[i] = quantiles_test[i, 1] + Q_i

    return lower, upper


# ---------------------------------------------------------------------------
# Helper: extract BNE quantities for conformal methods
# ---------------------------------------------------------------------------

def extract_bne_predictions(bne_samples_dict, alpha=0.05):
    """Extract point predictions and quantiles from BNE samples dict.

    Args:
        bne_samples_dict: Dict returned by make_bne_samples(), must have key 'y'.
            bne_samples_dict['y'] shape: (num_mcmc, num_data, 1) or (num_mcmc, num_data).
        alpha: Miscoverage level for quantile extraction.

    Returns:
        Dict with keys:
            'mean': (num_data,) posterior mean.
            'median': (num_data,) posterior median.
            'quantiles': (num_data, 2) array of [alpha/2, 1-alpha/2] quantiles.
            'samples': (num_mcmc, num_data) raw samples.
            'cdf_at_obs': None (must be computed separately with y_true).
    """
    y_samples = np.asarray(bne_samples_dict['y'])
    # Squeeze trailing dimension if present
    if y_samples.ndim == 3:
        y_samples = y_samples.squeeze(-1)  # (num_mcmc, num_data)

    mean = np.mean(y_samples, axis=0)
    median = np.median(y_samples, axis=0)
    q_lo = np.percentile(y_samples, 100 * (alpha / 2), axis=0)
    q_hi = np.percentile(y_samples, 100 * (1 - alpha / 2), axis=0)
    quantiles = np.stack([q_lo, q_hi], axis=-1)  # (num_data, 2)

    return {
        'mean': mean,
        'median': median,
        'quantiles': quantiles,
        'samples': y_samples,
        'cdf_at_obs': None,
    }


def compute_cdf_at_obs(bne_samples, y_obs):
    """Compute empirical CDF of BNE posterior at observed values.

    Args:
        bne_samples: (num_mcmc, n) array of posterior predictive samples.
        y_obs: (n,) array of observed values.

    Returns:
        (n,) array of F_hat(y_i | x_i) values.
    """
    bne_samples = np.asarray(bne_samples)
    y_obs = np.asarray(y_obs).ravel()
    return np.mean(bne_samples < y_obs[None, :], axis=0)

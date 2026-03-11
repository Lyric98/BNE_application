"""Evaluation metrics for conformal prediction intervals."""

import numpy as np


def marginal_coverage(y_true, lower, upper):
    """Compute marginal coverage: fraction of y_true within [lower, upper].

    Args:
        y_true: (n,) array of true values.
        lower: (n,) array of lower bounds.
        upper: (n,) array of upper bounds.

    Returns:
        Float coverage rate in [0, 1].
    """
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    return np.mean((y_true >= lower) & (y_true <= upper))


def conditional_coverage(y_true, lower, upper, groups):
    """Compute coverage by group (e.g., spatial region, quantile bin).

    Args:
        y_true: (n,) array of true values.
        lower: (n,) array of lower bounds.
        upper: (n,) array of upper bounds.
        groups: (n,) array of group labels (int or str).

    Returns:
        Dict mapping group label -> coverage rate.
    """
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    groups = np.asarray(groups)
    results = {}
    for g in np.unique(groups):
        mask = groups == g
        results[g] = marginal_coverage(y_true[mask], lower[mask], upper[mask])
    return results


def average_interval_width(lower, upper):
    """Compute mean interval width.

    Args:
        lower: (n,) array of lower bounds.
        upper: (n,) array of upper bounds.

    Returns:
        Float mean width.
    """
    return np.mean(np.asarray(upper) - np.asarray(lower))


def interval_width_by_group(lower, upper, groups):
    """Compute mean interval width by group.

    Args:
        lower: (n,) array of lower bounds.
        upper: (n,) array of upper bounds.
        groups: (n,) array of group labels.

    Returns:
        Dict mapping group label -> mean width.
    """
    lower, upper, groups = np.asarray(lower), np.asarray(upper), np.asarray(groups)
    results = {}
    for g in np.unique(groups):
        mask = groups == g
        results[g] = average_interval_width(lower[mask], upper[mask])
    return results


def coverage_by_quantile(y_true, pred_cdf_values, n_bins=20):
    """ECE-style calibration: bin observations by predicted CDF value.

    For a well-calibrated model, the fraction of observations with
    CDF(y) <= p should be approximately p for all p.

    Args:
        y_true: (n,) array (unused, kept for API consistency).
        pred_cdf_values: (n,) array of F_hat(y_i | x_i) for each observation.
        n_bins: Number of bins for the calibration histogram.

    Returns:
        Dict with keys:
            'bin_edges': (n_bins+1,) array of bin edges.
            'expected': (n_bins,) array of expected coverage per bin (bin midpoints).
            'observed': (n_bins,) array of observed fraction per bin.
            'ece': Expected Calibration Error (mean absolute deviation).
    """
    pred_cdf_values = np.asarray(pred_cdf_values)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    observed = np.zeros(n_bins)
    expected = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (pred_cdf_values >= lo) & (pred_cdf_values < hi)
        expected[i] = (lo + hi) / 2
        if mask.sum() > 0:
            observed[i] = mask.sum() / len(pred_cdf_values)
        else:
            observed[i] = 0.0

    # Normalize observed to be cumulative fractions
    observed_cumulative = np.array([
        np.mean(pred_cdf_values <= edge) for edge in bin_edges[1:]
    ])
    expected_cumulative = bin_edges[1:]
    ece = np.mean(np.abs(observed_cumulative - expected_cumulative))

    return {
        'bin_edges': bin_edges,
        'expected': expected_cumulative,
        'observed': observed_cumulative,
        'ece': ece,
    }


def negative_log_likelihood(y_true, bne_samples):
    """Compute mean NLL using BNE posterior samples via KDE.

    Args:
        y_true: (n,) array of true values.
        bne_samples: (num_mcmc, n) array of posterior predictive samples.

    Returns:
        Float mean NLL across test points.
    """
    y_true = np.asarray(y_true)
    bne_samples = np.asarray(bne_samples)
    n = len(y_true)
    nll = 0.0
    for i in range(n):
        samples_i = bne_samples[:, i]
        # Gaussian KDE estimate of log p(y_i)
        std = np.std(samples_i)
        if std < 1e-10:
            std = 1e-10
        bandwidth = 1.06 * std * len(samples_i) ** (-1 / 5)  # Silverman's rule
        log_densities = -0.5 * ((y_true[i] - samples_i) / bandwidth) ** 2 \
                        - np.log(bandwidth) - 0.5 * np.log(2 * np.pi)
        # Log-sum-exp for numerical stability
        max_log = np.max(log_densities)
        log_pdf = max_log + np.log(np.mean(np.exp(log_densities - max_log)))
        nll -= log_pdf
    return nll / n

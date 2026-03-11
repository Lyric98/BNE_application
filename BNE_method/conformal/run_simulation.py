"""Conformalized BNE — Simulation Prototype (HPC script).

Usage:
    python run_simulation.py --n_train 500 --seed 42
    python run_simulation.py --n_train 250 --seed 0

Outputs saved to: conformal/results/sim_n{n_train}_s{seed}/
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for HPC
import matplotlib.pyplot as plt

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'case_study_results'))

from wrapper_functions import (
    generate_data_2d, run_base_models, run_bma_model, run_bne_model,
)
from conformal.conformal_bne import (
    split_conformal, conformalized_quantile_regression, spatial_cqr,
    bne_credible_interval, extract_bne_predictions, compute_cdf_at_obs,
)
from conformal.metrics import (
    marginal_coverage, conditional_coverage, average_interval_width,
    interval_width_by_group, coverage_by_quantile, negative_log_likelihood,
)


def run_experiment(n_train, seed, alpha=0.05, outdir=None):
    """Run full conformal BNE simulation experiment."""

    if outdir is None:
        outdir = os.path.join(SCRIPT_DIR, 'results', f'sim_n{n_train}_s{seed}')
    os.makedirs(outdir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Conformal BNE Simulation: N_train={n_train}, seed={seed}, alpha={alpha}")
    print(f"Output: {outdir}")
    print(f"{'='*60}\n")

    np.random.seed(seed)
    N_BASE = 100
    N_TEST = 2000

    # ---- 1. Generate data ----
    t0 = time.time()
    print("[1/5] Generating 2D data...")
    X_base, X_train_full, X_test, Y_base, Y_train_full, Y_test, mean_test = \
        generate_data_2d(N_train=n_train, N_base=N_BASE, N_test=N_TEST, seed=seed)

    # Split into train (70%) + calibration (30%)
    n_full = len(X_train_full)
    n_tr = int(n_full * 0.7)
    idx = np.random.permutation(n_full)
    X_train, Y_train = X_train_full[idx[:n_tr]], Y_train_full[idx[:n_tr]]
    X_cal, Y_cal = X_train_full[idx[n_tr:]], Y_train_full[idx[n_tr:]]
    n_cal = len(X_cal)

    # Combined prediction set
    X_pred = np.concatenate([X_cal, X_test], axis=0)
    Y_pred = np.concatenate([Y_cal, Y_test], axis=0)

    # Region labels
    in_domain_mask = (np.abs(X_test[:, 0]) <= np.pi) & (np.abs(X_test[:, 1]) <= np.pi)
    test_region = np.where(in_domain_mask, 'in-domain', 'OOD')

    print(f"  Train: {n_tr}, Cal: {n_cal}, Test: {N_TEST}")
    print(f"  In-domain: {in_domain_mask.sum()}, OOD: {(~in_domain_mask).sum()}")
    print(f"  Data generation: {time.time()-t0:.1f}s\n")

    # ---- 2. Base models ----
    t0 = time.time()
    print("[2/5] Training base models...")
    base_preds_train, base_preds_pred, kernel_names = run_base_models(
        X_base, X_train, X_pred, Y_base, Y_train, Y_pred
    )
    print(f"  Kernels: {kernel_names}")
    print(f"  Base models: {time.time()-t0:.1f}s\n")

    # ---- 3. BMA ----
    t0 = time.time()
    print("[3/5] Training BMA...")
    bma_joint_samples, X_train_mcmc, Y_train_mcmc, means_train_mcmc, means_pred_mcmc = \
        run_bma_model(
            X_train, X_pred, Y_train,
            base_preds_train, base_preds_pred,
            gp_lengthscale=1., gp_l2_regularizer=0.1, y_noise_std=0.1,
            map_step_size=0.1, map_num_steps=10_000,
            mcmc_step_size=0.1, mcmc_num_steps=10_000,
            mcmc_nchain=10, mcmc_burnin=2_500,
            mcmc_initialize_from_map=True,
            n_samples_eval=1000, n_samples_train=100, n_samples_test=200,
            return_mcmc_examples=True, seed=seed,
        )
    print(f"  BMA: {time.time()-t0:.1f}s\n")

    # ---- 4. BNE ----
    t0 = time.time()
    print("[4/5] Training BNE (variance mode)...")
    bne_samples_dict = run_bne_model(
        X_train=X_train_mcmc, Y_train=Y_train_mcmc,
        X_test=X_pred,
        base_model_samples_train=means_train_mcmc,
        base_model_samples_test=means_pred_mcmc,
        moment_mode='variance',
        gp_lengthscale=1., gp_l2_regularizer=10.,
        map_step_size=5e-3, map_num_steps=10_000,
        mcmc_step_size=1e-2, mcmc_num_steps=10_000,
        mcmc_burnin=2_500, mcmc_nchain=10,
        mcmc_initialize_from_map=True, seed=seed,
    )
    print(f"  BNE: {time.time()-t0:.1f}s\n")

    # ---- 5. Conformal methods + evaluation ----
    t0 = time.time()
    print("[5/5] Computing conformal intervals & evaluation...")

    bne_preds = extract_bne_predictions(bne_samples_dict, alpha=alpha)
    bne_mean_cal = bne_preds['mean'][:n_cal]
    bne_mean_test = bne_preds['mean'][n_cal:]
    bne_quantiles_cal = bne_preds['quantiles'][:n_cal]
    bne_quantiles_test = bne_preds['quantiles'][n_cal:]
    bne_samples_test = bne_preds['samples'][:, n_cal:]

    # All methods
    methods = {}
    methods['BNE_Credible'] = bne_credible_interval(bne_samples_test, alpha=alpha)
    methods['Split_CP'] = split_conformal(Y_cal, bne_mean_cal, bne_mean_test, alpha=alpha)
    methods['CQR'] = conformalized_quantile_regression(
        Y_cal, bne_quantiles_cal, bne_quantiles_test, alpha=alpha)

    # Spatial CQR with multiple lengthscales
    for ls in [0.5, 1.0, 2.0, 5.0]:
        methods[f'Spatial_CQR_l{ls}'] = spatial_cqr(
            Y_cal, bne_quantiles_cal, bne_quantiles_test,
            X_cal, X_test, alpha=alpha, kernel='matern32', lengthscale=ls)

    # Build results table
    rows = []
    for name, (lo, hi) in methods.items():
        cov_all = marginal_coverage(Y_test, lo, hi)
        width_all = average_interval_width(lo, hi)
        cov_cond = conditional_coverage(Y_test, lo, hi, test_region)
        width_cond = interval_width_by_group(lo, hi, test_region)
        rows.append({
            'method': name,
            'n_train': n_train,
            'seed': seed,
            'alpha': alpha,
            'coverage_all': cov_all,
            'coverage_in_domain': cov_cond.get('in-domain', np.nan),
            'coverage_ood': cov_cond.get('OOD', np.nan),
            'width_all': width_all,
            'width_in_domain': width_cond.get('in-domain', np.nan),
            'width_ood': width_cond.get('OOD', np.nan),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n{df.to_string(index=False)}")

    # NLL
    nll = negative_log_likelihood(Y_test, bne_samples_test)
    print(f"\nBNE Posterior NLL: {nll:.4f}")

    # Calibration diagnostics
    cdf_vals = compute_cdf_at_obs(bne_samples_test, Y_test)
    cal_diag = coverage_by_quantile(Y_test, cdf_vals, n_bins=20)
    print(f"BNE ECE: {cal_diag['ece']:.4f}")

    # Save supplementary data
    np.savez(os.path.join(outdir, 'data.npz'),
             X_test=X_test, Y_test=Y_test,
             X_cal=X_cal, Y_cal=Y_cal,
             test_region=test_region,
             cdf_vals=cdf_vals)

    print(f"  Evaluation: {time.time()-t0:.1f}s\n")

    # ---- Plots ----
    _make_plots(outdir, X_test, Y_test, methods, test_region, alpha, cal_diag)
    print(f"Done. Results saved to {outdir}")
    return df


def _make_plots(outdir, X_test, Y_test, methods, test_region, alpha, cal_diag):
    """Generate and save all figures."""

    # Select main 4 methods for 2x2 plots
    main_methods = {k: v for k, v in methods.items()
                    if k in ['BNE_Credible', 'Split_CP', 'CQR', 'Spatial_CQR_l2.0']}

    # --- Fig 1: Interval width maps ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (name, (lo, hi)) in zip(axes.ravel(), main_methods.items()):
        widths = hi - lo
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=widths, s=3,
                        cmap='viridis', alpha=0.7)
        plt.colorbar(sc, ax=ax, label='Interval width')
        rect = plt.Rectangle((-np.pi, -np.pi), 2*np.pi, 2*np.pi,
                              fill=False, edgecolor='red', linewidth=1.5, linestyle='--')
        ax.add_patch(rect)
        cov = marginal_coverage(Y_test, lo, hi)
        avg_w = average_interval_width(lo, hi)
        ax.set_title(f'{name}\nCov={cov:.3f}, Width={avg_w:.2f}')
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
    plt.suptitle(f'Interval Width Maps (target: {1-alpha:.0%})', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_width_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Fig 2: Coverage maps ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (name, (lo, hi)) in zip(axes.ravel(), main_methods.items()):
        covered = (Y_test >= lo) & (Y_test <= hi)
        ax.scatter(X_test[covered, 0], X_test[covered, 1], c='steelblue', s=3, alpha=0.4, label='Covered')
        ax.scatter(X_test[~covered, 0], X_test[~covered, 1], c='red', s=10, alpha=0.8, label='Missed', marker='x')
        rect = plt.Rectangle((-np.pi, -np.pi), 2*np.pi, 2*np.pi,
                              fill=False, edgecolor='black', linewidth=1.5, linestyle='--', label='Train region')
        ax.add_patch(rect)
        cov = marginal_coverage(Y_test, lo, hi)
        ax.set_title(f'{name} (Cov={cov:.3f})')
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        ax.legend(loc='lower right', fontsize=8)
    plt.suptitle('Coverage Maps', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_coverage_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Fig 3: PIT calibration ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.plot(cal_diag['expected'], cal_diag['observed'], 'b.-',
            label=f'BNE (ECE={cal_diag["ece"]:.3f})')
    ax.set_xlabel('Expected'); ax.set_ylabel('Observed')
    ax.set_title('BNE Posterior Calibration (PIT)')
    ax.legend(); ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig_pit_calibration.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Figures saved to {outdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conformalized BNE simulation')
    parser.add_argument('--n_train', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()

    run_experiment(args.n_train, args.seed, args.alpha, args.outdir)

# BNE Application — Project Context for Claude Code

## Project Overview

**BNE (Bayesian Nonparametric Ensemble):** A hierarchical ensemble method for spatial prediction with uncertainty quantification. Published in Bayesian Analysis 2026.

**Current Extension:** Conformalized BNE — adding conformal prediction to BNE, targeting NeurIPS 2026 submission. This is a side project, phased approach.

## Repository Structure

```
BNE_method/
├── case_study_results/
│   ├── wrapper_functions.py    # Core module (2084 lines) — ALL model code lives here
│   ├── BNE_1213.ipynb          # Case study notebook
│   └── BNE_spatialCV.ipynb     # Spatial cross-validation
├── conformal/                  # NEW — Conformalized BNE (Phase 1)
│   ├── __init__.py
│   ├── conformal_bne.py        # Core: split CP, CQR, spatial CQR, BNE credible intervals
│   ├── metrics.py              # Coverage, interval width, ECE, NLL
│   ├── run_simulation.py       # HPC standalone script (CLI: --n_train --seed --alpha)
│   ├── submit_hpc.sh           # Slurm array job (12 jobs: 3 sizes x 4 seeds)
│   ├── collect_results.py      # Aggregate results after HPC jobs finish
│   └── simulation_prototype.ipynb  # Interactive experiment notebook
├── BNE_examine.ipynb
├── BNE_simu.ipynb
├── Tuning/                     # Parameter tuning scripts + Slurm jobs
└── data/
```

## Key Data Flow

```
generate_data_2d()  →  run_base_models()  →  run_bma_model()  →  run_bne_model()
                                                                       ↓
                                                        bne_samples_dict['y']
                                                        shape: (num_mcmc, num_data, 1)
                                                                       ↓
                                              extract_bne_predictions() → conformal methods
```

- `make_bne_samples()['y']`: shape `(num_mcmc, num_data, 1)` — posterior predictive samples
- Conformal methods are **pure post-processing** — no changes to BNE training pipeline

## Conformal Methods (conformal_bne.py)

1. **`bne_credible_interval()`** — baseline: raw posterior quantiles
2. **`split_conformal()`** — standard split CP, absolute residual scores
3. **`conformalized_quantile_regression()`** — CQR (Romano et al. 2019) on BNE quantiles
4. **`spatial_cqr()`** — **KEY NOVELTY**: kernel-weighted CQR with Matern32/RBF/Exponential, spatially-varying conformal quantiles, finite-sample correction (Barber et al. 2023)
5. **`extract_bne_predictions()`** — helper to interface with `make_bne_samples()` output

## HPC (Ginsburg @ Columbia)

- Slurm cluster, existing pattern: 12 CPUs, 80GB RAM
- Conda env name: `BNE` (check `submit_hpc.sh` line `source activate BNE`)
- Run: `cd BNE_method/conformal && sbatch submit_hpc.sh`
- Array job: n_train ∈ {250, 500, 1000}, seeds ∈ {0, 42, 123, 456}
- Results: `conformal/results/sim_n{N}_s{SEED}/` (CSV + PNG figures)
- After jobs: `python collect_results.py` to aggregate

## Phase 1 Status (current)

- [x] conformal_bne.py — 4 methods implemented
- [x] metrics.py — all metrics implemented
- [x] simulation_prototype.ipynb — full notebook
- [x] HPC scripts — run_simulation.py, submit_hpc.sh, collect_results.py
- [x] Sanity check — N(0,1) conformalization verified locally
- [ ] **Run on HPC** ← NEXT STEP
- [ ] Review results, especially: spatial CQR conditional coverage in OOD region

## Phase 2: Full Paper (pending Phase 1 results — 结果好再扩展)

### 2.1 Theory (both coverage + optimality)
- **Theorem 1:** Finite-sample marginal coverage P(Y ∈ C(X)) ≥ 1 - α
- **Theorem 2:** Conditional coverage bound |P(Y ∈ C(X)|X=x) - (1-α)| ≤ O(1/√n_eff(x))
- **Theorem 3:** Minimax optimal interval width rate under spatial kernel weighting
- Connection to BNE's existing Theorem 1 (calibration preserves accuracy)

### 2.2 Data expansion (4 datasets target)
- UCI regression benchmarks (standard, for generality)
- PM2.5 扩展 (more monitors or multi-year) — extends the BA paper case study
- Another spatial dataset (housing prices, temperature, etc.)
- Traffic spatiotemporal (METR-LA)
- 先跑 simulation 再决定具体用哪些

### 2.3 Extended simulations
- Non-stationary variance / sparse vs dense monitoring
- Model misspecification stress test
- Covariate shift scenarios
- Higher-dimensional: 先跑再决定是否保留

### 2.4 Paper positioning
- **Story:** BNE gives good posterior but credible intervals can undercover, especially OOD. Conformal prediction fixes coverage guarantee. Spatial CQR additionally gives adaptive intervals (tight in-domain, wide OOD).
- **Key selling point:** spatial CQR is the methodological novelty. The other methods (split CP, CQR) are baselines.

---

## Implementation Details & Pitfalls

### Architecture
- All BNE model code lives in `wrapper_functions.py` (2084 lines) — do NOT modify for conformal work
- Conformal methods are pure post-processing on `make_bne_samples()['y']`
- `bne_samples_dict['y']` shape: `(num_mcmc, num_data, 1)` — squeeze last dim before use
- 2D simulation uses Mishra's Bird function for mean, Rosenbrock for std

### Spatial CQR Algorithm
- For each test point x_test, compute kernel-weighted quantile of calibration scores
- Uses finite-sample correction from Barber et al. (2023) — adds weight for test point at score=+inf
- When no nearby calibration points exist, returns (-inf, +inf)
- Kernel options: Matern32 (default), RBF, Exponential
- Lengthscale is a hyperparameter — sweep over {0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0}

### Data Split Strategy
- `generate_data_2d()` returns train_full; we further split 70/30 into train/calibration
- Training region: [-π, π]², test region: [-1.25π, 1.25π]² (expanded, so OOD region exists)
- In-domain vs OOD labeling based on whether test point falls within [-π, π]²

### Potential Issues to Watch
- `generate_data_2d()` sets `std_base = std_train = std_test = 1e-3` (near-zero noise) — this is a DEBUG setting in wrapper_functions.py. If coverage looks weird, check this and increase noise.
- BMA + BNE MCMC can be slow — 4hr wall time per job should be enough for n≤1000
- Conda env name assumed `BNE` — verify on Ginsburg with `conda env list`

---

## Verification Checklist (Phase 1)

1. **Sanity check:** ✅ Conformalize N(0,1) → verified 95% coverage locally
2. **Visual:** 2D simulation 上画 BNE intervals vs conformalized intervals 的 spatial map (in notebook/script)
3. **Coverage table:** Format similar to BA paper Table 2, with conformal methods added
4. **关键对比:** Conditional coverage in OOD region — spatial CQR 应该显著赢

## What to look for in Phase 1 results
- BNE credible intervals 在 OOD region 是否 undercoverage？（应该是）
- Split CP 和 CQR 是否达到 marginal coverage target？（应该是，by theory）
- Spatial CQR 在 OOD region 的 conditional coverage 是否比 CQR 好？（关键！）
- Spatial CQR 在 in-domain 的 interval width 是否比 CQR 窄？（希望是）
- Lengthscale sensitivity：哪个 lengthscale 效果最好？

---

## Environment

- TensorFlow 2.7, TensorFlow Probability 0.15, Edward2, GPFlow
- Python 3.8+
- `environment.yml` not in repo — env managed via conda on HPC

## Conventions

- Slurm scripts follow existing pattern in `Tuning/*.sh`
- User language: mixed English/Chinese
- This is a research project — prioritize getting results quickly
- 先跑 simulation prototype，结果好再扩展到 real data 和 theory

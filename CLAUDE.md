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

## Phase 2 (pending Phase 1 results)

- Theory: coverage guarantees + minimax optimality rate
- Data: 4 datasets (UCI + PM2.5 + spatial + traffic METR-LA)
- Extended simulations: non-stationary variance, model misspecification, covariate shift
- Decision point: results 好再扩展

## Environment

- TensorFlow 2.7, TensorFlow Probability 0.15, Edward2, GPFlow
- Python 3.8+
- `environment.yml` not in repo — env managed via conda on HPC

## Conventions

- Slurm scripts follow existing pattern in `Tuning/*.sh`
- User language: mixed English/Chinese
- This is a research project — prioritize getting results quickly

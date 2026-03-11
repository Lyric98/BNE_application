"""Collect and aggregate results from all simulation runs.

Usage (after all HPC jobs finish):
    python collect_results.py
"""

import os
import glob
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def collect():
    csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'sim_*/results.csv')))
    if not csv_files:
        print(f"No results found in {RESULTS_DIR}")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"Collected {len(csv_files)} runs, {len(df)} rows total\n")

    # Aggregate: mean and std across seeds, grouped by (method, n_train)
    metrics = ['coverage_all', 'coverage_in_domain', 'coverage_ood',
               'width_all', 'width_in_domain', 'width_ood']

    agg = df.groupby(['method', 'n_train'])[metrics].agg(['mean', 'std'])

    # Flatten column names
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    agg = agg.reset_index()

    # Pretty print
    print("="*80)
    print("AGGREGATED RESULTS (mean ± std across seeds)")
    print("="*80)
    for n in sorted(df['n_train'].unique()):
        print(f"\n--- N_train = {n} ---")
        sub = agg[agg['n_train'] == n].copy()
        for _, row in sub.iterrows():
            print(f"  {row['method']:20s}  "
                  f"Cov={row['coverage_all_mean']:.3f}±{row['coverage_all_std']:.3f}  "
                  f"InDom={row['coverage_in_domain_mean']:.3f}±{row['coverage_in_domain_std']:.3f}  "
                  f"OOD={row['coverage_ood_mean']:.3f}±{row['coverage_ood_std']:.3f}  "
                  f"Width={row['width_all_mean']:.2f}±{row['width_all_std']:.2f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, 'aggregated.csv')
    agg.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    collect()

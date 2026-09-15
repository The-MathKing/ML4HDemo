#!/usr/bin/env python3
"""
SpotGuard Full Benchmark Execution Script
Runs end-to-end evaluation:
- Data generation & cross-donor partitioning
- Noise-injected pseudo-spot simulation & Discriminator Realism Gate check
- Deconvolution & Bayesian highest-posterior-density interval computation
- Total Variation conformal coverage calibration & covariate-shift reweighting
- Real-time diagnostics & result logging
"""

import json
import numpy as np
from spotguard.data import generate_synthetic_benchmark_pair
from spotguard.evaluate import run_full_benchmark_pipeline

def main():
    print("=" * 70)
    print("SPOTGUARD: END-TO-END BENCHMARK & CALIBRATION PIPELINE")
    print("=" * 70)

    print("\n[1/4] Generating reference single-cell atlas and paired ST sections...")
    ref_atlas, st_donor_a, st_donor_b = generate_synthetic_benchmark_pair(
        n_genes=300,
        n_cells_per_donor=1000,
        n_spots_per_tissue=500,
        seed=42
    )
    print(f"  ✓ Reference Atlas: {ref_atlas.n_cells} cells, {ref_atlas.n_genes} genes, {ref_atlas.n_types} cell types")
    print(f"  ✓ In-Distribution (Donor A): {st_donor_a.n_spots} spots")
    print(f"  ✓ Shifted (Donor B): {st_donor_b.n_spots} spots")

    print("\n[2/4] Running simulation, Discriminator Realism Gate & Deconvolution...")
    results = run_full_benchmark_pipeline(
        ref_atlas=ref_atlas,
        st_donor_a=st_donor_a,
        st_donor_b=st_donor_b,
        alpha=0.10,
        seed=42
    )

    gate = results["discriminator_gate"]
    print(f"  ✓ Discriminator Realism AUC: {gate['discriminator_auc']:.3f} -> {gate['status']}")
    weight_diag = results["weight_diagnostics"]
    print(f"  ✓ Importance Weight ESS: {weight_diag['ess']:.1f} ({weight_diag['ess_fraction']*100:.1f}% of calibration set)")

    print("\n[3/4] Quantitative Benchmark Results (Target nominal 90% coverage):")
    id_res = results["in_distribution"]
    shift_res = results["shifted"]

    print("\n" + "-" * 70)
    print(f"{'Regime':<20} | {'Method':<25} | {'Empirical Coverage':<18} | {'95% Bootstrap CI'}")
    print("-" * 70)
    print(f"{'In-Distribution':<20} | {'Bayesian Joint CI':<25} | {id_res['bayesian_joint_coverage']*100:>6.1f}%            | N/A")
    print(f"{'In-Distribution':<20} | {'SpotGuard Conformal':<25} | {id_res['spotguard_coverage']*100:>6.1f}%            | [{id_res['spotguard_ci'][0]*100:.1f}%, {id_res['spotguard_ci'][1]*100:.1f}%]")
    print("-" * 70)
    print(f"{'Shifted (Cross-Donor)':<20} | {'Bayesian Joint CI':<25} | {shift_res['bayesian_joint_coverage']*100:>6.1f}%            | N/A")
    print(f"{'Shifted (Cross-Donor)':<20} | {'Unweighted Conformal':<25} | {shift_res['spotguard_unweighted_coverage']*100:>6.1f}%            | N/A")
    print(f"{'Shifted (Cross-Donor)':<20} | {'SpotGuard Weighted':<25} | {shift_res['spotguard_weighted_coverage']*100:>6.1f}%            | [{shift_res['spotguard_ci'][0]*100:.1f}%, {shift_res['spotguard_ci'][1]*100:.1f}%]")
    print("-" * 70)

    print(f"\n[4/4] Selective Deconvolution AURC: {shift_res['aurc']:.4f}")
    print(f"  ✓ SpotGuard TV Quantile Bound (q_alpha): {shift_res['q_alpha_weighted']:.3f}")
    
    # Save benchmark json artifact
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n✓ Full benchmark results exported to benchmark_results.json")
    print("=" * 70)

if __name__ == '__main__':
    main()

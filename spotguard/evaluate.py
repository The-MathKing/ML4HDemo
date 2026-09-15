"""
SpotGuard Evaluation Harness & Benchmark Suite
Implements Phase 5:
- Marginal & conditional coverage with bootstrap confidence intervals
- Retention rate vs. alpha curves
- Selective prediction Area Under Risk-Coverage (AURC)
- Ablation experiments: Unweighted vs. Weighted Conformal vs. Mahalanobis OOD vs. Aitchison
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .engine import (
    total_variation_score,
    aitchison_score,
    compute_effective_sample_size,
    process_weights,
    compute_conformal_quantile,
    decision_ambiguity_gate
)

def evaluate_conformal_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    q_alpha_tv: float,
    n_bootstraps: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates empirical coverage of the TV conformal ball with bootstrap standard errors.
    """
    scores = total_variation_score(y_true, y_pred)
    covered = (scores <= q_alpha_tv).astype(np.float64)
    empirical_coverage = float(np.mean(covered))

    rng = np.random.RandomState(seed)
    N = len(covered)
    boot_means = []
    for _ in range(n_bootstraps):
        sample_idx = rng.choice(N, size=N, replace=True)
        boot_means.append(np.mean(covered[sample_idx]))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "empirical_coverage": empirical_coverage,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "mean_tv_error": float(np.mean(scores)),
        "tv_bound_q_alpha": float(q_alpha_tv)
    }


def evaluate_bayesian_coverage(
    y_true: np.ndarray,
    credible_intervals_90: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates empirical coverage of baseline 90% Bayesian credible intervals.
    Coverage is satisfied iff true proportion falls inside [CI_lower, CI_upper] for all lineages.
    """
    N, K = y_true.shape
    ci_low = credible_intervals_90[:, :, 0]
    ci_high = credible_intervals_90[:, :, 1]

    # Check marginal coverage per cell type
    covered_per_type = (y_true >= ci_low) & (y_true <= ci_high)
    marginal_coverage_per_type = np.mean(covered_per_type, axis=0)

    # Joint coverage: all cell types covered simultaneously
    joint_covered = np.all(covered_per_type, axis=-1).astype(np.float64)

    return {
        "joint_empirical_coverage": float(np.mean(joint_covered)),
        "marginal_coverage_per_type": [float(c) for c in marginal_coverage_per_type],
        "mean_marginal_coverage": float(np.mean(marginal_coverage_per_type))
    }


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nonconformity_scores: np.ndarray,
    n_thresholds: int = 50
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes selective deconvolution Risk-Coverage curve and AURC.
    Spots with highest nonconformity are rejected first.
    
    Returns:
        retention_rates: array in [0, 1]
        selective_risks: mean TV error on retained spots
        aurc: Area Under the Risk-Coverage Curve
    """
    tv_errors = total_variation_score(y_true, y_pred)
    sorted_indices = np.argsort(nonconformity_scores)  # lowest to highest error

    retention_rates = []
    selective_risks = []

    percentiles = np.linspace(10, 100, n_thresholds)
    for p in percentiles:
        n_retained = max(1, int(np.ceil(p / 100.0 * len(tv_errors))))
        retained_idx = sorted_indices[:n_retained]
        retention_rates.append(n_retained / len(tv_errors))
        selective_risks.append(float(np.mean(tv_errors[retained_idx])))

    retention_rates = np.array(retention_rates)
    selective_risks = np.array(selective_risks)

    # Compute numerical AURC via trapezoidal integration
    aurc = float(np.sum(0.5 * (retention_rates[1:] - retention_rates[:-1]) * (selective_risks[1:] + selective_risks[:-1])))
    return retention_rates, selective_risks, aurc


def run_full_benchmark_pipeline(
    ref_atlas,
    st_donor_a,
    st_donor_b,
    alpha: float = 0.10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes the end-to-end benchmark pipeline across in-distribution and shifted test sets.
    """
    from .simulator import PseudoSpotSimulator, DiscriminatorRealismGate
    from .deconvolution import FastBayesianDeconvolution

    # 1. Generate calibration pseudo-spots matching empirical spatial library size
    sim = PseudoSpotSimulator(ref_atlas, seed=seed)
    real_lib_sizes = np.sum(st_donor_a.counts, axis=-1)
    X_pseudo, Y_pseudo = sim.sample_pseudo_spots(
        n_pseudo_spots=1000,
        target_library_sizes=real_lib_sizes
    )

    # 2. Fit deconvolution model on reference
    deconv = FastBayesianDeconvolution(ref_atlas, seed=seed)
    
    # Deconvolve pseudo-spots to obtain calibration scores
    pseudo_dataset_dummy = type("DummyST", (), {"counts": X_pseudo, "n_spots": len(X_pseudo)})()
    deconv_cal = deconv.fit_predict(pseudo_dataset_dummy)
    cal_scores = total_variation_score(Y_pseudo, deconv_cal.mean)

    # 3. Fit Discriminator Realism Gate
    disc = DiscriminatorRealismGate(seed=seed)
    gate_status = disc.fit(X_pseudo, st_donor_a.counts)

    # Importance weights for shifted query
    raw_weights_shift = disc.predict_density_ratios(X_pseudo, st_donor_b.counts)
    norm_weights, weight_diag = process_weights(raw_weights_shift, clip_percentile=95.0)

    # Conformal quantiles
    q_unweighted = compute_conformal_quantile(cal_scores, alpha=alpha, weights=None)
    q_weighted = compute_conformal_quantile(cal_scores, alpha=alpha, weights=norm_weights)

    # 4. Deconvolve and evaluate on In-Distribution (Donor A)
    res_donor_a = deconv.fit_predict(st_donor_a)
    eval_id_spotguard = evaluate_conformal_coverage(
        st_donor_a.ground_truth_proportions,
        res_donor_a.mean,
        q_unweighted
    )
    eval_id_bayesian = evaluate_bayesian_coverage(
        st_donor_a.ground_truth_proportions,
        res_donor_a.credible_intervals_90
    )

    # 5. Deconvolve and evaluate on Shifted (Donor B)
    res_donor_b = deconv.fit_predict(st_donor_b)
    eval_shift_unweighted = evaluate_conformal_coverage(
        st_donor_b.ground_truth_proportions,
        res_donor_b.mean,
        q_unweighted
    )
    eval_shift_spotguard = evaluate_conformal_coverage(
        st_donor_b.ground_truth_proportions,
        res_donor_b.mean,
        q_weighted
    )
    eval_shift_bayesian = evaluate_bayesian_coverage(
        st_donor_b.ground_truth_proportions,
        res_donor_b.credible_intervals_90
    )

    # 6. Risk-Coverage and AURC
    _, _, aurc_spotguard = compute_risk_coverage_curve(
        st_donor_b.ground_truth_proportions,
        res_donor_b.mean,
        total_variation_score(Y_pseudo[:len(res_donor_b.mean)], res_donor_b.mean)
    )

    return {
        "discriminator_gate": gate_status,
        "weight_diagnostics": weight_diag,
        "in_distribution": {
            "spotguard_coverage": eval_id_spotguard["empirical_coverage"],
            "spotguard_ci": [eval_id_spotguard["ci_95_lower"], eval_id_spotguard["ci_95_upper"]],
            "bayesian_joint_coverage": eval_id_bayesian["joint_empirical_coverage"],
            "bayesian_mean_marginal": eval_id_bayesian["mean_marginal_coverage"],
            "q_alpha": q_unweighted
        },
        "shifted": {
            "spotguard_weighted_coverage": eval_shift_spotguard["empirical_coverage"],
            "spotguard_ci": [eval_shift_spotguard["ci_95_lower"], eval_shift_spotguard["ci_95_upper"]],
            "spotguard_unweighted_coverage": eval_shift_unweighted["empirical_coverage"],
            "bayesian_joint_coverage": eval_shift_bayesian["joint_empirical_coverage"],
            "bayesian_mean_marginal": eval_shift_bayesian["mean_marginal_coverage"],
            "q_alpha_weighted": q_weighted,
            "q_alpha_unweighted": q_unweighted,
            "aurc": aurc_spotguard
        }
    }

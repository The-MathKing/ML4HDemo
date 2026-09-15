"""
Unit tests for SpotGuard Conformal Engine
Validates mathematical invariants, coverage calibration, and decision ambiguity gating.
"""

import unittest
import numpy as np
from spotguard.engine import (
    total_variation_score,
    aitchison_score,
    compute_effective_sample_size,
    process_weights,
    compute_conformal_quantile,
    decision_ambiguity_gate
)

class TestSpotGuardEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_total_variation_score_properties(self):
        # Test exact match
        p1 = np.array([0.5, 0.3, 0.2])
        self.assertAlmostEqual(total_variation_score(p1, p1), 0.0)

        # Test orthogonal simplex vertices
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(total_variation_score(v1, v2), 1.0)

        # Test bounded strictly in [0, 1] on random simplex compositions
        for _ in range(100):
            p = np.random.dirichlet(np.ones(5))
            q = np.random.dirichlet(np.ones(5))
            tv = total_variation_score(p, q)
            self.assertGreaterEqual(tv, 0.0)
            self.assertLessEqual(tv, 1.0)

    def test_aitchison_score_handles_zeros(self):
        # Aitchison with sparse zero proportions should not crash due to eps-regularization
        p_sparse = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        q_sparse = np.array([[0.8, 0.1, 0.1], [0.4, 0.4, 0.2]])
        scores = aitchison_score(p_sparse, q_sparse, eps=1e-4)
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_split_conformal_coverage_monte_carlo(self):
        """
        Under exchangeability, marginal conformal prediction over random calibration splits
        must land at 1 - alpha +/- MC error.
        """
        N_cal = 500
        N_test = 2000
        alpha = 0.10  # 90% target
        K = 4
        n_trials = 30
        coverages = []

        for _ in range(n_trials):
            # Generate exchangeable true and estimated proportions
            y_cal_true = np.random.dirichlet(np.ones(K), size=N_cal)
            noise_cal = np.random.dirichlet(np.ones(K) * 5, size=N_cal)
            y_cal_pred = 0.8 * y_cal_true + 0.2 * noise_cal
            y_cal_pred = y_cal_pred / np.sum(y_cal_pred, axis=-1, keepdims=True)

            cal_scores = total_variation_score(y_cal_true, y_cal_pred)
            q_alpha = compute_conformal_quantile(cal_scores, alpha)

            # Test set
            y_test_true = np.random.dirichlet(np.ones(K), size=N_test)
            noise_test = np.random.dirichlet(np.ones(K) * 5, size=N_test)
            y_test_pred = 0.8 * y_test_true + 0.2 * noise_test
            y_test_pred = y_test_pred / np.sum(y_test_pred, axis=-1, keepdims=True)

            test_scores = total_variation_score(y_test_true, y_test_pred)
            coverages.append(np.mean(test_scores <= q_alpha))

        mean_coverage = float(np.mean(coverages))
        # Marginal coverage across splits must land within [0.89, 0.915]
        self.assertGreaterEqual(mean_coverage, 0.890)
        self.assertLessEqual(mean_coverage, 0.915)

    def test_weighted_conformal_shift_adaptation(self):
        """
        Test weighted conformal under known synthetic density ratio weights.
        """
        M = 1000
        scores = np.random.exponential(scale=0.1, size=M)
        weights = np.random.lognormal(mean=0.0, sigma=0.5, size=M)

        q_unweighted = compute_conformal_quantile(scores, alpha=0.10)
        q_weighted = compute_conformal_quantile(scores, alpha=0.10, weights=weights)

        self.assertTrue(np.isfinite(q_unweighted))
        self.assertTrue(np.isfinite(q_weighted))

    def test_effective_sample_size_and_clipping(self):
        # Equal weights -> ESS = M
        weights_equal = np.ones(100)
        norm_w, diag = process_weights(weights_equal, clip_percentile=None)
        self.assertAlmostEqual(diag["ess"], 100.0, places=2)
        self.assertAlmostEqual(diag["ess_fraction"], 1.0, places=2)

        # One extreme spike -> ESS collapses
        weights_spiked = np.ones(100)
        weights_spiked[0] = 1000.0
        norm_w, diag_clipped = process_weights(weights_spiked, clip_percentile=95.0)
        # After clipping at 95th percentile, ESS should be stabilized
        self.assertGreater(diag_clipped["ess"], 10.0)

    def test_decision_ambiguity_gating(self):
        threshold = 0.05  # Clinical boundary for CD8+ T-cell infiltration
        q_alpha = 0.02    # Conformal uncertainty bound

        # Case 1: Unambiguous Positive (Estimate 0.12 +/- 0.02 -> [0.10, 0.14] > 0.05)
        res_pos = decision_ambiguity_gate(0.12, q_alpha, threshold)
        self.assertFalse(res_pos["is_ambiguous"])
        self.assertTrue(res_pos["is_trusted"])
        self.assertEqual(res_pos["decision"], "UNANIMOUS_POSITIVE")

        # Case 2: Unambiguous Negative (Estimate 0.01 +/- 0.02 -> [0.00, 0.03] < 0.05)
        res_neg = decision_ambiguity_gate(0.01, q_alpha, threshold)
        self.assertFalse(res_neg["is_ambiguous"])
        self.assertTrue(res_neg["is_trusted"])
        self.assertEqual(res_neg["decision"], "UNANIMOUS_NEGATIVE")

        # Case 3: Ambiguous Boundary (Estimate 0.06 +/- 0.02 -> [0.04, 0.08] straddles 0.05)
        res_amb = decision_ambiguity_gate(0.06, q_alpha, threshold)
        self.assertTrue(res_amb["is_ambiguous"])
        self.assertFalse(res_amb["is_trusted"])
        self.assertEqual(res_amb["decision"], "AMBIGUOUS")

if __name__ == '__main__':
    unittest.main()

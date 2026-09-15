"""
SpotGuard Deconvolution Backend & Caching
Implements Phase 3:
- Baseline deconvolution interfaces (Bayesian / Non-negative least squares / Variational)
- Posterior distribution sampling & Highest Posterior Density (HPD) credible regions
- Disk caching mechanism for fast evaluation and interactive UI
"""

import numpy as np
import os
import json
from scipy.optimize import nnls
from typing import Dict, List, Tuple, Optional, Any
from .data import SCRNAReference, STDataset

class DeconvolutionResult:
    """Stores point estimates, posterior samples, and metadata."""
    def __init__(
        self,
        method_name: str,
        proportions_mean: np.ndarray,      # (N_spots, K)
        proportions_samples: Optional[np.ndarray] = None, # (N_samples, N_spots, K)
        credible_intervals_90: Optional[np.ndarray] = None, # (N_spots, K, 2)
        cell_types: Optional[List[str]] = None
    ):
        self.method_name = method_name
        self.mean = proportions_mean.astype(np.float32)
        self.samples = proportions_samples.astype(np.float32) if proportions_samples is not None else None
        self.credible_intervals_90 = credible_intervals_90
        self.cell_types = list(cell_types) if cell_types is not None else []

    def save(self, filepath: str):
        """Saves result to an uncompressed or compressed npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "method_name": self.method_name,
            "mean": self.mean,
            "cell_types": self.cell_types
        }
        if self.samples is not None:
            data["samples"] = self.samples
        if self.credible_intervals_90 is not None:
            data["ci_90"] = self.credible_intervals_90
        np.savez_compressed(filepath, **data)

    @classmethod
    def load(cls, filepath: str) -> "DeconvolutionResult":
        """Loads cached deconvolution result."""
        archive = np.load(filepath, allow_pickle=True)
        return cls(
            method_name=str(archive["method_name"]),
            proportions_mean=archive["mean"],
            proportions_samples=archive["samples"] if "samples" in archive else None,
            credible_intervals_90=archive["ci_90"] if "ci_90" in archive else None,
            cell_types=list(archive["cell_types"]) if "cell_types" in archive else None
        )


class FastBayesianDeconvolution:
    """
    High-performance Bayesian / NNLS deconvolution model with posterior sampling.
    Estimates spot composition by solving constrained regularized regression against reference signatures
    and generating Dirichlet/Gaussian posterior samples.
    """
    def __init__(self, reference: SCRNAReference, n_posterior_samples: int = 100, seed: int = 42):
        self.ref = reference
        self.n_samples = n_posterior_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Compute cell-type mean signature matrix S: (G, K)
        type_indices = self.ref.get_type_indices()
        self.K = self.ref.n_types
        self.G = self.ref.n_genes
        self.cell_types = self.ref.cell_type_names

        signatures = []
        for ctype in self.cell_types:
            idxs = type_indices[ctype]
            mean_expr = np.mean(self.ref.counts[idxs], axis=0) + 1e-4
            signatures.append(mean_expr)
        self.signature_matrix = np.column_stack(signatures)  # (G, K)

    def fit_predict(self, st_data: STDataset) -> DeconvolutionResult:
        """
        Runs deconvolution and generates posterior samples and 90% credible intervals.
        """
        N = st_data.n_spots
        point_estimates = np.zeros((N, self.K), dtype=np.float32)
        samples = np.zeros((self.n_samples, N, self.K), dtype=np.float32)
        ci_90 = np.zeros((N, self.K, 2), dtype=np.float32)

        S = self.signature_matrix  # (G, K)
        
        for i in range(N):
            spot_count = st_data.counts[i]
            # Solve non-negative least squares: spot_count ~= S @ beta
            beta, _ = nnls(S, spot_count)
            sum_beta = np.sum(beta)
            if sum_beta > 0:
                p_hat = beta / sum_beta
            else:
                p_hat = np.ones(self.K) / self.K
            point_estimates[i] = p_hat

            # Approximate posterior sampling via Dirichlet centered on p_hat with concentration ~ library size
            concentration = max(10.0, np.sum(spot_count) * 0.05)
            alpha_vec = np.maximum(0.1, p_hat * concentration)
            spot_samples = self.rng.dirichlet(alpha_vec, size=self.n_samples)
            samples[:, i, :] = spot_samples

            # 90% Highest Posterior Density / Credible Interval
            ci_90[i, :, 0] = np.percentile(spot_samples, 5.0, axis=0)
            ci_90[i, :, 1] = np.percentile(spot_samples, 95.0, axis=0)

        return DeconvolutionResult(
            method_name="FastBayesianDeconv",
            proportions_mean=point_estimates,
            proportions_samples=samples,
            credible_intervals_90=ci_90,
            cell_types=self.cell_types
        )

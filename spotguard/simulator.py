"""
SpotGuard Pseudo-Spot Simulator & Discriminator Realism Gate
Implements Phase 2:
- Dirichlet mixture proportion sampling
- Poisson cell count aggregation
- Negative binomial overdispersion
- Empty droplet ambient RNA contamination injection
- Discriminator Realism Gate (evaluating 0.60 <= AUC <= 0.80)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, Optional
from .data import SCRNAReference

class PseudoSpotSimulator:
    """
    Synthesizes calibration pseudo-spots from single-cell reference atlases.
    Includes ambient RNA contamination, library-size variance, and negative binomial count sampling.
    """
    def __init__(
        self,
        reference: SCRNAReference,
        mean_cells_per_spot: float = 10.0,
        nb_dispersion: float = 2.0,
        ambient_rna_alpha: float = 2.0,
        ambient_rna_beta: float = 10.0,
        seed: int = 42
    ):
        self.ref = reference
        self.mean_cells_per_spot = mean_cells_per_spot
        self.nb_dispersion = nb_dispersion
        self.ambient_a = ambient_rna_alpha
        self.ambient_b = ambient_rna_beta
        self.rng = np.random.RandomState(seed)

        # Precompute per-cell-type cell indices and average expression
        self.type_indices = self.ref.get_type_indices()
        self.K = self.ref.n_types
        self.ambient_profile = np.mean(self.ref.counts, axis=0)
        self.ambient_profile = self.ambient_profile / (np.sum(self.ambient_profile) + 1e-8)

    def sample_pseudo_spots(
        self,
        n_pseudo_spots: int = 1500,
        dirichlet_prior: Optional[np.ndarray] = None,
        target_library_sizes: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates N pseudo-spots with library size matching and ambient RNA injection.
        """
        if dirichlet_prior is None:
            dirichlet_prior = np.ones(self.K) * 0.8  # Mildly sparse Dirichlet

        Y_pseudo = self.rng.dirichlet(dirichlet_prior, size=n_pseudo_spots)
        X_pseudo = np.zeros((n_pseudo_spots, self.ref.n_genes), dtype=np.float32)

        for i in range(n_pseudo_spots):
            # Sample number of cells in this spot
            N_cells = max(3, self.rng.poisson(self.mean_cells_per_spot))
            proportions = Y_pseudo[i]

            # Sample individual cells according to proportions
            spot_counts = np.zeros(self.ref.n_genes, dtype=np.float32)
            for k_idx, ctype in enumerate(self.ref.cell_type_names):
                cell_idxs = self.type_indices[ctype]
                n_k = int(np.round(proportions[k_idx] * N_cells))
                if n_k > 0 and len(cell_idxs) > 0:
                    sampled_cell_idxs = self.rng.choice(cell_idxs, size=n_k, replace=True)
                    spot_counts += np.sum(self.ref.counts[sampled_cell_idxs], axis=0)

            # Sample ambient RNA contamination fraction rho ~ Beta(a, b)
            rho = self.rng.beta(self.ambient_a, self.ambient_b)
            total_umi = max(100.0, np.sum(spot_counts))
            ambient_counts = rho * total_umi * self.ambient_profile

            # Add technical sequencing noise
            combined_mu = (1.0 - rho) * spot_counts + ambient_counts
            
            # Library size scaling if provided
            if target_library_sizes is not None and len(target_library_sizes) > 0:
                target_lib = self.rng.choice(target_library_sizes)
                current_lib = max(1.0, np.sum(combined_mu))
                combined_mu = combined_mu * (target_lib / current_lib)

            p_nb = 1.0 / (1.0 + combined_mu / self.nb_dispersion)
            p_nb = np.clip(p_nb, 1e-4, 1.0 - 1e-4)

            X_pseudo[i] = self.rng.negative_binomial(n=self.nb_dispersion, p=p_nb)

        return X_pseudo, Y_pseudo


class DiscriminatorRealismGate:
    """
    Evaluates whether synthetic pseudo-spots match real spatial spots with appropriate realism.
    Computes importance density-ratio weights w(x) and verifies discriminator AUC in [0.60, 0.80].
    """
    def __init__(self, n_components_pca: int = 30, reg_c: float = 0.1, seed: int = 42):
        self.n_components = n_components_pca
        self.reg_c = reg_c
        self.seed = seed
        self.pca = PCA(n_components=n_components_pca, random_state=seed)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=reg_c, max_iter=500, random_state=seed)
        self.fitted = False
        self.auc_score = 0.0

    def fit(self, X_pseudo: np.ndarray, X_real: np.ndarray) -> Dict[str, Any]:
        """
        Fits domain discriminator to distinguish pseudo-spots (Class 0) from real spots (Class 1).
        
        Returns:
            Dict containing discriminator AUC, realism status, and summary statistics.
        """
        N_pseudo = X_pseudo.shape[0]
        N_real = X_real.shape[0]

        # Log1p normalization
        X_pseudo_norm = np.log1p(X_pseudo / (np.sum(X_pseudo, axis=-1, keepdims=True) + 1e-8) * 1e4)
        X_real_norm = np.log1p(X_real / (np.sum(X_real, axis=-1, keepdims=True) + 1e-8) * 1e4)

        X_all = np.vstack([X_pseudo_norm, X_real_norm])
        y_all = np.array([0] * N_pseudo + [1] * N_real)

        # Dimensionality reduction
        X_pca = self.pca.fit_transform(X_all)
        X_scaled = self.scaler.fit_transform(X_pca)

        # Cross-validation / train split for AUC check
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(len(y_all))
        train_idx, val_idx = perm[:int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

        self.model.fit(X_scaled[train_idx], y_all[train_idx])
        val_probs = self.model.predict_proba(X_scaled[val_idx])[:, 1]
        self.auc_score = float(roc_auc_score(y_all[val_idx], val_probs))
        self.fitted = True

        # Check realism window [0.60, 0.80]
        is_realistic = (0.55 <= self.auc_score <= 0.85)
        is_optimal = (0.60 <= self.auc_score <= 0.80)

        return {
            "discriminator_auc": self.auc_score,
            "is_realistic": is_realistic,
            "is_optimal": is_optimal,
            "n_pseudo": N_pseudo,
            "n_real": N_real,
            "status": "PASS (Optimal Realism)" if is_optimal else ("ACCEPTABLE" if is_realistic else "FAIL (Simulator Discrepancy)")
        }

    def predict_density_ratios(self, X_pseudo: np.ndarray, X_query: np.ndarray) -> np.ndarray:
        """
        Computes importance weights w(x) = P(query) / P(pseudo) for calibration instances.
        w(x) = (d(x) / (1 - d(x))) * (N_pseudo / N_query)
        """
        if not self.fitted:
            raise RuntimeError("Discriminator must be fitted before computing density ratios.")

        X_norm = np.log1p(X_pseudo / (np.sum(X_pseudo, axis=-1, keepdims=True) + 1e-8) * 1e4)
        X_pca = self.pca.transform(X_norm)
        X_scaled = self.scaler.transform(X_pca)

        probs = self.model.predict_proba(X_scaled)[:, 1]
        probs = np.clip(probs, 1e-3, 1.0 - 1e-3)

        N_pseudo = X_pseudo.shape[0]
        N_query = X_query.shape[0]
        odds = probs / (1.0 - probs)
        weights = odds * (N_pseudo / N_query)

        return weights

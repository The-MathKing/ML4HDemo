"""
SpotGuard Data Pipeline
Handles single-cell RNA-seq references, spatial transcriptomics query datasets,
gene filtering, HVG selection, and donor-level cross-validation splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class STDataset:
    """Container for spatial transcriptomics count matrix and metadata."""
    def __init__(
        self,
        counts: np.ndarray,
        gene_names: List[str],
        coordinates: np.ndarray,
        spot_ids: List[str],
        donor_id: str,
        ground_truth_proportions: Optional[np.ndarray] = None,
        cell_types: Optional[List[str]] = None,
        region_labels: Optional[List[str]] = None
    ):
        self.counts = counts.astype(np.float32)  # (N_spots, G_genes)
        self.gene_names = list(gene_names)
        self.coordinates = coordinates.astype(np.float32)  # (N_spots, 2)
        self.spot_ids = list(spot_ids)
        self.donor_id = donor_id
        self.ground_truth_proportions = ground_truth_proportions  # Optional (N_spots, K_types)
        self.cell_types = list(cell_types) if cell_types is not None else []
        self.region_labels = list(region_labels) if region_labels is not None else []

    @property
    def n_spots(self) -> int:
        return self.counts.shape[0]

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]


class SCRNAReference:
    """Container for single-cell reference atlas."""
    def __init__(
        self,
        counts: np.ndarray,
        gene_names: List[str],
        cell_types: List[str],
        donor_ids: List[str],
        cell_type_names: Optional[List[str]] = None
    ):
        self.counts = counts.astype(np.float32)  # (N_cells, G_genes)
        self.gene_names = list(gene_names)
        self.cell_types = list(cell_types)
        self.donor_ids = list(donor_ids)
        if cell_type_names is None:
            self.cell_type_names = sorted(list(set(cell_types)))
        else:
            self.cell_type_names = list(cell_type_names)

    @property
    def n_cells(self) -> int:
        return self.counts.shape[0]

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]

    @property
    def n_types(self) -> int:
        return len(self.cell_type_names)

    def get_type_indices(self) -> Dict[str, np.ndarray]:
        """Returns dict mapping cell type name to array of cell indices."""
        arr_types = np.array(self.cell_types)
        return {ctype: np.where(arr_types == ctype)[0] for ctype in self.cell_type_names}


def generate_synthetic_benchmark_pair(
    n_genes: int = 500,
    n_cells_per_donor: int = 1500,
    n_spots_per_tissue: int = 600,
    cell_types: Optional[List[str]] = None,
    seed: int = 42
) -> Tuple[SCRNAReference, STDataset, STDataset]:
    """
    Generates a realistic paired synthetic scRNA-seq reference and two spatial tissue sections (Donor A, Donor B)
    with biological expression signatures, cell-type specific marker genes, library size variation,
    and cross-donor batch variance. Useful for rapid testing and CI/CD validation.
    """
    rng = np.random.RandomState(seed)
    if cell_types is None:
        cell_types = ["Tumor_Epithelial", "Stroma_Fibroblast", "CD8_T_Cell", "B_Cell", "Myeloid"]

    K = len(cell_types)
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]

    # Generate cell-type baseline expression profiles with marker peaks
    base_profiles = rng.gamma(shape=0.5, scale=2.0, size=(K, n_genes))
    # Assign specific marker genes with high expression
    for k in range(K):
        marker_start = k * (n_genes // K)
        marker_end = (k + 1) * (n_genes // K)
        base_profiles[k, marker_start:marker_end] *= rng.uniform(4.0, 8.0, size=(marker_end - marker_start))

    # --- Generate scRNA-seq Reference (Donors 1, 2) ---
    all_sc_counts = []
    all_sc_types = []
    all_sc_donors = []

    for donor in ["Donor_A", "Donor_B"]:
        donor_shift = rng.normal(loc=1.0, scale=0.15, size=(1, n_genes))
        donor_shift = np.clip(donor_shift, 0.4, 2.5)
        for k_idx, ctype in enumerate(cell_types):
            n_cells_k = n_cells_per_donor // K
            mu = base_profiles[k_idx:k_idx+1] * donor_shift
            # Negative binomial sampling: mean mu, overdispersion theta = 2.0
            p = 1.0 / (1.0 + mu / 2.0)
            p = np.clip(p, 1e-4, 1.0 - 1e-4)
            counts_k = rng.negative_binomial(n=2.0, p=p, size=(n_cells_k, n_genes))
            all_sc_counts.append(counts_k)
            all_sc_types.extend([ctype] * n_cells_k)
            all_sc_donors.extend([donor] * n_cells_k)

    sc_counts = np.vstack(all_sc_counts)
    ref_atlas = SCRNAReference(
        counts=sc_counts,
        gene_names=genes,
        cell_types=all_sc_types,
        donor_ids=all_sc_donors,
        cell_type_names=cell_types
    )

    # --- Helper to generate a spatial tissue section ---
    def _create_st_section(donor_name: str, tech_shift_factor: float = 1.0) -> STDataset:
        grid_dim = int(np.ceil(np.sqrt(n_spots_per_tissue)))
        coords = []
        true_props = []
        st_counts = []
        regions = []

        donor_shift = rng.normal(loc=1.0, scale=0.2, size=(1, n_genes)) * tech_shift_factor
        donor_shift = np.clip(donor_shift, 0.3, 3.0)

        for i in range(n_spots_per_tissue):
            gx = i % grid_dim
            gy = i // grid_dim
            x_coord = gx * 100.0 + rng.normal(0, 5)
            y_coord = gy * 100.0 + rng.normal(0, 5)
            coords.append([x_coord, y_coord])

            # Histological layout: radial tumor center
            r = np.sqrt((gx - grid_dim/2)**2 + (gy - grid_dim/2)**2) / (grid_dim / 2)
            if r < 0.4:
                reg = "Tumor_Core"
                alpha_dir = [8.0, 1.0, 0.5, 0.2, 0.8]
            elif r < 0.7:
                reg = "Tumor_Margin"
                alpha_dir = [3.0, 3.0, 3.5, 1.0, 2.0]
            else:
                reg = "Stroma"
                alpha_dir = [0.2, 8.0, 0.8, 1.0, 1.0]

            prop = rng.dirichlet(alpha_dir)
            true_props.append(prop)
            regions.append(reg)

            # Spot expression: mixture of cell profiles + Poisson count of cells N ~ Poisson(10)
            n_cells_spot = max(3, rng.poisson(10))
            spot_mu = np.zeros(n_genes)
            for k_idx in range(K):
                spot_mu += prop[k_idx] * base_profiles[k_idx] * n_cells_spot * donor_shift[0]

            # Ambient RNA noise injection
            ambient = np.mean(base_profiles, axis=0) * rng.uniform(0.05, 0.2) * n_cells_spot
            spot_mu += ambient

            # Negative binomial counts
            p_st = 1.0 / (1.0 + spot_mu / 2.0)
            p_st = np.clip(p_st, 1e-4, 1.0 - 1e-4)
            c = rng.negative_binomial(n=2.0, p=p_st)
            st_counts.append(c)

        return STDataset(
            counts=np.array(st_counts),
            gene_names=genes,
            coordinates=np.array(coords),
            spot_ids=[f"{donor_name}_SPOT_{i:04d}" for i in range(n_spots_per_tissue)],
            donor_id=donor_name,
            ground_truth_proportions=np.array(true_props),
            cell_types=cell_types,
            region_labels=regions
        )

    # In-distribution tissue (Donor A)
    st_donor_a = _create_st_section("Donor_A", tech_shift_factor=1.0)
    # Shifted tissue (Donor B with platform/capture shift)
    st_donor_b = _create_st_section("Donor_B", tech_shift_factor=0.75)

    return ref_atlas, st_donor_a, st_donor_b

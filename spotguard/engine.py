"""
SpotGuard Conformal Engine: Core Mathematical Implementation
Modules:
- scores: Total Variation (TV) and Aitchison (with eps-replacement)
- weights: Importance weights, clipping, ESS calculation
- conformal: Split & weighted conformal quantile estimation
- gate: Decision-ambiguity gating on the probability simplex
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

def total_variation_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Computes Total Variation (TV) nonconformity score on the probability simplex.
    S_TV = 0.5 * ||y_true - y_pred||_1
    Defined everywhere on the closed simplex [0, 1]^K, strictly bounded in [0, 1].
    
    Args:
        y_true: (N, K) or (K,) true proportion vectors.
        y_pred: (N, K) or (K,) predicted proportion vectors.
    Returns:
        (N,) array of nonconformity scores in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        return 0.5 * np.sum(np.abs(y_true - y_pred))
    return 0.5 * np.sum(np.abs(y_true - y_pred), axis=-1)


def aitchison_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Computes Aitchison nonconformity score in centered log-ratio (clr) coordinates
    with epsilon-replacement for zero components.
    
    Args:
        y_true: (N, K) array.
        y_pred: (N, K) array.
        eps: Regularization parameter for zero proportions.
    Returns:
        (N,) array of Aitchison Euclidean distances in clr space.
    """
    def _clr(p: np.ndarray) -> np.ndarray:
        # Epsilon replacement
        p_reg = np.clip(p, eps, 1.0)
        p_reg = p_reg / np.sum(p_reg, axis=-1, keepdims=True)
        log_p = np.log(p_reg)
        geom_mean = np.mean(log_p, axis=-1, keepdims=True)
        return log_p - geom_mean

    clr_true = _clr(y_true)
    clr_pred = _clr(y_pred)
    return np.linalg.norm(clr_true - clr_pred, axis=-1)


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Computes Kish's Effective Sample Size (ESS) for importance weights:
    ESS = (sum(w))^2 / sum(w^2)
    """
    w = np.asarray(weights, dtype=np.float64)
    sum_w = np.sum(w)
    sum_w_sq = np.sum(w ** 2)
    if sum_w_sq == 0:
        return 0.0
    return float((sum_w ** 2) / sum_w_sq)


def process_weights(
    raw_weights: np.ndarray,
    clip_percentile: Optional[float] = 95.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalizes and optionally clips importance weights to stabilize conformal inference.
    
    Returns:
        normalized_weights: (M,) weights summing to 1.
        diagnostics: dict with ESS, ESS fraction, max weight, and clipping threshold.
    """
    w = np.copy(raw_weights).astype(np.float64)
    w = np.maximum(w, 1e-8)
    
    threshold = float(np.percentile(w, clip_percentile)) if clip_percentile is not None else float(np.max(w))
    if clip_percentile is not None:
        w = np.clip(w, a_min=None, a_max=threshold)
        
    ess = compute_effective_sample_size(w)
    ess_frac = ess / len(w) if len(w) > 0 else 0.0
    
    norm_w = w / np.sum(w)
    return norm_w, {
        "ess": ess,
        "ess_fraction": ess_frac,
        "max_weight": float(np.max(w)),
        "clip_threshold": threshold
    }


def compute_conformal_quantile(
    scores: np.ndarray,
    alpha: float,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Computes split (unweighted) or weighted conformal quantile at miscoverage level alpha.
    
    Args:
        scores: (M,) calibration nonconformity scores.
        alpha: Miscoverage level in (0, 1).
        weights: Optional (M,) importance weights.
    Returns:
        Conformal quantile q_alpha.
    """
    M = len(scores)
    scores = np.asarray(scores, dtype=np.float64)
    
    if weights is None:
        # Standard split conformal: (1 - alpha) * (1 + 1/M) quantile
        k = int(np.ceil((1.0 - alpha) * (M + 1)))
        k = np.clip(k, 1, M)
        sorted_scores = np.sort(scores)
        return float(sorted_scores[k - 1])
    else:
        # Weighted conformal quantile
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / np.sum(weights)
        
        sort_idx = np.argsort(scores)
        sorted_scores = scores[sort_idx]
        sorted_weights = weights[sort_idx]
        
        cum_weights = np.cumsum(sorted_weights)
        target = 1.0 - alpha
        idx = np.searchsorted(cum_weights, target, side='left')
        idx = min(idx, M - 1)
        return float(sorted_scores[idx])


def decision_ambiguity_gate(
    y_pred_k: float,
    q_alpha_tv: float,
    threshold: float
) -> Dict[str, Any]:
    """
    Gates a spot based on clinical decision ambiguity for a target lineage (e.g., CD8+ T-cell).
    Under TV radius q_alpha, the plausible interval for lineage fraction k is:
    [max(0, y_pred_k - q_alpha), min(1, y_pred_k + q_alpha)].
    
    The spot is ambiguous (masked) iff the interval contains values BOTH > threshold and <= threshold.
    
    Args:
        y_pred_k: Estimated proportion for lineage k (e.g. 0.08).
        q_alpha_tv: Conformal Total Variation bound at level 1 - alpha.
        threshold: Clinical decision threshold (e.g. 0.05 for 'immune-hot').
        
    Returns:
        Dict with 'is_ambiguous' (bool), 'lower_bound', 'upper_bound', and 'unambiguous_decision'.
    """
    lower = max(0.0, float(y_pred_k - q_alpha_tv))
    upper = min(1.0, float(y_pred_k + q_alpha_tv))
    
    straddles = (lower <= threshold <= upper)
    
    if straddles:
        decision = "AMBIGUOUS"
    elif lower > threshold:
        decision = "UNANIMOUS_POSITIVE"
    else:
        decision = "UNANIMOUS_NEGATIVE"
        
    return {
        "is_ambiguous": straddles,
        "is_trusted": not straddles,
        "lower_bound": lower,
        "upper_bound": upper,
        "decision": decision
    }

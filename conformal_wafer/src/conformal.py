import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class ConformalPredictor:
    """
    Split Conformal Prediction for Multi-Class Wafer Defect Recognition.
    Implements:
    1. APS (Adaptive Prediction Sets - Romano et al. 2020)
    2. RAPS (Regularized Adaptive Prediction Sets - Angelopoulos et al. 2021)
    
    Guarantees finite-sample marginal coverage P(Y in C(X)) >= 1 - alpha.
    """
    def __init__(self, method="raps", k_reg=1, lam=0.01, randomize=False, seed=42):
        assert method in ["aps", "raps"], f"Method {method} not supported"
        self.method = method
        self.k_reg = k_reg
        self.lam = lam if method == "raps" else 0.0
        self.randomize = randomize
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_hat = {}

    def _compute_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity scores on calibration samples.
        probs: (N, K) softmax probabilities
        labels: (N,) true class indices
        """
        n, k = probs.shape
        # Sort probabilities descending
        sort_idx = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sort_idx, axis=1)

        # Find rank of true label
        ranks = np.where(sort_idx == labels[:, None])[1] + 1  # 1-indexed rank

        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            r = ranks[i]
            if self.randomize:
                u = self.rng.uniform(0, 1)
                cum_prev = cumsum_probs[i, r - 2] if r > 1 else 0.0
                curr_p = sorted_probs[i, r - 1]
                base_score = cum_prev + u * curr_p
            else:
                base_score = cumsum_probs[i, r - 1]

            reg = self.lam * max(0, r - self.k_reg)
            scores[i] = base_score + reg

        return scores

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray, alphas: List[float]):
        """
        Calculates exact conformal quantiles on calibration data.
        Level: ceil((n + 1) * (1 - alpha)) / n
        """
        scores = self._compute_scores(cal_probs, cal_labels)
        n = len(scores)
        sorted_scores = np.sort(scores)

        self.q_hat = {}
        for alpha in alphas:
            idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
            idx = min(max(idx, 0), n - 1)
            self.q_hat[alpha] = sorted_scores[idx]

        return self.q_hat

    def predict(self, test_probs: np.ndarray, alpha: float) -> List[List[int]]:
        """
        Generate prediction sets for test probabilities at significance level alpha.
        Returns a list of class index sets.
        """
        assert alpha in self.q_hat, f"Alpha {alpha} not calibrated"
        q = self.q_hat[alpha]
        n, k = test_probs.shape

        sort_idx = np.argsort(-test_probs, axis=1)
        sorted_probs = np.take_along_axis(test_probs, sort_idx, axis=1)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        prediction_sets = []
        for i in range(n):
            curr_set = []
            for r in range(1, k + 1):
                curr_class = sort_idx[i, r - 1]
                curr_set.append(int(curr_class))

                # Check stopping criterion
                if self.randomize:
                    u = self.rng.uniform(0, 1)
                    cum_prev = cumsum_probs[i, r - 2] if r > 1 else 0.0
                    curr_p = sorted_probs[i, r - 1]
                    score = cum_prev + u * curr_p + self.lam * max(0, r - self.k_reg)
                else:
                    score = cumsum_probs[i, r - 1] + self.lam * max(0, r - self.k_reg)

                if score >= q:
                    break
            prediction_sets.append(curr_set)

        return prediction_sets

def evaluate_conformal_predictions(
    pred_sets: List[List[int]],
    test_labels: np.ndarray,
    test_depths: np.ndarray
) -> Dict:
    """
    Evaluates marginal coverage, efficiency (set size), singleton %,
    and per-mix-depth coverage breakdown.
    """
    n = len(test_labels)
    covered = np.array([test_labels[i] in pred_sets[i] for i in range(n)])
    set_sizes = np.array([len(pred_sets[i]) for i in range(n)])

    overall_coverage = float(np.mean(covered))
    avg_set_size = float(np.mean(set_sizes))
    median_set_size = float(np.median(set_sizes))
    singleton_pct = float(np.mean(set_sizes == 1) * 100.0)

    # Breakdown by mix depth
    depth_stats = {}
    unique_depths = sorted(np.unique(test_depths))
    for d in unique_depths:
        mask = (test_depths == d)
        if np.sum(mask) > 0:
            depth_stats[int(d)] = {
                'count': int(np.sum(mask)),
                'coverage': float(np.mean(covered[mask])),
                'avg_set_size': float(np.mean(set_sizes[mask])),
                'median_set_size': float(np.median(set_sizes[mask])),
            }

    return {
        'marginal_coverage': overall_coverage,
        'avg_set_size': avg_set_size,
        'median_set_size': median_set_size,
        'singleton_pct': singleton_pct,
        'depth_stats': depth_stats,
        'covered_array': covered,
        'set_sizes_array': set_sizes
    }

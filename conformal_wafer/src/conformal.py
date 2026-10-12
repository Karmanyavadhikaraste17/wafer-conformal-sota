import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class ConformalPredictor:
    """
    Split Conformal Prediction for Multi-Class Wafer Defect Recognition.
    Implements:
    1. APS (Adaptive Prediction Sets - Romano et al. 2020)
    2. RAPS (Regularized Adaptive Prediction Sets - Angelopoulos et al. 2021)
    
    Supports:
    - allow_empty=True: Exact randomized conformal sets guaranteeing
      1 - alpha <= P(Y in C(X)) <= 1 - alpha + 1/(n_cal + 1).
    - allow_empty=False: Non-empty prediction sets (always includes at least top-1 argmax class).
    """
    def __init__(self, method="raps", k_reg=1, lam=0.01, randomize=True, allow_empty=False, seed=42):
        assert method in ["aps", "raps"], f"Method {method} not supported"
        self.method = method
        self.k_reg = k_reg
        self.lam = lam if method == "raps" else 0.0
        self.randomize = randomize
        self.allow_empty = allow_empty
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_hat = {}

    def _compute_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity scores on calibration samples:
        S_i = sum_{j=1}^{r-1} pi_{(j)}(x_i) + U_i * pi_{(r)}(x_i) + lambda * max(0, r - k_reg)
        """
        n, k = probs.shape
        sort_idx = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sort_idx, axis=1)

        # 1-indexed rank of true label
        ranks = np.where(sort_idx == labels[:, None])[1] + 1
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            r = ranks[i]
            cum_prev = cumsum_probs[i, r - 2] if r > 1 else 0.0
            curr_p = sorted_probs[i, r - 1]
            if self.randomize:
                u = self.rng.uniform(0.0, 1.0)
                base_score = cum_prev + u * curr_p
            else:
                base_score = cum_prev + curr_p

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
            self.q_hat[alpha] = float(sorted_scores[idx])

        return self.q_hat

    def predict(self, test_probs: np.ndarray, alpha: float) -> List[List[int]]:
        """
        Generate prediction sets for test probabilities at significance level alpha.
        Following Angelopoulos et al. (ICLR 2021):
        Include rank r if S(x, r, u) <= q_hat.
        If allow_empty=False and set is empty, include top-1 argmax class.
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
            u = self.rng.uniform(0.0, 1.0) if self.randomize else 1.0
            for r in range(1, k + 1):
                cum_prev = cumsum_probs[i, r - 2] if r > 1 else 0.0
                curr_p = sorted_probs[i, r - 1]
                score = cum_prev + u * curr_p + self.lam * max(0, r - self.k_reg)
                if score <= q:
                    curr_set.append(int(sort_idx[i, r - 1]))
                else:
                    break

            if not self.allow_empty and len(curr_set) == 0:
                curr_set.append(int(sort_idx[i, 0]))

            prediction_sets.append(curr_set)

        return prediction_sets

def evaluate_conformal_predictions(
    pred_sets: List[List[int]],
    test_labels: np.ndarray,
    test_depths: np.ndarray
) -> Dict:
    """
    Evaluates marginal coverage, efficiency (set size), singleton %, empty %,
    and per-mix-depth coverage breakdown with absolute counts.
    """
    n = len(test_labels)
    covered = np.array([test_labels[i] in pred_sets[i] for i in range(n)])
    set_sizes = np.array([len(pred_sets[i]) for i in range(n)])

    overall_coverage = float(np.mean(covered))
    avg_set_size = float(np.mean(set_sizes))
    median_set_size = float(np.median(set_sizes))
    singleton_pct = float(np.mean(set_sizes == 1) * 100.0)
    empty_pct = float(np.mean(set_sizes == 0) * 100.0)

    # Breakdown by mix depth
    depth_stats = {}
    unique_depths = sorted(np.unique(test_depths))
    for d in unique_depths:
        mask = (test_depths == d)
        cnt = int(np.sum(mask))
        if cnt > 0:
            cov_cnt = int(np.sum(covered[mask]))
            depth_stats[int(d)] = {
                'count': cnt,
                'covered_count': cov_cnt,
                'coverage': float(np.mean(covered[mask])),
                'avg_set_size': float(np.mean(set_sizes[mask])),
                'median_set_size': float(np.median(set_sizes[mask])),
                'singleton_pct': float(np.mean(set_sizes[mask] == 1) * 100.0)
            }

    return {
        'marginal_coverage': overall_coverage,
        'covered_count': int(np.sum(covered)),
        'total_count': n,
        'avg_set_size': avg_set_size,
        'median_set_size': median_set_size,
        'singleton_pct': singleton_pct,
        'empty_pct': empty_pct,
        'depth_stats': depth_stats,
        'covered_array': covered,
        'set_sizes_array': set_sizes
    }

"""
Conformal prediction and calibration logic for Chest X-Ray classification.
Supports Least Ambiguous Set-Valued Classifiers (LAC), Adaptive Prediction Sets (APS),
and MAPIE-compatible interfaces.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_nonconformity_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    method: str = "lac",
) -> np.ndarray:
    """
    Computes nonconformity scores for calibration set.

    Args:
        probs: Array of predicted probabilities of shape (n_samples, n_classes).
        labels: True integer class indices of shape (n_samples,).
        method: Scoring method ('lac' for 1 - prob[true_class], 'aps' for cumulative sorted sum).

    Returns:
        Array of conformity/nonconformity scores of shape (n_samples,).
    """
    n_samples = probs.shape[0]
    labels = np.asarray(labels, dtype=int)

    if method.lower() == "lac":
        # LAC nonconformity score: 1 - P(Y = y_true | X)
        true_class_probs = probs[np.arange(n_samples), labels]
        scores = 1.0 - true_class_probs
        return scores

    elif method.lower() == "aps":
        # Adaptive Prediction Sets (APS): cumulative probability until true class
        # Sort descending
        sorted_indices = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        scores = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            rank = np.where(sorted_indices[i] == labels[i])[0][0]
            scores[i] = cumsum_probs[i, rank]
        return scores

    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods: 'lac', 'aps'.")


def calibrate_threshold(
    scores: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Computes the conformal calibration quantile threshold q_hat.

    Formula: q_hat = Quantile_{ (ceil((n + 1) * (1 - alpha)) / n) }(scores)

    Args:
        scores: Nonconformity scores on the calibration set of shape (n_samples,).
        alpha: Error rate budget (confidence level = 1 - alpha).

    Returns:
        Calibrated threshold q_hat.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Scores array cannot be empty.")

    # Conformal quantile level with finite-sample correction
    quantile_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    quantile_level = min(1.0, max(0.0, quantile_level))

    q_hat = float(np.quantile(scores, quantile_level, method="higher"))
    return q_hat


def predict_conformal_sets(
    probs: np.ndarray,
    q_hat: float,
    method: str = "lac",
    class_names: Optional[List[str]] = None,
) -> List[List[Union[int, str]]]:
    """
    Constructs prediction sets C(x) for test probabilities using calibrated threshold.

    Args:
        probs: Array of predicted probabilities of shape (n_samples, n_classes).
        q_hat: Calibrated threshold quantile from calibrate_threshold().
        method: Conformal method ('lac' or 'aps').
        class_names: Optional list of class names to map integer indices to strings.

    Returns:
        List of prediction sets for each test sample.
    """
    n_samples, n_classes = probs.shape
    prediction_sets = []

    if method.lower() == "lac":
        # Include classes where (1 - prob) <= q_hat  <=>  prob >= 1 - q_hat
        threshold = 1.0 - q_hat
        for i in range(n_samples):
            selected = np.where(probs[i] >= threshold)[0]
            # Guarantee non-empty set by picking argmax if none meet threshold
            if len(selected) == 0:
                selected = np.array([np.argmax(probs[i])])
            if class_names is not None:
                prediction_sets.append([class_names[idx] for idx in selected])
            else:
                prediction_sets.append(selected.tolist())

    elif method.lower() == "aps":
        sorted_indices = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
        cumsum_probs = np.cumsum(sorted_probs, axis=1)

        for i in range(n_samples):
            # Include classes until cumsum exceeds q_hat
            cutoff_idx = np.searchsorted(cumsum_probs[i], q_hat)
            selected = sorted_indices[i, : max(1, cutoff_idx + 1)]
            if class_names is not None:
                prediction_sets.append([class_names[idx] for idx in selected])
            else:
                prediction_sets.append(selected.tolist())
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return prediction_sets


def evaluate_conformal_coverage(
    prediction_sets: List[List[Union[int, str]]],
    true_labels: Union[np.ndarray, List[Union[int, str]]],
) -> Dict[str, float]:
    """
    Calculates empirical coverage and average set size.

    Args:
        prediction_sets: Conformal prediction sets for test samples.
        true_labels: True ground-truth labels for test samples.

    Returns:
        Dictionary with 'coverage' and 'avg_set_size'.
    """
    n = len(prediction_sets)
    if n == 0:
        return {"coverage": 0.0, "avg_set_size": 0.0}

    covered_count = sum(1 for pred_set, y_true in zip(prediction_sets, true_labels) if y_true in pred_set)
    set_sizes = [len(s) for s in prediction_sets]

    return {
        "coverage": float(covered_count / n),
        "avg_set_size": float(np.mean(set_sizes)),
    }

"""
Evaluation metrics for Conformal Prediction and Explainable AI (Grad-CAM) alignment.
"""

from typing import Dict, List, Optional, Sequence, Union
import numpy as np


def calculate_coverage(
    prediction_sets: Sequence[Sequence[Union[int, str]]],
    true_labels: Sequence[Union[int, str]],
) -> float:
    """
    Computes empirical coverage: fraction of test samples where y_true in C(x).

    Args:
        prediction_sets: Predicted sets C(x).
        true_labels: Ground truth labels.

    Returns:
        Empirical coverage in range [0.0, 1.0].
    """
    if len(prediction_sets) == 0:
        return 0.0

    hits = sum(1 for p_set, y_true in zip(prediction_sets, true_labels) if y_true in p_set)
    return float(hits / len(prediction_sets))


def calculate_average_set_size(
    prediction_sets: Sequence[Sequence[Union[int, str]]],
) -> float:
    """
    Computes average size of prediction sets |C(x)| (efficiency metric).

    Args:
        prediction_sets: Predicted sets C(x).

    Returns:
        Mean cardinality of prediction sets.
    """
    if len(prediction_sets) == 0:
        return 0.0

    return float(np.mean([len(s) for s in prediction_sets]))


def calculate_set_size_distribution(
    prediction_sets: Sequence[Sequence[Union[int, str]]],
) -> Dict[int, float]:
    """
    Calculates proportion of prediction sets of each cardinality (e.g. 1, 2, 3+).

    Args:
        prediction_sets: Predicted sets C(x).

    Returns:
        Dictionary mapping set size to proportion.
    """
    if len(prediction_sets) == 0:
        return {}

    sizes = [len(s) for s in prediction_sets]
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    total = len(prediction_sets)

    return {int(sz): float(cnt / total) for sz, cnt in zip(unique_sizes, counts)}


def conformal_classification_metrics(
    prediction_sets: Sequence[Sequence[Union[int, str]]],
    true_labels: Sequence[Union[int, str]],
    alpha: float,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Summarizes key conformal prediction metrics.

    Args:
        prediction_sets: Conformal prediction sets.
        true_labels: True ground-truth labels.
        alpha: Targeted error budget (1 - alpha target coverage).

    Returns:
        Dictionary with empirical coverage, target coverage, violation error,
        mean set size, and size distribution.
    """
    cov = calculate_coverage(prediction_sets, true_labels)
    target_cov = 1.0 - alpha
    avg_size = calculate_average_set_size(prediction_sets)
    dist = calculate_set_size_distribution(prediction_sets)

    return {
        "target_coverage": target_cov,
        "empirical_coverage": cov,
        "coverage_gap": cov - target_cov,
        "avg_set_size": avg_size,
        "size_distribution": dist,
    }


def cam_conformal_alignment_score(
    cam_heatmaps: Dict[Union[int, str], np.ndarray],
    conformal_set: Sequence[Union[int, str]],
    method: str = "iou",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Measures visual alignment / disagreement among Grad-CAM heatmaps
    for candidate pathologies included in the conformal prediction set C(x).

    When |C(x)| > 1, this evaluates whether different candidate diagnoses
    attend to the same anatomical region (shared visual finding) or distinct areas.

    Args:
        cam_heatmaps: Dictionary mapping class label to 2D Grad-CAM heatmap [0, 1].
        conformal_set: Candidate classes in C(x) for this sample.
        method: Comparison method ('iou' or 'cosine').
        threshold: Saliency binarization threshold for IoU calculation.

    Returns:
        Dictionary containing mean pairwise similarity and ambiguity metrics.
    """
    classes_in_set = [c for c in conformal_set if c in cam_heatmaps]
    k = len(classes_in_set)

    if k <= 1:
        return {
            "num_candidates": k,
            "mean_pairwise_similarity": 1.0 if k == 1 else 0.0,
            "is_ambiguous": 0.0,
        }

    similarities = []
    for i in range(k):
        for j in range(i + 1, k):
            map_i = np.nan_to_num(cam_heatmaps[classes_in_set[i]])
            map_j = np.nan_to_num(cam_heatmaps[classes_in_set[j]])

            if method.lower() == "iou":
                bin_i = map_i >= threshold
                bin_j = map_j >= threshold
                intersection = np.logical_and(bin_i, bin_j).sum()
                union = np.logical_or(bin_i, bin_j).sum()
                iou = intersection / union if union > 0 else 1.0
                similarities.append(iou)
            elif method.lower() == "cosine":
                vec_i = map_i.flatten()
                vec_j = map_j.flatten()
                norm_i = np.linalg.norm(vec_i)
                norm_j = np.linalg.norm(vec_j)
                if norm_i > 0 and norm_j > 0:
                    sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))
                else:
                    sim = 0.0
                similarities.append(sim)
            else:
                raise ValueError(f"Unknown alignment method: {method}")

    mean_sim = float(np.mean(similarities)) if similarities else 0.0

    return {
        "num_candidates": k,
        "mean_pairwise_similarity": mean_sim,
        # High ambiguity when multiple classes are in C(x) but look at different regions
        "spatial_disagreement": float(1.0 - mean_sim),
    }

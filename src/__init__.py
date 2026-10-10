"""
CXR Conformal XAI package.

Modules for conformal prediction, calibration, and evaluation metrics with Grad-CAM.
"""

from .conformal import (
    compute_nonconformity_scores,
    calibrate_threshold,
    predict_conformal_sets,
    evaluate_conformal_coverage,
)
from .metrics import (
    calculate_coverage,
    calculate_average_set_size,
    conformal_classification_metrics,
    cam_conformal_alignment_score,
)

__all__ = [
    "compute_nonconformity_scores",
    "calibrate_threshold",
    "predict_conformal_sets",
    "evaluate_conformal_coverage",
    "calculate_coverage",
    "calculate_average_set_size",
    "conformal_classification_metrics",
    "cam_conformal_alignment_score",
]

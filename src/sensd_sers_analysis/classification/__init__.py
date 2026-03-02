"""
Phase 2: Serotyping & Classification.

Uses clean Phase 1 data (Pass sensors, inlier points only) to train baseline
ML models for 3-class classification: ST, SE, Rinsate.
"""

from .data_prep import prepare_phase2_data
from .models import ClassificationResult, train_classifiers
from .plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
)

__all__ = [
    "ClassificationResult",
    "prepare_phase2_data",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_pca_classification",
    "train_classifiers",
]

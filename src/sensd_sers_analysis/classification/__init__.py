"""
Serotype classification: baseline ML for ST / SE / Rinsate.

Uses QA-clean rows (Pass sensors, inlier points only) to train baseline models.
"""

from .data_prep import prepare_classification_dataset
from .models import ClassificationResult, train_classifiers, train_classifiers_on_arrays
from .plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pca_classification,
)

__all__ = [
    "ClassificationResult",
    "prepare_classification_dataset",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_pca_classification",
    "train_classifiers",
    "train_classifiers_on_arrays",
]

"""Serotype classifiers with sensor holdout and training-only model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sensd_sers_analysis.config import (
    CLASSIFICATION_HYPERPARAMETER_TUNING,
    CLASSIFICATION_RANDOM_STATE,
    CLASSIFICATION_RF_N_ESTIMATORS,
    CLASSIFICATION_RF_SEARCH_MAX_DEPTH,
    CLASSIFICATION_RF_SEARCH_MIN_SAMPLES_LEAF,
    CLASSIFICATION_RF_SEARCH_N_ESTIMATORS,
    CLASSIFICATION_SVM_SEARCH_C,
    CLASSIFICATION_SVM_SEARCH_GAMMA,
    CLASSIFICATION_TEST_SIZE,
    CLASSIFICATION_TUNING_CV_SPLITS,
    CLASSIFICATION_TUNING_MIN_TRAIN_SAMPLES,
    CLASSIFICATION_TUNING_RANDOM_SEARCH_ITER,
)
from sensd_sers_analysis.modeling import feature_matrix, fit_model_pipeline
from sensd_sers_analysis.splits import group_train_test_indices


@dataclass
class ClassificationResult:
    """Held-out metrics with a pipeline that owns its input transformation."""

    model_name: str
    model: object
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    feature_importances: np.ndarray | None = None
    scaler: StandardScaler | None = None
    best_params: dict[str, Any] | None = None
    cv_score: float | None = None
    train_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None
    split_seed: int | None = None


def _classification_cv(y: np.ndarray, groups: np.ndarray | None, seed: int):
    """Return one shared viable fold plan; unsupported training classes skip search."""
    if not CLASSIFICATION_HYPERPARAMETER_TUNING or len(y) < CLASSIFICATION_TUNING_MIN_TRAIN_SAMPLES:
        return None
    if groups is not None:
        count = min(CLASSIFICATION_TUNING_CV_SPLITS, len(np.unique(groups)))
        if count < 2:
            return None
        splits = list(GroupKFold(count).split(np.zeros((len(y), 1)), y, groups))
    else:
        _, counts = np.unique(y, return_counts=True)
        count = min(CLASSIFICATION_TUNING_CV_SPLITS, int(counts.min()))
        if count < 2:
            return None
        splits = list(
            StratifiedKFold(count, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y)
        )
    labels = set(y)
    if any(set(y[train]) != labels for train, _ in splits):
        return None
    return splits


def train_classifiers_on_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    random_state: int = CLASSIFICATION_RANDOM_STATE,
    groups_train: np.ndarray | None = None,
) -> tuple[ClassificationResult, ClassificationResult]:
    """Fit RF/SVM on a caller-defined split and report held-out metrics.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Finite unscaled selected features; measured zeros remain zero.
    y_train, y_test : np.ndarray
        Class labels. Test classes must belong to the training vocabulary.
    feature_names : list[str]
        Names aligned with the feature columns.
    random_state : int
        Estimator/search seed.
    groups_train : np.ndarray, optional
        Training sensor IDs for grouped tuning. Without IDs, caller-defined
        independent arrays use stratified row CV.

    Returns
    -------
    tuple[ClassificationResult, ClassificationResult]
        RF and SVM results with training CV selection scores when feasible.
        Pipelines predict from unscaled inputs; ``scaler`` is unused.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=object)
    y_test = np.asarray(y_test, dtype=object)
    if (
        X_train.ndim != 2
        or X_test.ndim != 2
        or X_train.shape[1] != len(feature_names)
        or X_test.shape[1] != len(feature_names)
    ):
        raise ValueError("Classifier feature matrices must align with the selected feature names.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test) or not len(y_test):
        raise ValueError("Classifier labels and nonempty split rows must align.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(
            "Selected classifier inputs must be finite; missing measurements are unavailable."
        )
    classes = sorted(pd.unique(y_train).tolist())
    if len(classes) < 2:
        raise ValueError("Classification requires at least two training classes.")
    unknown = set(y_test) - set(classes)
    if unknown:
        raise ValueError(f"Held-out classes have no training support: {sorted(unknown)}.")
    if groups_train is not None:
        groups_train = np.asarray(groups_train, dtype=object)
        if (
            groups_train.shape != y_train.shape
            or pd.isna(groups_train).any()
            or any(not str(group).strip() for group in groups_train)
        ):
            raise ValueError("Training group IDs must align with rows and be non-missing/nonempty.")
        groups_train = np.asarray([str(group).strip() for group in groups_train], dtype=object)
    cv = _classification_cv(y_train, groups_train, random_state)
    specifications = [
        (
            "Random Forest",
            RandomForestClassifier(
                random_state=random_state, n_estimators=CLASSIFICATION_RF_N_ESTIMATORS
            ),
            {
                "n_estimators": list(CLASSIFICATION_RF_SEARCH_N_ESTIMATORS),
                "max_depth": list(CLASSIFICATION_RF_SEARCH_MAX_DEPTH),
                "min_samples_leaf": list(CLASSIFICATION_RF_SEARCH_MIN_SAMPLES_LEAF),
            },
        ),
        (
            "SVM (RBF)",
            SVC(kernel="rbf", random_state=random_state),
            {
                "C": list(CLASSIFICATION_SVM_SEARCH_C),
                "gamma": list(CLASSIFICATION_SVM_SEARCH_GAMMA),
            },
        ),
    ]
    results = []
    for name, estimator, parameters in specifications:
        model, best_params, cv_score = fit_model_pipeline(
            estimator,
            X_train,
            y_train,
            cv=cv,
            parameters=parameters,
            scoring="f1_weighted",
            n_iter=CLASSIFICATION_TUNING_RANDOM_SEARCH_ITER,
            random_state=random_state,
        )
        prediction = model.predict(X_test)
        results.append(
            ClassificationResult(
                model_name=name,
                model=model,
                y_true=y_test,
                y_pred=prediction,
                accuracy=float(accuracy_score(y_test, prediction)),
                precision=float(
                    precision_score(y_test, prediction, average="weighted", zero_division=0)
                ),
                recall=float(recall_score(y_test, prediction, average="weighted", zero_division=0)),
                f1=float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
                confusion_matrix=confusion_matrix(y_test, prediction, labels=classes).astype(int),
                class_names=classes,
                feature_names=feature_names,
                feature_importances=getattr(
                    model.named_steps["model"], "feature_importances_", None
                ),
                best_params=best_params,
                cv_score=cv_score,
            )
        )
    return results[0], results[1]


def train_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    *,
    test_size: float = CLASSIFICATION_TEST_SIZE,
    random_state: int = CLASSIFICATION_RANDOM_STATE,
    group_col: str = "sensor_id",
) -> tuple[ClassificationResult, ClassificationResult]:
    """Evaluate RF/SVM on a sensor-group holdout of identity-eligible rows.

    Learned scaling and tuning use training folds only. Feature inputs must be
    independently computed per spectrum; cohort PCA belongs to exploration.
    The supplied dataframe's row labels/order remain unchanged.
    """
    X = feature_matrix(df, feature_cols)
    y = df[target_col].map(str).to_numpy(dtype=object)
    groups = df[group_col].to_numpy(dtype=object)
    train, test = group_train_test_indices(groups, test_size=test_size, random_state=random_state)
    results = train_classifiers_on_arrays(
        X[train],
        X[test],
        y[train],
        y[test],
        feature_cols,
        random_state=random_state,
        groups_train=groups[train],
    )
    for result in results:
        result.train_indices = train
        result.test_indices = test
        result.split_seed = random_state
    return results

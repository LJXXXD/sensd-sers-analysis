"""
Baseline serotype classification: Random Forest and SVM classifiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
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

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Results from training a serotype classifier."""

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
    feature_importances: Optional[np.ndarray] = None
    scaler: Optional[StandardScaler] = None
    best_params: Optional[dict[str, Any]] = None


def _effective_cv_splits(y_train: np.ndarray, requested: int) -> int:
    """
    Number of stratified CV folds that are valid for y_train.

    Returns
    -------
    int
        ``0`` if stratified CV is not viable, otherwise at least ``2``.
    """
    _, counts = np.unique(y_train, return_counts=True)
    min_c = int(counts.min())
    if min_c < 2:
        return 0
    return max(2, min(int(requested), min_c))


def _should_run_hyperparameter_search(n_train: int, y_train: np.ndarray) -> bool:
    if not CLASSIFICATION_HYPERPARAMETER_TUNING:
        return False
    if n_train < CLASSIFICATION_TUNING_MIN_TRAIN_SAMPLES:
        logger.info(
            "Classification hyperparameter search skipped: train size %d < %d",
            n_train,
            CLASSIFICATION_TUNING_MIN_TRAIN_SAMPLES,
        )
        return False
    if _effective_cv_splits(y_train, CLASSIFICATION_TUNING_CV_SPLITS) == 0:
        logger.info(
            "Classification hyperparameter search skipped: need at least 2 training samples "
            "per class for stratified CV."
        )
        return False
    return True


def _fit_random_forest(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    *,
    run_hyperparameter_search: bool,
) -> tuple[RandomForestClassifier, Optional[dict[str, Any]]]:
    if run_hyperparameter_search:
        n_splits = _effective_cv_splits(y_train, CLASSIFICATION_TUNING_CV_SPLITS)
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        base = RandomForestClassifier(random_state=random_state)
        param_distributions = {
            "n_estimators": list(CLASSIFICATION_RF_SEARCH_N_ESTIMATORS),
            "max_depth": list(CLASSIFICATION_RF_SEARCH_MAX_DEPTH),
            "min_samples_leaf": list(CLASSIFICATION_RF_SEARCH_MIN_SAMPLES_LEAF),
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=min(CLASSIFICATION_TUNING_RANDOM_SEARCH_ITER, 48),
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_s, y_train)
        best = search.best_estimator_
        return best, dict(search.best_params_)
    rf = RandomForestClassifier(
        random_state=random_state,
        n_estimators=CLASSIFICATION_RF_N_ESTIMATORS,
    )
    rf.fit(X_train_s, y_train)
    return rf, None


def _fit_svc(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    *,
    run_hyperparameter_search: bool,
) -> tuple[SVC, Optional[dict[str, Any]]]:
    if run_hyperparameter_search:
        n_splits = _effective_cv_splits(y_train, CLASSIFICATION_TUNING_CV_SPLITS)
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        base = SVC(kernel="rbf", random_state=random_state)
        param_distributions = {
            "C": list(CLASSIFICATION_SVM_SEARCH_C),
            "gamma": list(CLASSIFICATION_SVM_SEARCH_GAMMA),
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=min(CLASSIFICATION_TUNING_RANDOM_SEARCH_ITER, 20),
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_s, y_train)
        best = search.best_estimator_
        return best, dict(search.best_params_)
    svm = SVC(kernel="rbf", random_state=random_state)
    svm.fit(X_train_s, y_train)
    return svm, None


def _classification_results_from_scaled(
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    scaler: StandardScaler,
    *,
    random_state: int,
) -> tuple[ClassificationResult, ClassificationResult]:
    """
    Fit RF + SVM on scaled training rows and evaluate on scaled test rows.

    Parameters
    ----------
    X_train_s, X_test_s:
        Standardized feature matrices.
    y_train, y_test:
        String class labels.
    feature_names:
        Feature names aligned with columns of ``X_*``.
    scaler:
        Fitted ``StandardScaler`` used to produce ``X_train_s`` / ``X_test_s``.
    random_state:
        Random seed for estimators and search.

    Returns
    -------
    tuple[ClassificationResult, ClassificationResult]
        ``(rf_result, svm_result)``.
    """
    class_names = sorted(pd.unique(np.concatenate([y_train, y_test])).tolist())

    run_hp = _should_run_hyperparameter_search(len(X_train_s), y_train)

    rf, rf_params = _fit_random_forest(
        X_train_s, y_train, random_state, run_hyperparameter_search=run_hp
    )
    y_pred_rf = rf.predict(X_test_s)

    svm, svm_params = _fit_svc(X_train_s, y_train, random_state, run_hyperparameter_search=run_hp)
    y_pred_svm = svm.predict(X_test_s)

    rf_result = ClassificationResult(
        model_name="Random Forest",
        model=rf,
        y_true=y_test,
        y_pred=y_pred_rf,
        accuracy=float(accuracy_score(y_test, y_pred_rf)),
        precision=float(precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)),
        recall=float(recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)),
        f1=float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)),
        confusion_matrix=confusion_matrix(y_test, y_pred_rf, labels=class_names).astype(int),
        class_names=class_names,
        feature_names=feature_names,
        feature_importances=rf.feature_importances_,
        scaler=scaler,
        best_params=rf_params,
    )

    svm_result = ClassificationResult(
        model_name="SVM (RBF)",
        model=svm,
        y_true=y_test,
        y_pred=y_pred_svm,
        accuracy=float(accuracy_score(y_test, y_pred_svm)),
        precision=float(precision_score(y_test, y_pred_svm, average="weighted", zero_division=0)),
        recall=float(recall_score(y_test, y_pred_svm, average="weighted", zero_division=0)),
        f1=float(f1_score(y_test, y_pred_svm, average="weighted", zero_division=0)),
        confusion_matrix=confusion_matrix(y_test, y_pred_svm, labels=class_names).astype(int),
        class_names=class_names,
        feature_names=feature_names,
        feature_importances=None,
        scaler=scaler,
        best_params=svm_params,
    )

    return rf_result, svm_result


def train_classifiers_on_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    random_state: int = CLASSIFICATION_RANDOM_STATE,
) -> tuple[ClassificationResult, ClassificationResult]:
    """
    Train serotype classifiers on a caller-defined train/test feature split.

    Use when the split is already fixed (e.g. group-by-sensor holdout). Fits a
    new ``StandardScaler`` on the training rows only.

    Parameters
    ----------
    X_train, X_test:
        Unscaled feature matrices (same columns as ``feature_names``).
    y_train, y_test:
        String class labels (e.g. serotype names and ``Rinsate``).
    feature_names:
        Names aligned with columns of ``X_train`` / ``X_test``.
    random_state:
        Random seed.

    Returns
    -------
    tuple[ClassificationResult, ClassificationResult]
        ``(rf_result, svm_result)``.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return _classification_results_from_scaled(
        X_train_s,
        X_test_s,
        y_train,
        y_test,
        feature_names,
        scaler,
        random_state=random_state,
    )


def train_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    *,
    test_size: float = CLASSIFICATION_TEST_SIZE,
    random_state: int = CLASSIFICATION_RANDOM_STATE,
) -> tuple[ClassificationResult, ClassificationResult]:
    """
    Train Random Forest and SVM with 80/20 stratified split.

    When ``CLASSIFICATION_HYPERPARAMETER_TUNING`` is True and the training split is at
    least ``CLASSIFICATION_TUNING_MIN_TRAIN_SAMPLES`` rows, each model is tuned with
    ``RandomizedSearchCV`` (stratified folds) on the training data only; the
    held-out test set is used once for the reported metrics.

    Parameters
    ----------
    df:
        Clean DataFrame with feature columns and target.
    feature_cols:
        Feature column names.
    target_col:
        Target column (dynamic serotypes + ``Rinsate``).
    test_size:
        Fraction for test set.
    random_state:
        Random seed.

    Returns
    -------
    tuple[ClassificationResult, ClassificationResult]
        ``(rf_result, svm_result)``. RF result includes ``feature_importances``.
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No feature columns found. Needed: {feature_cols}")

    # Fill NaN in peak columns (and any other features) with 0 before ML.
    # Do NOT use dropna—legitimate Rinsate/low-CFU samples often have NaN peaks.
    # Arrow-backed columns: .values breaks sklearn train_test_split indexing.
    X = df[available].fillna(0).to_numpy(dtype=np.float64, copy=False)
    y = df[target_col].map(str).to_numpy(dtype=object)
    class_names = sorted(pd.unique(y).tolist())

    min_per_class = 2
    if any((y == c).sum() < min_per_class for c in class_names):
        raise ValueError(
            f"Need at least {min_per_class} samples per class for stratified split. "
            f"Counts: {dict(zip(class_names, [(y == c).sum() for c in class_names]))}."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return _classification_results_from_scaled(
        X_train_s,
        X_test_s,
        y_train,
        y_test,
        available,
        scaler,
        random_state=random_state,
    )

"""
Phase 2 baseline ML: Random Forest and SVM classifiers.
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
    PHASE2_HYPERPARAMETER_TUNING,
    PHASE2_RANDOM_STATE,
    PHASE2_RF_N_ESTIMATORS,
    PHASE2_RF_SEARCH_MAX_DEPTH,
    PHASE2_RF_SEARCH_MIN_SAMPLES_LEAF,
    PHASE2_RF_SEARCH_N_ESTIMATORS,
    PHASE2_SVM_SEARCH_C,
    PHASE2_SVM_SEARCH_GAMMA,
    PHASE2_TEST_SIZE,
    PHASE2_TUNING_CV_SPLITS,
    PHASE2_TUNING_MIN_TRAIN_SAMPLES,
    PHASE2_TUNING_RANDOM_SEARCH_ITER,
)

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Results from training a Phase 2 classifier."""

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
    if not PHASE2_HYPERPARAMETER_TUNING:
        return False
    if n_train < PHASE2_TUNING_MIN_TRAIN_SAMPLES:
        logger.info(
            "Phase 2 hyperparameter search skipped: train size %d < %d",
            n_train,
            PHASE2_TUNING_MIN_TRAIN_SAMPLES,
        )
        return False
    if _effective_cv_splits(y_train, PHASE2_TUNING_CV_SPLITS) == 0:
        logger.info(
            "Phase 2 hyperparameter search skipped: need at least 2 training samples "
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
        n_splits = _effective_cv_splits(y_train, PHASE2_TUNING_CV_SPLITS)
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        base = RandomForestClassifier(random_state=random_state)
        param_distributions = {
            "n_estimators": list(PHASE2_RF_SEARCH_N_ESTIMATORS),
            "max_depth": list(PHASE2_RF_SEARCH_MAX_DEPTH),
            "min_samples_leaf": list(PHASE2_RF_SEARCH_MIN_SAMPLES_LEAF),
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=min(PHASE2_TUNING_RANDOM_SEARCH_ITER, 48),
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
        n_estimators=PHASE2_RF_N_ESTIMATORS,
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
        n_splits = _effective_cv_splits(y_train, PHASE2_TUNING_CV_SPLITS)
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        base = SVC(kernel="rbf", random_state=random_state)
        param_distributions = {
            "C": list(PHASE2_SVM_SEARCH_C),
            "gamma": list(PHASE2_SVM_SEARCH_GAMMA),
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_distributions,
            n_iter=min(PHASE2_TUNING_RANDOM_SEARCH_ITER, 20),
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


def train_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    *,
    test_size: float = PHASE2_TEST_SIZE,
    random_state: int = PHASE2_RANDOM_STATE,
) -> tuple[ClassificationResult, ClassificationResult]:
    """
    Train Random Forest and SVM with 80/20 stratified split.

    When ``PHASE2_HYPERPARAMETER_TUNING`` is True and the training split is at
    least ``PHASE2_TUNING_MIN_TRAIN_SAMPLES`` rows, each model is tuned with
    ``RandomizedSearchCV`` (stratified folds) on the training data only; the
    held-out test set is used once for the reported metrics.

    Parameters
    ----------
    df:
        Clean DataFrame with feature columns and target.
    feature_cols:
        Feature column names.
    target_col:
        Target column (ST, SE, Rinsate).
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
        feature_names=available,
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
        feature_names=available,
        feature_importances=None,
        scaler=scaler,
        best_params=svm_params,
    )

    return rf_result, svm_result

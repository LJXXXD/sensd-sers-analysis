"""
Phase 2 baseline ML: Random Forest and SVM classifiers.
"""

from dataclasses import dataclass
from typing import Optional

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


def train_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ClassificationResult, ClassificationResult]:
    """
    Train Random Forest and SVM with 80/20 stratified split.

    Args:
        df: Clean DataFrame with feature columns and target.
        feature_cols: Feature column names.
        target_col: Target column (ST, SE, Rinsate).
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        (rf_result, svm_result). rf_result has feature_importances.
    """
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No feature columns found. Needed: {feature_cols}")

    # Fill NaN in peak columns (and any other features) with 0 before ML.
    # Do NOT use dropna—legitimate Rinsate/low-CFU samples often have NaN peaks.
    X = df[available].fillna(0).values
    y = df[target_col].astype(str).values
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

    rf = RandomForestClassifier(random_state=random_state, n_estimators=100)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)

    svm = SVC(kernel="rbf", random_state=random_state)
    svm.fit(X_train_s, y_train)
    y_pred_svm = svm.predict(X_test_s)

    rf_result = ClassificationResult(
        model_name="Random Forest",
        model=rf,
        y_true=y_test,
        y_pred=y_pred_rf,
        accuracy=float(accuracy_score(y_test, y_pred_rf)),
        precision=float(
            precision_score(y_test, y_pred_rf, average="weighted", zero_division=0)
        ),
        recall=float(
            recall_score(y_test, y_pred_rf, average="weighted", zero_division=0)
        ),
        f1=float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)),
        confusion_matrix=confusion_matrix(y_test, y_pred_rf, labels=class_names).astype(
            int
        ),
        class_names=class_names,
        feature_names=available,
        feature_importances=rf.feature_importances_,
        scaler=scaler,
    )

    svm_result = ClassificationResult(
        model_name="SVM",
        model=svm,
        y_true=y_test,
        y_pred=y_pred_svm,
        accuracy=float(accuracy_score(y_test, y_pred_svm)),
        precision=float(
            precision_score(y_test, y_pred_svm, average="weighted", zero_division=0)
        ),
        recall=float(
            recall_score(y_test, y_pred_svm, average="weighted", zero_division=0)
        ),
        f1=float(f1_score(y_test, y_pred_svm, average="weighted", zero_division=0)),
        confusion_matrix=confusion_matrix(
            y_test, y_pred_svm, labels=class_names
        ).astype(int),
        class_names=class_names,
        feature_names=available,
        feature_importances=None,
        scaler=scaler,
    )

    return rf_result, svm_result

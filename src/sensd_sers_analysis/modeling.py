"""Finite model inputs, fold-local preprocessing, and training-only selection."""

from __future__ import annotations

from math import prod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def feature_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Return selected finite predictors without changing rows, columns, or zeros.

    Missing selected measurements are unavailable, rather than implicit zeros.
    Optional metadata and unselected exploratory features do not affect this check.
    """
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Model features must be a nonempty, unique list.")
    missing = [name for name in feature_names if name not in df]
    if missing:
        raise ValueError(f"Missing selected model features: {missing}.")
    X = (
        df[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64, na_value=np.nan)
    )
    invalid = np.argwhere(~np.isfinite(X))
    if invalid.size:
        row, column = invalid[0]
        raise ValueError(
            f"Model input unavailable at source row {df.index[row]!r}, "
            f"feature {feature_names[column]!r}. Choose a covered feature/range; "
            "missing measurements cannot be replaced with zero."
        )
    return X


def fit_model_pipeline(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: list[tuple[np.ndarray, np.ndarray]] | None = None,
    parameters: dict[str, list] | None = None,
    scoring: str,
    n_iter: int,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any] | None, float | None]:
    """Fit scaling inside every search fold, then refit on all provided training rows.

    Returned pipelines predict from unscaled selected features. Searches use the
    caller's realized folds and scorer; any fit/score failure aborts the request.
    A missing CV plan uses the predeclared estimator defaults without a CV score.
    """
    pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    if cv is None:
        pipeline.fit(X, y)
        return pipeline, None, None
    distributions = {f"model__{key}": values for key, values in parameters.items()}
    search = RandomizedSearchCV(
        pipeline,
        distributions,
        n_iter=min(n_iter, prod(len(values) for values in distributions.values())),
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    scores = [
        np.asarray(values)
        for key, values in search.cv_results_.items()
        if key.endswith("_test_score")
    ]
    if any(not np.isfinite(values).all() for values in scores) or not np.isfinite(
        search.best_score_
    ):
        raise ValueError("Training CV produced a non-finite model-selection score.")
    params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    return search.best_estimator_, params, float(search.best_score_)


def select_by_training_cv(first: Any, second: Any) -> Any:
    """Choose the higher comparable CV score; without viable CV use the first reference.

    The first candidate is the predeclared Random Forest reference. Held-out
    accuracy/F1/RMSE never enters this decision. Search failures propagate before
    selection; they are not a reason to use the reference.
    """
    scores = (first.cv_score, second.cv_score)
    if any(score is not None and not np.isfinite(score) for score in scores):
        raise ValueError("Model selection requires finite training CV scores.")
    if all(score is None for score in scores):
        return first
    if any(score is None for score in scores):
        raise ValueError(
            "Candidate selection scores are not comparable: only one candidate has CV."
        )
    return first if scores[0] >= scores[1] else second

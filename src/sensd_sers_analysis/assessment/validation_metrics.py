"""
Validation metric tables matching the SENS-D Metrics tables.docx layout.

Builds three publication-ready summary tables:

1. **Concentration & repeatability** — counts, Repeatability CV%, identification Accuracy.
2. **Quantification** — predicted CFU ranges, FP/FN, Meet target?
3. **Consistency & reusability** — repeated-use drift, failures, reuse Accuracy.

ML columns use **repeated sensor-holdout** (≈80/20 sensors × ``VALIDATION_N_SPLITS``
rounds); reported rates are the **mean across folds**. Repeatability / reuse
statistics still use all clean rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sensd_sers_analysis.processing.metadata import sample_type_masks

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sensd_sers_analysis.assessment.consistency import coefficient_of_variation
from sensd_sers_analysis.assessment.degradation import prepare_degradation_data
from sensd_sers_analysis.assessment.sensor_assessment_regression import (
    get_global_model_consistency_qa,
)
from sensd_sers_analysis.config import (
    CLASSIFICATION_INLIER_FEATURE,
    CLASSIFICATION_RANDOM_STATE,
    CLASSIFICATION_RF_N_ESTIMATORS,
    REGRESSION_RANDOM_STATE,
    REGRESSION_TEST_SIZE,
    VALIDATION_ACCURACY_MIN_THRESHOLD,
    VALIDATION_MIN_EVAL_ROWS,
    VALIDATION_N_SPLITS,
)
from sensd_sers_analysis.processing import (
    add_target_concentration_group,
    extract_scalar_concentration,
)
from sensd_sers_analysis.regression.splits import iter_group_train_test_indices
from sensd_sers_analysis.utils import order_concentration_labels

logger = logging.getLogger(__name__)

_TARGET_GROUP_COL = "_target_group"

# Docx Table 1 (+ Concentration CV% / Eval N for transparency).
TABLE1_COLUMNS = [
    "Concentration (CFU/mL)",
    "Serovar / Sample Group",
    "No. of Sensors Tested",
    "Replicates per Sensor",
    "Total Tests",
    "Identification",
    "Repeatability CV% or SD",
    "Concentration CV%",
    "Accuracy (% Correct)",
    "Eval N",
]

# Docx Table 2 — quantification companion.
TABLE2_COLUMNS = [
    "Concentration (CFU/mL)",
    "Serovar / Sample Group",
    "Quantification Accuracy",
    "False Positive Rate",
    "False Negative Rate",
    "Meet Target?",
    "Eval N",
]

# Docx Table 3 — consistency / reusability.
TABLE3_COLUMNS = [
    "Serovar",
    "Concentration (CFU/ml)",
    "No. of Tested sensors",
    "Repeated Tests per Sensor (n)",
    "Total Tests",
    "Mean Signal Change First to Last Test (%)",
    "Repeatability CV%",
    "No. of Failed Sensors",
    "Average Uses Before Failure",
    "Accuracy Across Repeated Uses",
    "Eval N",
    "Reliability Notes",
]


@dataclass(frozen=True, slots=True)
class ValidationFoldPredictions:
    """One sensor-holdout round: predictions on all rows; metrics use ``eval_mask``."""

    y_pred: np.ndarray
    pred_log_conc: Optional[np.ndarray]
    eval_mask: np.ndarray
    n_train_sensors: int
    n_test_sensors: int


@dataclass
class ValidationPredictions:
    """
    Repeated sensor-holdout model outputs for validation tables.

    Each fold trains on held-in sensors and predicts all rows; ML metrics are
    averaged across folds using only that fold's held-out test-sensor rows.
    """

    y_true: np.ndarray
    folds: tuple[ValidationFoldPredictions, ...]
    sensor_holdout_available: bool
    n_splits: int

    @property
    def n_train_sensors(self) -> float:
        """Mean train-sensor count across folds."""
        if not self.folds:
            return 0.0
        return float(np.mean([f.n_train_sensors for f in self.folds]))

    @property
    def n_test_sensors(self) -> float:
        """Mean test-sensor count across folds."""
        if not self.folds:
            return 0.0
        return float(np.mean([f.n_test_sensors for f in self.folds]))

    @property
    def n_eval_rows(self) -> int:
        """Total held-out evaluation rows summed across folds."""
        return int(sum(int(f.eval_mask.sum()) for f in self.folds))


@dataclass
class ValidationTableArtifacts:
    """Container for the three Metrics-docx validation tables."""

    concentration_repeatability: pd.DataFrame
    quantification: pd.DataFrame
    consistency_reusability: pd.DataFrame
    n_classification_rows: int
    n_regression_rows: int
    predictions: Optional[ValidationPredictions] = None


def _scalar_concentration_series(df: pd.DataFrame) -> pd.Series:
    """Return numeric CFU/mL per row (raw metadata; used only as fallback)."""
    if "concentration" in df.columns:
        return extract_scalar_concentration(df["concentration"], df)
    return pd.Series(np.nan, index=df.index)


def _add_target_concentration_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ``_target_group`` for table grouping using the nominal target.

    Reuses :func:`~sensd_sers_analysis.processing.add_target_concentration_group`
    so the binning policy (target-first, with actual-concentration fallback)
    matches the rest of the pipeline. The resulting label is copied into the
    internal ``_TARGET_GROUP_COL`` used by the table builders.
    """
    out = df.copy()
    if "target_concentration_group" in out.columns:
        labels = out["target_concentration_group"].astype(str).str.strip()
    else:
        out = add_target_concentration_group(out)
        labels = out["target_concentration_group"].astype(str).str.strip()
    invalid = labels.isin(("", "nan", "None", "NaN", "<NA>"))
    out[_TARGET_GROUP_COL] = labels.mask(invalid, "Unknown")
    return out


def _sorted_target_groups(series: pd.Series) -> list[str]:
    """Return unique target concentration_group labels in natural order."""
    unique = [
        g
        for g in series.dropna().astype(str).str.strip().unique().tolist()
        if g and g.lower() not in ("nan", "none", "<na>")
    ]
    return order_concentration_labels(unique)


def _format_target_concentration_label(group_label: str) -> str:
    """Format a binned label (e.g. ``1000 CFU``) as a display CFU value (``1000``)."""
    if group_label == "Overall":
        return "Overall"
    label = str(group_label).strip()
    if label.upper() == "0 CFU":
        return "0"
    if label.endswith("CFU"):
        numeric = label.replace("CFU", "").strip()
        if numeric.isdigit():
            return numeric
    return label


def _repeatability_display(cv_fraction: float, std_value: float) -> str:
    """Format repeatability as CV% when defined, otherwise SD."""
    if np.isfinite(cv_fraction):
        return f"{cv_fraction * 100:.1f}%"
    if np.isfinite(std_value):
        return f"SD {std_value:.3g}"
    return ""


def _group_concentration_cv(group_df: pd.DataFrame) -> float:
    """
    CV (fraction) of the actual concentration across a group's replicates.

    Uses the measured ``concentration`` (plate-count CFU/mL) to quantify
    sample-to-sample variability, reusing the shared
    :func:`~sensd_sers_analysis.assessment.consistency.coefficient_of_variation`.
    Returns NaN when no numeric concentration is available.
    """
    if "concentration" not in group_df.columns:
        return np.nan
    conc = extract_scalar_concentration(group_df["concentration"], group_df).dropna()
    if conc.empty:
        return np.nan
    return coefficient_of_variation(conc)


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build a float64 feature matrix with NaN filled to zero."""
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No feature columns found. Needed: {feature_cols}")
    X = df[available].fillna(0).to_numpy(dtype=np.float64, copy=False)
    return X, available


def _fit_global_classifier_predict_all(
    work: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    *,
    target_col: str = "target",
) -> np.ndarray:
    """
    Train one global classifier on ``train_idx`` rows and predict all rows.

    Parameters
    ----------
    work:
        Full labeled dataframe (reset index).
    feature_cols:
        Model input columns.
    train_idx:
        Positional indices of training rows (held-in sensors).
    target_col:
        Ground-truth class column.

    Returns
    -------
    np.ndarray
        Predicted class label per row in ``work``.
    """
    X, _ = _feature_matrix(work, feature_cols)
    y = work[target_col].map(str).to_numpy(dtype=object)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    clf = RandomForestClassifier(
        random_state=CLASSIFICATION_RANDOM_STATE,
        n_estimators=CLASSIFICATION_RF_N_ESTIMATORS,
    )
    clf.fit(X_train_s, y[train_idx])
    return clf.predict(scaler.transform(X))


def _fit_global_regressor_predict_all(
    regression_work: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    *,
    target_col: str = "log_concentration",
) -> np.ndarray:
    """
    Train one global log10 regressor on ``train_idx`` rows and predict all rows.

    Parameters
    ----------
    regression_work:
        Positive-CFU regression dataframe (reset index).
    feature_cols:
        Model input columns.
    train_idx:
        Positional indices of training rows (held-in sensors).
    target_col:
        Log10 concentration target column.

    Returns
    -------
    np.ndarray
        Predicted log10 concentration per row in ``regression_work``.
    """
    X, _ = _feature_matrix(regression_work, feature_cols)
    y = regression_work[target_col].to_numpy(dtype=np.float64, copy=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    reg = RandomForestRegressor(
        random_state=REGRESSION_RANDOM_STATE,
        n_estimators=CLASSIFICATION_RF_N_ESTIMATORS,
    )
    reg.fit(X_train_s, y[train_idx])
    return reg.predict(scaler.transform(X))


def _map_regression_predictions_to_classification_rows(
    classification_work: pd.DataFrame,
    regression_df: pd.DataFrame,
    pred_log_reg: np.ndarray,
) -> np.ndarray:
    """Align regressor outputs (regression row order) onto classification rows."""
    n_rows = len(classification_work)
    pred_log_all = np.full(n_rows, np.nan, dtype=float)
    orig_to_pos = {orig_idx: pos for pos, orig_idx in enumerate(classification_work.index)}
    for reg_pos, orig_idx in enumerate(regression_df.index):
        work_pos = orig_to_pos.get(orig_idx)
        if work_pos is not None:
            pred_log_all[work_pos] = float(pred_log_reg[reg_pos])
    return pred_log_all


def _fit_one_validation_fold(
    classification_work: pd.DataFrame,
    regression_df: pd.DataFrame,
    feature_cols: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray,
    *,
    sensor_col: str,
    target_col: str,
) -> ValidationFoldPredictions:
    """Train classifier/regressor on ``train_idx`` sensors; mark ``test_idx`` for eval."""
    n_rows = len(classification_work)
    eval_mask = np.zeros(n_rows, dtype=bool)
    eval_mask[test_idx] = True
    y_true = classification_work[target_col].map(str).to_numpy(dtype=object)

    try:
        y_pred = _fit_global_classifier_predict_all(
            classification_work,
            feature_cols,
            train_idx,
            target_col=target_col,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Validation classifier fold failed: %s", exc)
        y_pred = y_true.copy()

    pred_log_all: Optional[np.ndarray] = None
    if (
        not regression_df.empty
        and "log_concentration" in regression_df.columns
        and sensor_col in regression_df.columns
    ):
        reg_work = regression_df.reset_index(drop=True)
        reg_groups = reg_work[sensor_col].astype(str).to_numpy(dtype=object)
        train_sensors = set(groups[train_idx].astype(str))
        reg_train_idx = np.flatnonzero(np.isin(reg_groups, list(train_sensors)))
        if reg_train_idx.size >= 2:
            try:
                pred_log_reg = _fit_global_regressor_predict_all(
                    reg_work,
                    feature_cols,
                    reg_train_idx,
                )
                pred_log_all = _map_regression_predictions_to_classification_rows(
                    classification_work,
                    regression_df,
                    pred_log_reg,
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Validation regressor fold failed: %s", exc)

    return ValidationFoldPredictions(
        y_pred=y_pred,
        pred_log_conc=pred_log_all,
        eval_mask=eval_mask,
        n_train_sensors=int(np.unique(groups[train_idx]).size),
        n_test_sensors=int(np.unique(groups[test_idx]).size),
    )


def fit_validation_predictions(
    classification_work: pd.DataFrame,
    regression_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    test_size: float = REGRESSION_TEST_SIZE,
    n_splits: int = VALIDATION_N_SPLITS,
    random_state: int = REGRESSION_RANDOM_STATE,
    sensor_col: str = "sensor_id",
    target_col: str = "target",
) -> ValidationPredictions:
    """
    Fit classifier/regressor under repeated sensor-holdout splits.

    Each round reserves ``test_size`` of **sensors** for evaluation. ML metrics
    should be averaged across rounds on held-out sensors only.

    Parameters
    ----------
    classification_work:
        Classification dataframe with ``_target_group`` already attached.
    regression_df:
        Positive-CFU regression rows (may be empty).
    feature_cols:
        Shared ML feature columns.
    test_size:
        Fraction of sensors reserved for evaluation each round.
    n_splits:
        Number of independent sensor-holdout rounds.
    random_state:
        Base RNG seed.
    sensor_col:
        Sensor grouping column.
    target_col:
        Classification target column.

    Returns
    -------
    ValidationPredictions
        True labels plus one prediction bundle per successful fold.
    """
    n_rows = len(classification_work)
    y_true = classification_work[target_col].map(str).to_numpy(dtype=object)
    groups = classification_work[sensor_col].astype(str).to_numpy(dtype=object)
    unique_sensors = np.unique(groups)

    folds: list[ValidationFoldPredictions] = []
    if unique_sensors.size >= 2:
        try:
            split_pairs = iter_group_train_test_indices(
                groups,
                n_splits=n_splits,
                test_size=test_size,
                random_state=random_state,
            )
            for train_idx, test_idx in split_pairs:
                folds.append(
                    _fit_one_validation_fold(
                        classification_work,
                        regression_df,
                        feature_cols,
                        train_idx,
                        test_idx,
                        groups,
                        sensor_col=sensor_col,
                        target_col=target_col,
                    )
                )
        except ValueError as exc:
            logger.warning("Validation sensor holdout unavailable: %s", exc)
    else:
        logger.warning(
            "Validation sensor holdout skipped: need >=2 sensors, found %d.",
            unique_sensors.size,
        )

    if not folds:
        # Fallback: train on all rows; no honest eval mask (metrics stay blank).
        train_idx = np.arange(n_rows, dtype=np.intp)
        test_idx = np.array([], dtype=np.intp)
        folds.append(
            _fit_one_validation_fold(
                classification_work,
                regression_df,
                feature_cols,
                train_idx,
                test_idx,
                groups,
                sensor_col=sensor_col,
                target_col=target_col,
            )
        )
        return ValidationPredictions(
            y_true=y_true,
            folds=tuple(folds),
            sensor_holdout_available=False,
            n_splits=0,
        )

    return ValidationPredictions(
        y_true=y_true,
        folds=tuple(folds),
        sensor_holdout_available=True,
        n_splits=len(folds),
    )


def _quantification_range_string(pred_cfu: np.ndarray) -> str:
    """Format min/max predicted CFU as ``~lo-hi``."""
    valid = pred_cfu[np.isfinite(pred_cfu) & (pred_cfu > 0)]
    if valid.size == 0:
        return ""
    lo = int(round(float(np.min(valid))))
    hi = int(round(float(np.max(valid))))
    return f"~{lo}-{hi}"


def _meets_target(accuracy_fraction: float, threshold: float) -> str:
    """Return Pass/Fail from an accuracy fraction."""
    if not np.isfinite(accuracy_fraction):
        return "N/A"
    return "Pass" if accuracy_fraction >= threshold else "Fail"


def _format_pct(value: float) -> str:
    """Format a fraction as percent text, or blank if undefined."""
    if not np.isfinite(value):
        return ""
    return f"{value * 100:.1f}%"


def _format_optional_number(value: float, *, digits: int = 1) -> str:
    """
    Format an optional numeric value as text for Arrow-safe display tables.

    Mixing ``float`` with ``""`` in one column makes Streamlit/PyArrow fail
    (it infers float, then rejects the empty string). Always emit ``str``.
    """
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _group_rinsate_positive_masks(group_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return explicit rinsate-control and bacteria-sample membership masks."""
    rinsate, bacteria = sample_type_masks(group_df)
    return rinsate.to_numpy(), bacteria.to_numpy()


def _mean_fold_rate(
    fold_rates: list[float],
    *,
    total_eval_n: int,
    min_eval_rows: int = VALIDATION_MIN_EVAL_ROWS,
) -> float:
    """Mean of per-fold rates; NaN when total held-out N is below the guard."""
    if total_eval_n < min_eval_rows:
        return np.nan
    finite = [r for r in fold_rates if np.isfinite(r)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


def _ml_metrics_across_folds(
    group_idx: np.ndarray,
    *,
    serovar: str,
    y_true: np.ndarray,
    predictions: ValidationPredictions,
    is_rinsate: np.ndarray,
    is_positive: np.ndarray,
) -> tuple[float, float, float, str, int]:
    """
    Mean Accuracy / FP / FN across sensor-holdout folds; pooled quant range.

    Returns
    -------
    tuple
        ``(accuracy, fp_rate, fn_rate, quant_str, total_eval_n)``.
    """
    acc_folds: list[float] = []
    fp_folds: list[float] = []
    fn_folds: list[float] = []
    pooled_pred_cfu: list[float] = []
    total_eval_n = 0
    total_rinsate_n = 0
    total_positive_n = 0

    for fold in predictions.folds:
        if not predictions.sensor_holdout_available:
            break
        eval_local = fold.eval_mask[group_idx]
        n_eval = int(eval_local.sum())
        total_eval_n += n_eval
        if n_eval > 0:
            acc_folds.append(
                float(np.mean(y_true[group_idx][eval_local] == fold.y_pred[group_idx][eval_local]))
            )

        rinsate_eval = is_rinsate & eval_local
        positive_eval = is_positive & eval_local
        n_rin = int(rinsate_eval.sum())
        n_pos = int(positive_eval.sum())
        total_rinsate_n += n_rin
        total_positive_n += n_pos
        if n_rin > 0:
            fp_folds.append(float(np.mean(fold.y_pred[group_idx][rinsate_eval] != "Rinsate")))
        if n_pos > 0:
            fn_folds.append(float(np.mean(fold.y_pred[group_idx][positive_eval] != serovar)))
            if fold.pred_log_conc is not None:
                logs = fold.pred_log_conc[group_idx][positive_eval]
                cfu = np.power(10.0, logs)
                pooled_pred_cfu.extend(cfu[np.isfinite(cfu) & (cfu > 0)].tolist())

    accuracy = _mean_fold_rate(acc_folds, total_eval_n=total_eval_n)
    fp_rate = _mean_fold_rate(fp_folds, total_eval_n=total_rinsate_n)
    fn_rate = _mean_fold_rate(fn_folds, total_eval_n=total_positive_n)
    quant_str = ""
    if total_positive_n >= VALIDATION_MIN_EVAL_ROWS and pooled_pred_cfu:
        quant_str = _quantification_range_string(np.asarray(pooled_pred_cfu, dtype=float))
    return accuracy, fp_rate, fn_rate, quant_str, total_eval_n


def _repeatability_stats(
    group_df: pd.DataFrame,
    feature_col: str,
) -> tuple[int, float, float, float, float]:
    """Return n_sensors, replicates/sensor, total tests, signal CV, pooled std."""
    n_sensors = int(group_df["sensor_id"].nunique()) if "sensor_id" in group_df.columns else 0
    if "sensor_id" in group_df.columns:
        reps = group_df.groupby("sensor_id", dropna=False).size()
        replicates_per_sensor = float(reps.mean()) if len(reps) else np.nan
    else:
        replicates_per_sensor = np.nan
    total_tests = len(group_df)
    if feature_col in group_df.columns and "sensor_id" in group_df.columns:
        per_sensor_cv = (
            group_df.groupby("sensor_id", dropna=False)[feature_col]
            .apply(lambda s: coefficient_of_variation(s.dropna()))
            .replace([np.inf, -np.inf], np.nan)
        )
        mean_cv = float(per_sensor_cv.mean()) if per_sensor_cv.notna().any() else np.nan
        pooled_std = float(group_df[feature_col].dropna().std()) if total_tests > 1 else np.nan
    else:
        mean_cv = np.nan
        pooled_std = np.nan
    return n_sensors, replicates_per_sensor, total_tests, mean_cv, pooled_std


def _aggregate_classification_row(
    group_df: pd.DataFrame,
    group_idx: np.ndarray,
    *,
    serovar: str,
    concentration_label: str,
    y_true: np.ndarray,
    predictions: ValidationPredictions,
    feature_col: str,
) -> dict[str, object]:
    """One Metrics-docx Table 1 row (classification / repeatability)."""
    n_sensors, replicates_per_sensor, total_tests, mean_cv, pooled_std = _repeatability_stats(
        group_df, feature_col
    )
    conc_cv = _group_concentration_cv(group_df)
    is_rinsate, is_positive = _group_rinsate_positive_masks(group_df)
    accuracy, _, _, _, n_eval = _ml_metrics_across_folds(
        group_idx,
        serovar=serovar,
        y_true=y_true,
        predictions=predictions,
        is_rinsate=is_rinsate,
        is_positive=is_positive,
    )
    return {
        "Concentration (CFU/mL)": concentration_label,
        "Serovar / Sample Group": serovar,
        "No. of Sensors Tested": n_sensors,
        "Replicates per Sensor": _format_optional_number(replicates_per_sensor),
        "Total Tests": total_tests,
        "Identification": serovar,
        "Repeatability CV% or SD": _repeatability_display(mean_cv, pooled_std),
        "Concentration CV%": (f"{conc_cv * 100:.1f}%" if np.isfinite(conc_cv) else ""),
        "Accuracy (% Correct)": _format_pct(accuracy),
        "Eval N": n_eval,
    }


def _aggregate_quantification_row(
    group_df: pd.DataFrame,
    group_idx: np.ndarray,
    *,
    serovar: str,
    concentration_label: str,
    y_true: np.ndarray,
    predictions: ValidationPredictions,
    accuracy_threshold: float,
) -> dict[str, object]:
    """One Metrics-docx Table 2 row (quantification / FP / FN)."""
    is_rinsate, is_positive = _group_rinsate_positive_masks(group_df)
    accuracy, fp_rate, fn_rate, quant_str, n_eval = _ml_metrics_across_folds(
        group_idx,
        serovar=serovar,
        y_true=y_true,
        predictions=predictions,
        is_rinsate=is_rinsate,
        is_positive=is_positive,
    )
    return {
        "Concentration (CFU/mL)": concentration_label,
        "Serovar / Sample Group": serovar,
        "Quantification Accuracy": quant_str,
        "False Positive Rate": _format_pct(fp_rate),
        "False Negative Rate": _format_pct(fn_rate),
        "Meet Target?": _meets_target(accuracy, accuracy_threshold),
        "Eval N": n_eval,
    }


def _iter_serovar_target_groups(
    work: pd.DataFrame,
) -> list[tuple[str, str, pd.DataFrame, np.ndarray]]:
    """
    Yield ``(serovar, concentration_label, group_df, group_idx)`` plus Overall rows.

    Overall rows use ``concentration_label="Overall"``.
    """
    items: list[tuple[str, str, pd.DataFrame, np.ndarray]] = []
    serovars = sorted(work["serotype"].dropna().astype(str).unique().tolist())
    for serovar in serovars:
        sero_mask = work["serotype"].astype(str) == serovar
        sero_df = work.loc[sero_mask]
        if sero_df.empty:
            continue
        sero_idx = np.flatnonzero(sero_mask.to_numpy())
        for target_group in _sorted_target_groups(sero_df[_TARGET_GROUP_COL]):
            group_mask = sero_mask & (work[_TARGET_GROUP_COL] == target_group)
            group_df = work.loc[group_mask]
            if group_df.empty:
                continue
            items.append(
                (
                    serovar,
                    _format_target_concentration_label(target_group),
                    group_df,
                    np.flatnonzero(group_mask.to_numpy()),
                )
            )
        items.append((serovar, "Overall", sero_df, sero_idx))
    return items


def build_concentration_repeatability_table(
    work: pd.DataFrame,
    predictions: ValidationPredictions,
    *,
    repeatability_feature: str = CLASSIFICATION_INLIER_FEATURE,
) -> pd.DataFrame:
    """
    Build Metrics-docx Table 1: concentration and repeatability testing.

    Parameters
    ----------
    work:
        Classification dataframe with ``_target_group`` attached (reset index).
    predictions:
        Repeated sensor-holdout outputs from :func:`fit_validation_predictions`.
    repeatability_feature:
        Feature used to compute repeatability CV/SD.

    Returns
    -------
    pd.DataFrame
        Per-concentration rows and an Overall row per serovar.
    """
    if work.empty:
        return pd.DataFrame(columns=TABLE1_COLUMNS)

    required = {"serotype", "sensor_id", "target", _TARGET_GROUP_COL}
    if not required.issubset(work.columns):
        logger.warning(
            "build_concentration_repeatability_table: missing columns %s",
            required - set(work.columns),
        )
        return pd.DataFrame(columns=TABLE1_COLUMNS)

    rows = [
        _aggregate_classification_row(
            group_df,
            group_idx,
            serovar=serovar,
            concentration_label=label,
            y_true=predictions.y_true,
            predictions=predictions,
            feature_col=repeatability_feature,
        )
        for serovar, label, group_df, group_idx in _iter_serovar_target_groups(work)
    ]
    if not rows:
        return pd.DataFrame(columns=TABLE1_COLUMNS)
    return pd.DataFrame(rows, columns=TABLE1_COLUMNS)


def build_quantification_table(
    work: pd.DataFrame,
    predictions: ValidationPredictions,
    *,
    accuracy_threshold: float = VALIDATION_ACCURACY_MIN_THRESHOLD,
) -> pd.DataFrame:
    """
    Build Metrics-docx Table 2: quantification accuracy, FP/FN, Meet target?

    Parameters
    ----------
    work:
        Classification dataframe with ``_target_group`` attached (reset index).
    predictions:
        Repeated sensor-holdout outputs.
    accuracy_threshold:
        Minimum mean identification accuracy for ``Meet Target?`` = Pass.

    Returns
    -------
    pd.DataFrame
        One row per serovar × target concentration (plus Overall).
    """
    if work.empty:
        return pd.DataFrame(columns=TABLE2_COLUMNS)

    required = {"serotype", "sensor_id", "target", _TARGET_GROUP_COL}
    if not required.issubset(work.columns):
        logger.warning(
            "build_quantification_table: missing columns %s",
            required - set(work.columns),
        )
        return pd.DataFrame(columns=TABLE2_COLUMNS)

    rows = [
        _aggregate_quantification_row(
            group_df,
            group_idx,
            serovar=serovar,
            concentration_label=label,
            y_true=predictions.y_true,
            predictions=predictions,
            accuracy_threshold=accuracy_threshold,
        )
        for serovar, label, group_df, group_idx in _iter_serovar_target_groups(work)
    ]
    if not rows:
        return pd.DataFrame(columns=TABLE2_COLUMNS)
    return pd.DataFrame(rows, columns=TABLE2_COLUMNS)


def _mean_signal_change_pct(
    df: pd.DataFrame,
    feature_col: str,
) -> float:
    """
    Mean percent signal change (first → last test) across sensors.

    Uses one aggregated feature value per (sensor_id, test_id).
    """
    if df.empty or "sensor_id" not in df.columns:
        return np.nan
    try:
        deg = prepare_degradation_data(df, feature_col)
    except ValueError:
        return np.nan
    if deg.empty:
        return np.nan

    changes: list[float] = []
    for _, sensor_grp in deg.groupby("sensor_id", dropna=False):
        sensor_grp = sensor_grp.sort_values("test_ordinal")
        if len(sensor_grp) < 2:
            continue
        first_val = float(sensor_grp[feature_col].iloc[0])
        last_val = float(sensor_grp[feature_col].iloc[-1])
        if first_val == 0 or not np.isfinite(first_val):
            continue
        changes.append((last_val - first_val) / abs(first_val) * 100.0)
    if not changes:
        return np.nan
    return float(np.mean(changes))


def build_consistency_reusability_table(
    work: pd.DataFrame,
    predictions: ValidationPredictions,
    *,
    repeatability_feature: str = CLASSIFICATION_INLIER_FEATURE,
    qa_feature: str = CLASSIFICATION_INLIER_FEATURE,
) -> pd.DataFrame:
    """
    Build Metrics-docx Table 3: consistency testing and sensor reusability.

    Analyzes sensors with repeated ``test_id`` values at each serovar × concentration.

    Parameters
    ----------
    work:
        Classification dataframe with ``_target_group`` attached (reset index).
    predictions:
        Repeated sensor-holdout outputs from :func:`fit_validation_predictions`.
    repeatability_feature:
        Feature for CV and signal-change metrics.
    qa_feature:
        Feature used to identify failed (Excluded) sensors from global QA.

    Returns
    -------
    pd.DataFrame
        One row per serovar × concentration with repeated-use statistics.
    """
    if work.empty:
        return pd.DataFrame(columns=TABLE3_COLUMNS)

    required = {"serotype", "sensor_id", "test_id", "target", _TARGET_GROUP_COL}
    if not required.issubset(work.columns):
        logger.warning(
            "build_consistency_reusability_table: missing columns %s",
            required - set(work.columns),
        )
        return pd.DataFrame(columns=TABLE3_COLUMNS)

    _, excluded_map = get_global_model_consistency_qa(
        work,
        feature_cols=[qa_feature] if qa_feature in work.columns else [],
    )

    rows: list[dict[str, object]] = []
    serovars = sorted(work["serotype"].dropna().astype(str).unique().tolist())

    for serovar in serovars:
        sero_df = work[work["serotype"].astype(str) == serovar]
        target_groups = _sorted_target_groups(sero_df[_TARGET_GROUP_COL])
        excluded_sensors = excluded_map.get((serovar, qa_feature), set())

        for target_group in target_groups:
            group_df = sero_df.loc[sero_df[_TARGET_GROUP_COL] == target_group].copy()
            if group_df.empty:
                continue

            tests_per_sensor = group_df.groupby("sensor_id", dropna=False)["test_id"].nunique()
            reusable = tests_per_sensor[tests_per_sensor >= 2]
            n_reusable = int(len(reusable))
            if n_reusable == 0:
                continue

            reusable_ids = set(reusable.index.astype(str))
            reuse_mask = (
                (work["serotype"].astype(str) == serovar)
                & (work[_TARGET_GROUP_COL] == target_group)
                & work["sensor_id"].astype(str).isin(reusable_ids)
            )
            reuse_df = work.loc[reuse_mask]
            mean_tests = float(reusable.mean())
            total_tests = int(reusable.sum())
            signal_change = _mean_signal_change_pct(reuse_df, repeatability_feature)
            cv_val = (
                coefficient_of_variation(reuse_df[repeatability_feature].dropna())
                if repeatability_feature in reuse_df.columns
                else np.nan
            )

            failed_in_group = [
                s for s in reusable_ids if str(s) in {str(x) for x in excluded_sensors}
            ]
            n_failed = len(failed_in_group)
            avg_uses_before_failure = np.nan
            if n_failed > 0:
                uses = [int(tests_per_sensor.loc[s]) for s in failed_in_group]
                avg_uses_before_failure = float(np.mean(uses))

            group_idx = np.flatnonzero(reuse_mask.to_numpy())
            is_rinsate, is_positive = _group_rinsate_positive_masks(reuse_df)
            accuracy, _, _, _, n_eval = _ml_metrics_across_folds(
                group_idx,
                serovar=serovar,
                y_true=predictions.y_true,
                predictions=predictions,
                is_rinsate=is_rinsate,
                is_positive=is_positive,
            )

            rows.append(
                {
                    "Serovar": serovar,
                    "Concentration (CFU/ml)": _format_target_concentration_label(target_group),
                    "No. of Tested sensors": n_reusable,
                    "Repeated Tests per Sensor (n)": _format_optional_number(mean_tests),
                    "Total Tests": total_tests,
                    "Mean Signal Change First to Last Test (%)": (
                        f"{signal_change:.1f}%" if np.isfinite(signal_change) else ""
                    ),
                    "Repeatability CV%": (_format_pct(cv_val) if np.isfinite(cv_val) else ""),
                    "No. of Failed Sensors": n_failed,
                    "Average Uses Before Failure": _format_optional_number(avg_uses_before_failure),
                    "Accuracy Across Repeated Uses": _format_pct(accuracy),
                    "Eval N": n_eval,
                    "Reliability Notes": "",
                }
            )

    if not rows:
        return pd.DataFrame(columns=TABLE3_COLUMNS)
    return pd.DataFrame(rows, columns=TABLE3_COLUMNS)


def build_validation_tables(
    classification_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    repeatability_feature: str = CLASSIFICATION_INLIER_FEATURE,
    accuracy_threshold: float = VALIDATION_ACCURACY_MIN_THRESHOLD,
) -> ValidationTableArtifacts:
    """
    Build all three Metrics-docx validation tables.

    Parameters
    ----------
    classification_df:
        Rows prepared by :func:`~sensd_sers_analysis.classification.prepare_classification_dataset`.
    regression_df:
        Positive-CFU regression rows from
        :func:`~sensd_sers_analysis.regression.prepare_concentration_regression_data`.
    feature_cols:
        ML feature column names.
    repeatability_feature:
        Scalar feature for CV / signal-change metrics.
    accuracy_threshold:
        Pass/Fail accuracy cutoff for the quantification table.

    Returns
    -------
    ValidationTableArtifacts
        Tables 1–3, row counts, and repeated-holdout prediction artifacts.
    """
    if classification_df.empty:
        return ValidationTableArtifacts(
            concentration_repeatability=pd.DataFrame(columns=TABLE1_COLUMNS),
            quantification=pd.DataFrame(columns=TABLE2_COLUMNS),
            consistency_reusability=pd.DataFrame(columns=TABLE3_COLUMNS),
            n_classification_rows=0,
            n_regression_rows=len(regression_df),
            predictions=None,
        )

    work = _add_target_concentration_group(classification_df.copy()).reset_index(drop=True)
    predictions = fit_validation_predictions(work, regression_df, feature_cols)

    table1 = build_concentration_repeatability_table(
        work,
        predictions,
        repeatability_feature=repeatability_feature,
    )
    table2 = build_quantification_table(
        work,
        predictions,
        accuracy_threshold=accuracy_threshold,
    )
    table3 = build_consistency_reusability_table(
        work,
        predictions,
        repeatability_feature=repeatability_feature,
    )
    return ValidationTableArtifacts(
        concentration_repeatability=table1,
        quantification=table2,
        consistency_reusability=table3,
        n_classification_rows=len(classification_df),
        n_regression_rows=len(regression_df),
        predictions=predictions,
    )

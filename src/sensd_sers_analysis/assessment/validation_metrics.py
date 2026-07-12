"""
Validation metric tables for SENS-D concentration and sensor-reusability reporting.

Builds two publication-ready summary tables:

1. **Concentration & repeatability** — per serovar × concentration with an Overall row.
2. **Consistency & reusability** — repeated sensor use per serovar × concentration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

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
)
from sensd_sers_analysis.processing import extract_scalar_concentration
from sensd_sers_analysis.regression.splits import group_train_test_indices
from sensd_sers_analysis.utils import order_concentration_labels

logger = logging.getLogger(__name__)

_TARGET_GROUP_COL = "_target_group"

TABLE1_COLUMNS = [
    "Concentration (CFU/mL)",
    "Serovar / Sample Group",
    "No. of Sensors Tested",
    "Replicates per Sensor",
    "Total Tests",
    "Identification",
    "Repeatability CV% or SD",
    "Accuracy (% Correct)",
    "Quantification Accuracy",
    "False Positive Rate",
    "False Negative Rate",
    "Meets Target?",
]

TABLE2_COLUMNS = [
    "Serovar",
    "Concentration (CFU/ml)",
    "No. of Tested sensors",
    "Repeated Tests per Sensor (n)",
    "Total Tests (Sensors x n)",
    "Mean Signal Change First to Last Test (%)",
    "Repeatability CV%",
    "No. of Failed Sensors",
    "Average Uses Before Failure",
    "Accuracy Across Repeated Uses",
    "Reliability Notes",
]


@dataclass
class ValidationPredictions:
    """
    Globally trained model outputs for validation tables.

    Classifier and regressor are each fit once on training-sensor rows, then
    applied to the full dataset. ML metrics (accuracy, FP/FN, quantification)
    are computed only on held-out test-sensor rows (``eval_mask``).
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    pred_log_conc: Optional[np.ndarray]
    eval_mask: np.ndarray
    sensor_holdout_available: bool
    n_train_sensors: int
    n_test_sensors: int
    n_eval_rows: int


@dataclass
class ValidationTableArtifacts:
    """Container for both validation summary tables."""

    concentration_repeatability: pd.DataFrame
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
    Add ``_target_group`` for table grouping using binned target concentrations.

    Prefers ``concentration_group`` (0, 1, 10, 100, 1000 CFU bins from
    ``preprocess_metadata``). Falls back to nearest-log10 binning of raw
    concentration when the column is missing.
    """
    out = df.copy()
    if "concentration_group" in out.columns:
        labels = out["concentration_group"].astype(str).str.strip()
        invalid = labels.isin(("", "nan", "None", "NaN", "<NA>"))
        out[_TARGET_GROUP_COL] = labels.mask(invalid, "Unknown")
        return out

    conc = _scalar_concentration_series(out)
    groups = pd.Series("Unknown", index=out.index, dtype=object)
    valid = conc.notna()
    zero_mask = valid & (conc <= 0)
    groups.loc[zero_mask] = "0 CFU"
    pos_mask = valid & (conc > 0)
    if pos_mask.any():
        centers = np.array([1.0, 10.0, 100.0, 1000.0])
        center_labels = ["1 CFU", "10 CFU", "100 CFU", "1000 CFU"]
        log_conc = np.log10(conc[pos_mask].astype(float).values)
        dists = np.abs(log_conc[:, np.newaxis] - np.log10(centers))
        nearest = np.argmin(dists, axis=1)
        groups.loc[pos_mask] = [center_labels[i] for i in nearest]
    out[_TARGET_GROUP_COL] = groups
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


def fit_validation_predictions(
    classification_work: pd.DataFrame,
    regression_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    test_size: float = REGRESSION_TEST_SIZE,
    random_state: int = REGRESSION_RANDOM_STATE,
    sensor_col: str = "sensor_id",
    target_col: str = "target",
) -> ValidationPredictions:
    """
    Fit global classifier and regressor once, predict the full dataset.

    Uses a sensor-level holdout split (same policy as concentration regression).
    Rows from held-out test sensors are marked in ``eval_mask`` for honest
    per-group ML metrics; repeatability stats still use all rows.

    Parameters
    ----------
    classification_work:
        Classification dataframe with ``_target_group`` already attached.
    regression_df:
        Positive-CFU regression rows (may be empty).
    feature_cols:
        Shared ML feature columns.
    test_size:
        Fraction of sensors reserved for evaluation.
    random_state:
        RNG seed for the group split.
    sensor_col:
        Sensor grouping column.
    target_col:
        Classification target column.

    Returns
    -------
    ValidationPredictions
        Global predictions and the evaluation mask.
    """
    n_rows = len(classification_work)
    y_true = classification_work[target_col].map(str).to_numpy(dtype=object)
    eval_mask = np.zeros(n_rows, dtype=bool)
    sensor_holdout_available = False
    n_train_sensors = 0
    n_test_sensors = 0

    groups = classification_work[sensor_col].astype(str).to_numpy(dtype=object)
    unique_sensors = np.unique(groups)

    if unique_sensors.size >= 2:
        try:
            train_idx, test_idx = group_train_test_indices(
                groups,
                test_size=test_size,
                random_state=random_state,
            )
            eval_mask[test_idx] = True
            sensor_holdout_available = True
            n_train_sensors = int(np.unique(groups[train_idx]).size)
            n_test_sensors = int(np.unique(groups[test_idx]).size)
        except ValueError as exc:
            logger.warning("Validation sensor holdout unavailable: %s", exc)
            train_idx = np.arange(n_rows, dtype=np.intp)
    else:
        logger.warning(
            "Validation sensor holdout skipped: need >=2 sensors, found %d.",
            unique_sensors.size,
        )
        train_idx = np.arange(n_rows, dtype=np.intp)

    try:
        y_pred = _fit_global_classifier_predict_all(
            classification_work,
            feature_cols,
            train_idx,
            target_col=target_col,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("Global validation classifier fit failed: %s", exc)
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
                pred_log_all = np.full(n_rows, np.nan, dtype=float)
                orig_to_pos = {
                    orig_idx: pos for pos, orig_idx in enumerate(classification_work.index)
                }
                for reg_pos, orig_idx in enumerate(regression_df.index):
                    work_pos = orig_to_pos.get(orig_idx)
                    if work_pos is not None:
                        pred_log_all[work_pos] = float(pred_log_reg[reg_pos])
            except (ValueError, TypeError) as exc:
                logger.warning("Global validation regressor fit failed: %s", exc)
        else:
            logger.warning(
                "Global validation regressor skipped: need >=2 training rows, found %d.",
                reg_train_idx.size,
            )

    return ValidationPredictions(
        y_true=y_true,
        y_pred=y_pred,
        pred_log_conc=pred_log_all,
        eval_mask=eval_mask,
        sensor_holdout_available=sensor_holdout_available,
        n_train_sensors=n_train_sensors,
        n_test_sensors=n_test_sensors,
        n_eval_rows=int(eval_mask.sum()),
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


def _aggregate_table1_row(
    group_df: pd.DataFrame,
    *,
    serovar: str,
    concentration_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_log_conc: Optional[np.ndarray],
    eval_mask: np.ndarray,
    feature_col: str,
    accuracy_threshold: float,
) -> dict[str, object]:
    """Compute one Table 1 row from rows sharing serovar × concentration."""
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

    # ML metrics: held-out test-sensor rows only (honest generalization).
    ml_mask = eval_mask
    if ml_mask.any():
        accuracy = float(np.mean(y_true[ml_mask] == y_pred[ml_mask]))
    else:
        accuracy = np.nan

    if "concentration_group" in group_df.columns:
        cg = group_df["concentration_group"].astype(str).str.strip()
        is_rinsate = cg == "0 CFU"
        is_positive = cg.str.endswith("CFU") & (cg != "0 CFU") & (cg != "Unknown")
    elif _TARGET_GROUP_COL in group_df.columns:
        tg = group_df[_TARGET_GROUP_COL].astype(str).str.strip()
        is_rinsate = tg == "0 CFU"
        is_positive = tg.str.endswith("CFU") & (tg != "0 CFU") & (tg != "Unknown")
    else:
        conc_scalar = _scalar_concentration_series(group_df)
        is_rinsate = conc_scalar.notna() & (conc_scalar <= 0)
        is_positive = conc_scalar.notna() & (conc_scalar > 0)

    rinsate_eval = is_rinsate.to_numpy() & ml_mask
    positive_eval = is_positive.to_numpy() & ml_mask
    fp_denom = int(rinsate_eval.sum())
    fn_denom = int(positive_eval.sum())
    if fp_denom > 0:
        fp_rate = float(np.mean(y_pred[rinsate_eval] != "Rinsate"))
    else:
        fp_rate = np.nan
    if fn_denom > 0:
        fn_rate = float(np.mean(y_pred[positive_eval] != serovar))
    else:
        fn_rate = np.nan

    quant_str = ""
    if pred_log_conc is not None and positive_eval.any():
        pred_cfu = np.power(10.0, pred_log_conc[positive_eval])
        quant_str = _quantification_range_string(pred_cfu)

    return {
        "Concentration (CFU/mL)": concentration_label,
        "Serovar / Sample Group": serovar,
        "No. of Sensors Tested": n_sensors,
        "Replicates per Sensor": (
            round(replicates_per_sensor, 1) if np.isfinite(replicates_per_sensor) else ""
        ),
        "Total Tests": total_tests,
        "Identification": serovar,
        "Repeatability CV% or SD": _repeatability_display(mean_cv, pooled_std),
        "Accuracy (% Correct)": (f"{accuracy * 100:.1f}%" if np.isfinite(accuracy) else ""),
        "Quantification Accuracy": quant_str,
        "False Positive Rate": (f"{fp_rate * 100:.1f}%" if np.isfinite(fp_rate) else ""),
        "False Negative Rate": (f"{fn_rate * 100:.1f}%" if np.isfinite(fn_rate) else ""),
        "Meets Target?": _meets_target(accuracy, accuracy_threshold),
    }


def build_concentration_repeatability_table(
    work: pd.DataFrame,
    predictions: ValidationPredictions,
    *,
    repeatability_feature: str = CLASSIFICATION_INLIER_FEATURE,
    accuracy_threshold: float = VALIDATION_ACCURACY_MIN_THRESHOLD,
) -> pd.DataFrame:
    """
    Build Table 1: concentration and repeatability testing.

    Parameters
    ----------
    work:
        Classification dataframe with ``_target_group`` attached (reset index).
    predictions:
        Global model outputs from :func:`fit_validation_predictions`.
    repeatability_feature:
        Feature used to compute repeatability CV/SD.
    accuracy_threshold:
        Minimum accuracy fraction for Pass/Fail (default 0.80).

    Returns
    -------
    pd.DataFrame
        Table with per-concentration rows and an Overall row per serovar.
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

    y_true = predictions.y_true
    y_pred = predictions.y_pred
    pred_log_all = predictions.pred_log_conc
    eval_mask = predictions.eval_mask

    rows: list[dict[str, object]] = []
    serovars = sorted(work["serotype"].dropna().astype(str).unique().tolist())

    for serovar in serovars:
        sero_mask = work["serotype"].astype(str) == serovar
        sero_df = work.loc[sero_mask]
        if sero_df.empty:
            continue

        target_groups = _sorted_target_groups(sero_df[_TARGET_GROUP_COL])

        sero_idx = np.flatnonzero(sero_mask.to_numpy())
        sero_eval = eval_mask[sero_idx]
        sero_pred_log = pred_log_all[sero_idx] if pred_log_all is not None else None

        for target_group in target_groups:
            group_mask = sero_mask & (work[_TARGET_GROUP_COL] == target_group)
            group_df = work.loc[group_mask]
            if group_df.empty:
                continue
            group_idx = np.flatnonzero(group_mask.to_numpy())
            rows.append(
                _aggregate_table1_row(
                    group_df,
                    serovar=serovar,
                    concentration_label=_format_target_concentration_label(target_group),
                    y_true=y_true[group_idx],
                    y_pred=y_pred[group_idx],
                    pred_log_conc=(pred_log_all[group_idx] if pred_log_all is not None else None),
                    eval_mask=eval_mask[group_idx],
                    feature_col=repeatability_feature,
                    accuracy_threshold=accuracy_threshold,
                )
            )

        rows.append(
            _aggregate_table1_row(
                sero_df,
                serovar=serovar,
                concentration_label="Overall",
                y_true=y_true[sero_idx],
                y_pred=y_pred[sero_idx],
                pred_log_conc=sero_pred_log,
                eval_mask=sero_eval,
                feature_col=repeatability_feature,
                accuracy_threshold=accuracy_threshold,
            )
        )

    if not rows:
        return pd.DataFrame(columns=TABLE1_COLUMNS)
    return pd.DataFrame(rows, columns=TABLE1_COLUMNS)


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
    Build Table 2: consistency testing and sensor reusability.

    Analyzes sensors with repeated ``test_id`` values at each serovar × concentration.

    Parameters
    ----------
    work:
        Classification dataframe with ``_target_group`` attached (reset index).
    predictions:
        Global model outputs from :func:`fit_validation_predictions`.
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
        return pd.DataFrame(columns=TABLE2_COLUMNS)

    required = {"serotype", "sensor_id", "test_id", "target", _TARGET_GROUP_COL}
    if not required.issubset(work.columns):
        logger.warning(
            "build_consistency_reusability_table: missing columns %s",
            required - set(work.columns),
        )
        return pd.DataFrame(columns=TABLE2_COLUMNS)

    _, excluded_map = get_global_model_consistency_qa(
        work,
        feature_cols=[qa_feature] if qa_feature in work.columns else [],
    )

    y_true = predictions.y_true
    y_pred = predictions.y_pred
    eval_mask = predictions.eval_mask

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

            if repeatability_feature in reuse_df.columns:
                cv_val = coefficient_of_variation(reuse_df[repeatability_feature].dropna())
            else:
                cv_val = np.nan

            failed_in_group = [
                s for s in reusable_ids if str(s) in {str(x) for x in excluded_sensors}
            ]
            n_failed = len(failed_in_group)

            avg_uses_before_failure = np.nan
            if n_failed > 0:
                uses = [int(tests_per_sensor.loc[s]) for s in failed_in_group]
                avg_uses_before_failure = float(np.mean(uses))

            group_idx = np.flatnonzero(reuse_mask.to_numpy())
            ml_idx = group_idx[eval_mask[group_idx]]
            acc = float(np.mean(y_true[ml_idx] == y_pred[ml_idx])) if ml_idx.size else np.nan

            rows.append(
                {
                    "Serovar": serovar,
                    "Concentration (CFU/ml)": _format_target_concentration_label(target_group),
                    "No. of Tested sensors": n_reusable,
                    "Repeated Tests per Sensor (n)": round(mean_tests, 1),
                    "Total Tests (Sensors x n)": total_tests,
                    "Mean Signal Change First to Last Test (%)": (
                        f"{signal_change:.1f}%" if np.isfinite(signal_change) else ""
                    ),
                    "Repeatability CV%": (f"{cv_val * 100:.1f}%" if np.isfinite(cv_val) else ""),
                    "No. of Failed Sensors": n_failed,
                    "Average Uses Before Failure": (
                        round(avg_uses_before_failure, 1)
                        if np.isfinite(avg_uses_before_failure)
                        else ""
                    ),
                    "Accuracy Across Repeated Uses": (
                        f"{acc * 100:.1f}%" if np.isfinite(acc) else ""
                    ),
                    "Reliability Notes": "",
                }
            )

    if not rows:
        return pd.DataFrame(columns=TABLE2_COLUMNS)
    return pd.DataFrame(rows, columns=TABLE2_COLUMNS)


def build_validation_tables(
    classification_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    repeatability_feature: str = CLASSIFICATION_INLIER_FEATURE,
    accuracy_threshold: float = VALIDATION_ACCURACY_MIN_THRESHOLD,
) -> ValidationTableArtifacts:
    """
    Build both validation tables from clean classification and regression data.

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
        Pass/Fail accuracy cutoff.

    Returns
    -------
    ValidationTableArtifacts
        Both summary tables, row counts, and global prediction artifacts.
    """
    if classification_df.empty:
        return ValidationTableArtifacts(
            concentration_repeatability=pd.DataFrame(columns=TABLE1_COLUMNS),
            consistency_reusability=pd.DataFrame(columns=TABLE2_COLUMNS),
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
        accuracy_threshold=accuracy_threshold,
    )
    table2 = build_consistency_reusability_table(
        work,
        predictions,
        repeatability_feature=repeatability_feature,
    )
    return ValidationTableArtifacts(
        concentration_repeatability=table1,
        consistency_reusability=table2,
        n_classification_rows=len(classification_df),
        n_regression_rows=len(regression_df),
        predictions=predictions,
    )

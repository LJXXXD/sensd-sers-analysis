"""Pre-QA inventory of measured spectra and their embedded metadata."""

from dataclasses import dataclass

import pandas as pd

from sensd_sers_analysis.config.inventory import (
    COVERAGE_COLUMNS,
    PROVENANCE_COLUMNS,
    SESSION_COLUMNS,
    SPECTRUM_ID_COLUMNS,
)


@dataclass(slots=True)
class DataInventory:
    """Inventory tables; sessions are metadata combinations, not biological replicates."""

    spectra: pd.DataFrame
    summary: pd.DataFrame
    coverage: pd.DataFrame
    sensors: pd.DataFrame
    provenance: pd.DataFrame
    shared_ids: pd.DataFrame
    missing_metadata: pd.DataFrame
    duplicate_id_rows: int


def spectrum_metadata(data: pd.DataFrame, *, tidy: bool = False) -> pd.DataFrame:
    """
    Extract one metadata record per spectrum without counting Raman-shift rows.

    Parameters
    ----------
    data : pandas.DataFrame
        Loaded wide data or filtered tidy data.
    tidy : bool
        Collapse repeated metadata across Raman shifts when True. Wide rows
        remain intact, including repeated uploads, so ambiguity is visible.
    """
    columns = [
        column
        for column in data.columns
        if not column.startswith("rs_") and column not in ("raman_shift", "intensity")
    ]
    spectra = data.loc[:, columns].copy()
    if tidy:
        spectra = spectra.drop_duplicates()
    return spectra.reset_index(drop=True)


def _count_table(spectra: pd.DataFrame, dimensions: tuple[str, ...]) -> pd.DataFrame:
    """Count spectra, file basenames and composite sessions within observed groups."""
    dimensions = [column for column in dimensions if column in spectra.columns]
    if not dimensions or spectra.empty:
        return pd.DataFrame()
    aggregations = {"spectra": ("_session", "size"), "sessions": ("_session", "nunique")}
    if "filename" in spectra:
        aggregations["file_basenames"] = ("filename", "nunique")
    if "sensor_id" in spectra:
        aggregations["sensor_ids"] = ("sensor_id", "nunique")
    aggregations["first_date"] = ("_date", "min")
    aggregations["last_date"] = ("_date", "max")
    return (
        spectra.groupby(dimensions, dropna=False, observed=True).agg(**aggregations).reset_index()
    )


def build_data_inventory(spectra: pd.DataFrame) -> DataInventory:
    """
    Summarize spectrum records before any QA or model exclusion.

    Parameters
    ----------
    spectra : pandas.DataFrame
        One row per measured spectrum. No physical-device identity or independent
        biological-replicate identity is inferred from sensor IDs or test IDs.

    Returns
    -------
    DataInventory
        Observed coverage counts, date range, provenance and missingness tables.
        Sessions use the available operator, sensor, date, test, connection and
        serotype fields; a bare T1 is never treated as globally unique.
    """
    records = spectra.copy()
    for column in records.select_dtypes(include=["object", "string"]).columns:
        records[column] = records[column].replace(r"^\s*$", pd.NA, regex=True)
    session_columns = [column for column in SESSION_COLUMNS if column in records]
    if session_columns and not records.empty:
        records["_session"] = records.groupby(session_columns, dropna=False, observed=True).ngroup()
    else:
        records["_session"] = pd.NA
    dates = records.get("date", pd.Series(index=records.index, dtype="object"))
    records["_date"] = pd.to_datetime(dates, errors="coerce", format="mixed")
    identity = [column for column in SPECTRUM_ID_COLUMNS if column in records]
    duplicate_id_rows = (
        int(records.duplicated(identity, keep=False).sum())
        if len(identity) == len(SPECTRUM_ID_COLUMNS)
        else 0
    )
    summary = pd.DataFrame(
        [
            {
                "spectra": len(records),
                "sensor_ids": records["sensor_id"].nunique() if "sensor_id" in records else 0,
                "file_basenames": records["filename"].nunique() if "filename" in records else 0,
                "sessions": records["_session"].nunique(),
                "first_date": records["_date"].min(),
                "last_date": records["_date"].max(),
                "missing_or_invalid_dates": int(records["_date"].isna().sum()),
            }
        ]
    )
    provenance_columns = tuple(column for column in PROVENANCE_COLUMNS if column in records)
    shared = pd.DataFrame()
    if "sensor_id" in records and provenance_columns:
        shared = records.groupby("sensor_id", dropna=True)[list(provenance_columns)].nunique()
        shared = shared.loc[shared.gt(1).any(axis=1)].reset_index()
        shared = shared.rename(
            columns={column: f"distinct_{column}" for column in provenance_columns}
        )
    metadata_columns = tuple(dict.fromkeys((*SESSION_COLUMNS, *COVERAGE_COLUMNS, "filename")))
    missing = pd.DataFrame(
        [
            {
                "field": column,
                "missing_spectra": int(records[column].isna().sum())
                if column in records
                else len(records),
            }
            for column in metadata_columns
        ]
    )
    return DataInventory(
        spectra=spectra,
        summary=summary,
        coverage=_count_table(records, COVERAGE_COLUMNS),
        sensors=_count_table(records, ("sensor_id", "serotype")),
        provenance=_count_table(records, provenance_columns),
        shared_ids=shared,
        missing_metadata=missing,
        duplicate_id_rows=duplicate_id_rows,
    )


def build_inventory_chart_tables(
    spectra: pd.DataFrame, *, group_by: str = "operator"
) -> dict[str, pd.DataFrame]:
    """
    Aggregate measured spectra for composition, coverage and acquisition charts.

    Parameters
    ----------
    spectra : pandas.DataFrame
        One row per spectrum in the selected inventory scope.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Composition and coverage preserve all records, including missing labels.
        Monthly counts omit invalid dates and include zero-count calendar months
        between observed endpoints. Zeros refer only to the selected data.
    """
    from sensd_sers_analysis.config.inventory import MISSING_METADATA_LABEL

    records = spectra.copy()
    dimensions = tuple(
        dict.fromkeys((group_by, "operator", "serotype", "sensor_id", "target_concentration"))
    )
    for column in dimensions:
        if column not in records:
            records[column] = MISSING_METADATA_LABEL
        else:
            records[column] = (
                records[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .fillna(MISSING_METADATA_LABEL)
            )
    segment = "target_concentration" if group_by == "serotype" else "serotype"
    composition = build_inventory_counts(spectra, (group_by, segment))
    coverage = (
        records.groupby(list(COVERAGE_COLUMNS), observed=True)
        .size()
        .rename("spectra")
        .reset_index()
    )
    dates = pd.to_datetime(
        spectra.get("date", pd.Series(index=spectra.index, dtype="object")),
        errors="coerce",
        format="mixed",
    )
    months = dates.dropna().dt.to_period("M")
    if months.empty:
        timeline = pd.DataFrame(columns=["month", "spectra"])
    else:
        counts = months.value_counts().sort_index()
        counts = counts.reindex(pd.period_range(months.min(), months.max(), freq="M"), fill_value=0)
        timeline = counts.rename_axis("month").rename("spectra").reset_index()
        timeline["month"] = timeline["month"].astype(str)
    return {"composition": composition, "coverage": coverage, "timeline": timeline}


def metadata_issues(spectra: pd.DataFrame) -> pd.DataFrame:
    """Return actionable per-spectrum metadata problems without altering measurements."""
    from sensd_sers_analysis.processing.metadata import sample_type_masks

    issues = []
    identifiers = [c for c in ("filename", "signal_index", "sensor_id") if c in spectra]
    for field in (
        "date",
        "sensor_id",
        "serotype",
        "sample_type",
        "target_concentration",
        "concentration",
    ):
        values = spectra.get(field, pd.Series(index=spectra.index, dtype="object"))
        missing = values.isna() | values.astype("string").str.strip().eq("").fillna(False)
        if field == "date":
            missing = pd.to_datetime(values, errors="coerce", format="mixed").isna()
        elif field == "sample_type":
            controls, bacteria = sample_type_masks(spectra)
            missing = ~(controls | bacteria)
        elif field in ("target_concentration", "concentration"):
            missing = pd.to_numeric(values, errors="coerce").isna()
        if missing.any():
            rows = spectra.loc[missing, identifiers].copy()
            rows["field"] = field
            rows["value"] = values.loc[missing].astype("string")
            rows["issue"] = "Missing or invalid value"
            issues.append(rows)
    return pd.concat(issues, ignore_index=True) if issues else pd.DataFrame()


def inventory_dimension_values(spectra: pd.DataFrame, dimensions: tuple[str, ...]) -> pd.DataFrame:
    """Prepare readable categorical values, retaining unknowns and deriving acquisition month."""
    from sensd_sers_analysis.config.inventory import MISSING_METADATA_LABEL

    records = pd.DataFrame(index=spectra.index)
    for dimension in dimensions:
        values = spectra.get(dimension, pd.Series(index=spectra.index, dtype="object"))
        if dimension == "month":
            dates = spectra.get("date", pd.Series(index=spectra.index, dtype="object"))
            values = pd.to_datetime(dates, errors="coerce", format="mixed").dt.strftime("%Y-%m")
        elif dimension == "target_concentration":
            values = values.map(
                lambda value: (
                    f"{value:g}" if isinstance(value, (int, float)) and pd.notna(value) else value
                )
            )
        records[dimension] = (
            values.astype("string").str.strip().replace("", pd.NA).fillna(MISSING_METADATA_LABEL)
        )
    return records


def build_inventory_counts(spectra: pd.DataFrame, dimensions: tuple[str, ...]) -> pd.DataFrame:
    """Count spectra across distinct selected dimensions without dropping missing metadata."""
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Chart dimensions must be distinct.")
    records = inventory_dimension_values(spectra, dimensions)
    return records.groupby(list(dimensions), observed=True).size().rename("spectra").reset_index()

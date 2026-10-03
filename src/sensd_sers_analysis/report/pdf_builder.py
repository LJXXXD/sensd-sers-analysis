"""PDF reports with complete wrapped tables, proportional figures, and explicit scope."""

import io
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from sensd_sers_analysis.config import (
    BATCH_DEVIATION_Z_THRESHOLD,
    GLOBAL_QA_R2_MIN_THRESHOLD,
    GLOBAL_QA_REJECTION_MULTIPLIER,
)


def _styles():
    """Use packaged Unicode fonts for symbols in scientific quantities."""
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    if "SersSans" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SersSans", str(font_dir / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("SersSans-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFontFamily(
            "SersSans",
            normal="SersSans",
            bold="SersSans-Bold",
            italic="SersSans",
            boldItalic="SersSans-Bold",
        )
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = (
            "SersSans-Bold" if style.name.startswith(("Heading", "Title")) else "SersSans"
        )
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 13
    styles["Title"].fontSize = 18
    styles["Title"].leading = 23
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    styles["Heading2"].keepWithNext = True
    return styles


def _df_to_table_data(df: pd.DataFrame, *, float_fmt: str = "{:.4g}") -> list[list[str]]:
    """Keep every header/value; undefined values use an em dash."""
    rows = [[str(column) for column in df.columns]]
    for row in df.itertuples(index=False, name=None):
        rows.append(
            [
                "—"
                if pd.isna(value)
                else float_fmt.format(value)
                if isinstance(value, float)
                else str(value)
                for value in row
            ]
        )
    return rows


def _compute_table_col_widths(table_data: list[list[str]], usable_width: float) -> list[float]:
    """Bound content weights so a long identifier does not collapse other columns."""
    if not table_data or not table_data[0]:
        return []
    weights = [
        min(30, max(8, max(len(str(row[column])) for row in table_data)))
        for column in range(len(table_data[0]))
    ]
    return [usable_width * weight / sum(weights) for weight in weights]


def _figure_to_image_bytes(fig, *, dpi: int = 150, format: str = "png") -> bytes:
    """Serialize a caller-owned figure without changing its lifecycle."""
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
        return buffer.getvalue()


def _image(fig, *, max_width: float = 6.35 * inch, max_height: float = 4.2 * inch) -> Image:
    """Fit an image inside page bounds while preserving its aspect ratio."""
    image = Image(io.BytesIO(_figure_to_image_bytes(fig)))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _table(df: pd.DataFrame, styles, width: float) -> Table:
    """Wrap safe text, repeat headers, and preserve all rows in a page-spanning table."""
    raw = _df_to_table_data(df)
    body = ParagraphStyle(
        "Cell", parent=styles["Normal"], fontSize=7.5, leading=10, splitLongWords=True
    )
    header = ParagraphStyle(
        "HeaderCell", parent=body, fontName="SersSans-Bold", textColor=colors.white
    )
    cells = [
        [Paragraph(escape(value), header if row == 0 else body) for value in values]
        for row, values in enumerate(raw)
    ]
    table = Table(
        cells, colWidths=_compute_table_col_widths(raw, width), repeatRows=1, hAlign="LEFT"
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#345878")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _add_table(flow: list, df: pd.DataFrame | None, styles, width: float) -> None:
    """Split wide tables into panels with unambiguous repeated row identity."""
    if df is None or df.empty:
        return
    columns = list(df.columns)
    if len(columns) <= 8:
        flow.append(_table(df, styles, width))
    else:
        identity = [columns[0]]
        panel_frame = df
        if not df.iloc[:, 0].is_unique:
            row_label = "Report row"
            while row_label in columns:
                row_label += " #"
            panel_frame = df.copy()
            panel_frame.insert(0, row_label, range(1, len(df) + 1))
            identity.insert(0, row_label)
        panel_size = 8 - len(identity)
        for start in range(1, len(columns), panel_size):
            flow.append(
                _table(
                    panel_frame[[*identity, *columns[start : start + panel_size]]], styles, width
                )
            )
            flow.append(Spacer(1, 0.12 * inch))
    flow.append(Spacer(1, 0.18 * inch))


def _add_figure(
    flow: list,
    title: str,
    fig,
    styles,
    note: str | None = None,
    *,
    section_title: str | None = None,
) -> None:
    section = []
    if section_title:
        section.append(Paragraph(escape(section_title), styles["Heading2"]))
    section.append(Paragraph(escape(title), styles["Heading2"]))
    if note:
        section.append(Paragraph(escape(note), styles["Normal"]))
    if fig is not None:
        section.extend([Spacer(1, 0.08 * inch), _image(fig)])
    flow.append(KeepTogether(section))
    flow.append(Spacer(1, 0.15 * inch))


def _start(report_title: str):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = _styles()
    flow = [
        Paragraph(escape(report_title), styles["Title"]),
        Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]),
        Spacer(1, 0.15 * inch),
    ]
    return buffer, doc, styles, flow


def _finish(buffer, doc, flow, output_path) -> bytes:
    """Publish bytes only after successful document layout and serialization."""
    try:
        doc.build(flow)
        data = buffer.getvalue()
        if output_path is not None:
            Path(output_path).write_bytes(data)
        return data
    finally:
        buffer.close()


def build_sensor_assessment_pdf(
    *,
    consistency_table: pd.DataFrame | None = None,
    degradation_table: pd.DataFrame | None = None,
    degradation_fig: Any = None,
    batch_variance_table: pd.DataFrame | None = None,
    batch_boxplot_fig: Any = None,
    deviating_sensors_table: pd.DataFrame | None = None,
    outlier_method: str = "iqr",
    degradation_scope: str | None = None,
    report_title: str = "SERS Sensor Assessment Report",
    output_path: str | Path | None = None,
) -> bytes:
    """Compile retrospective consistency, trend, and batch diagnostics.

    Missing sections retain their heading and a data-availability explanation.
    No diagnostic establishes physical sensor failure or pre-use qualification.
    Input figures remain caller-owned. All supplied table values are retained.
    """
    buffer, doc, styles, flow = _start(report_title)
    flow.append(
        Paragraph(
            "Retrospective response diagnostics; these results do not establish physical sensor failure or prospective qualification.",
            styles["Normal"],
        )
    )
    sections = [
        (
            "1. Consistency Metrics (CV)",
            consistency_table,
            f"CV = SD/mean; raw and within-group filtered replicates ({outlier_method}). Undefined CV remains unavailable.",
        ),
        (
            "2. Response Trend Across Tests",
            degradation_table,
            (f"Scope: {degradation_scope}. " if degradation_scope else "")
            + "Linear trend across ordered tests. Negative slope describes signal decline; it does not establish its cause.",
        ),
        (
            "3. Batch Variance",
            batch_variance_table,
            "Per-sensor responses relative to the observed batch mean and SD.",
        ),
        (
            "4. Deviating Sensors",
            deviating_sensors_table,
            f"Configured diagnostic threshold: |z_from_batch| > {BATCH_DEVIATION_Z_THRESHOLD:g}.",
        ),
    ]
    for title, table, note in sections:
        flow.append(Paragraph(title, styles["Heading2"]))
        flow.append(Paragraph(escape(note), styles["Normal"]))
        if table is None or table.empty:
            reason = (
                "No deviations under the configured threshold."
                if title.startswith("4.")
                and batch_variance_table is not None
                and not batch_variance_table.empty
                else "Unavailable: no assessable rows in the selected scope."
            )
            flow.append(Paragraph(reason, styles["Normal"]))
        else:
            _add_table(flow, table, styles, doc.width)
        if title.startswith("2.") and degradation_fig is not None:
            _add_figure(flow, "Response trend", degradation_fig, styles)
        if title.startswith("3.") and batch_boxplot_fig is not None:
            _add_figure(flow, "Between-sensor stability", batch_boxplot_fig, styles)
    return _finish(buffer, doc, flow, output_path)


def build_sensor_assessment_qa_pdf(
    *,
    global_qa_table: pd.DataFrame | None = None,
    overlay_items: list[dict] | None = None,
    macro_items: list[dict] | None = None,
    report_title: str = "Sensor Consistency & Quality Assurance Report",
    output_path: str | Path | None = None,
    unavailable_sections: list[tuple[str, str]] | None = None,
) -> bytes:
    """Compile retrospective sensor fits, exclusions, overlays and pooled fits.

    Expected unavailable plots use explicit section notes. Unexpected plot/write
    errors propagate to the caller; they never become silently omitted sections.
    """
    buffer, doc, styles, flow = _start(report_title)
    flow.append(Paragraph("1. Individual Sensor Assessment", styles["Heading2"]))
    flow.append(
        Paragraph(
            f"Retrospective raw/clean response fits. Excluded when clean RMSE > {GLOBAL_QA_REJECTION_MULTIPLIER:g} × batch median or clean R² < {GLOBAL_QA_R2_MIN_THRESHOLD:.2f}. Unassessed pairs are not Pass. This does not establish physical failures.",
            styles["Normal"],
        )
    )
    if global_qa_table is None or global_qa_table.empty:
        flow.append(
            Paragraph(
                "Unavailable: no assessable sensor fits in the selected scope.", styles["Normal"]
            )
        )
    _add_table(flow, global_qa_table, styles, doc.width)
    for title, items in [
        ("2. Multi-Sensor Regression Overlays", overlay_items),
        ("3. Macro Batch Regressions", macro_items),
    ]:
        if not items:
            _add_figure(
                flow,
                title,
                None,
                styles,
                "No available plots in the selected scope; unavailable requests are listed below.",
            )
        for item_number, item in enumerate(items or []):
            result = item.get("macro_result")
            note = None
            if result is not None:
                note = f"Raw RMSE={result.raw_batch_rmse:.4g}, R²={result.raw_batch_r2:.4g}; clean RMSE={result.clean_batch_rmse:.4g}, R²={result.clean_batch_r2:.4g}; macro outliers={result.n_macro_outliers}."
            _add_figure(
                flow,
                f"{item['serotype']} — {item['feature']}",
                item["fig"],
                styles,
                note,
                section_title=title if item_number == 0 else None,
            )
    for title, reason in unavailable_sections or []:
        _add_figure(flow, title, None, styles, f"Unavailable: {reason}")
    return _finish(buffer, doc, flow, output_path)


def build_classification_report_pdf(
    *,
    pca_fig: Any = None,
    feature_importance_fig: Any = None,
    rf_confusion_matrix_fig: Any = None,
    svm_confusion_matrix_fig: Any = None,
    rf_accuracy: float | None = None,
    rf_f1: float | None = None,
    svm_accuracy: float | None = None,
    svm_f1: float | None = None,
    best_model_name: str | None = None,
    report_title: str = "Serotyping & Classification Report",
    output_path: str | Path | None = None,
    pca_unavailable_reason: str | None = None,
    caption_lines: tuple[str, ...] | None = None,
) -> bytes:
    """Report sensor-held-out metrics separately from exploratory cohort PCA.

    ``best_model_name`` is chosen using comparable training CV scores, or the
    predeclared RF reference when CV is infeasible. It is never a test winner.
    """
    buffer, doc, styles, flow = _start(report_title)
    flow.append(
        Paragraph(
            "Identity-eligible sample rows without retrospective response-based screening. Sensor-group holdout; scaling and tuning use training folds. Selected predictors are independent per-spectrum scalars and fixed peak heights. These are retrospective metrics, not prospective qualification.",
            styles["Normal"],
        )
    )
    if best_model_name:
        flow.append(
            Paragraph(
                f"Selected by training CV; RF reference when CV is unavailable: {escape(best_model_name)}.",
                styles["Normal"],
            )
        )
    for line in caption_lines or ():
        flow.append(Paragraph(escape(line), styles["Normal"]))
    _add_figure(
        flow,
        "1. Exploratory Cohort PCA",
        pca_fig,
        styles,
        pca_unavailable_reason
        or "Full-cohort PC1/PC2 colored by sample class. This exploratory view supplies no model predictors.",
    )
    _add_figure(
        flow,
        "2. Random Forest Feature Importance",
        feature_importance_fig,
        styles,
        "Impurity-based importance describes this fitted model; it does not validate peak specificity or causal value."
        if feature_importance_fig is not None
        else "Unavailable: this result has no feature importances.",
    )
    for title, fig, accuracy, f1 in [
        ("3. Random Forest Confusion Matrix", rf_confusion_matrix_fig, rf_accuracy, rf_f1),
        ("4. SVM (RBF) Confusion Matrix", svm_confusion_matrix_fig, svm_accuracy, svm_f1),
    ]:
        metrics = [
            f"Accuracy={accuracy:.3f}" if accuracy is not None else "Accuracy unavailable",
            f"Weighted F1={f1:.3f}" if f1 is not None else "F1 unavailable",
        ]
        _add_figure(flow, title, fig, styles, "; ".join(metrics))
    return _finish(buffer, doc, flow, output_path)


def build_regression_concentration_pdf(
    *,
    paradigm_title: str,
    metrics_table: pd.DataFrame | None = None,
    scatter_fig: Any = None,
    residual_fig: Any = None,
    extra_figures: list[tuple[str, Any]] | None = None,
    caption_lines: tuple[str, ...] | None = None,
    report_title: str = "Concentration regression report",
    output_path: str | Path | None = None,
) -> bytes:
    """Report positive actual-CFU log10 regression on held-out sensor groups."""
    buffer, doc, styles, flow = _start(report_title)
    flow.append(Paragraph(escape(paradigm_title), styles["Heading2"]))
    flow.append(
        Paragraph(
            "Identity-eligible bacterial rows with finite positive measured concentration; no response-based QA screening. Target: log10(actual CFU/mL). Evaluation uses sensor-group holdout and training-only preprocessing/model selection. Sensor identifiers do not establish independent biological preparations. These retrospective results do not establish prospective performance.",
            styles["Normal"],
        )
    )
    for line in caption_lines or ():
        flow.append(Paragraph(escape(line), styles["Normal"]))
    flow.append(Paragraph("Held-out metrics", styles["Heading2"]))
    _add_table(flow, metrics_table, styles, doc.width)
    _add_figure(flow, "Actual vs predicted", scatter_fig, styles)
    _add_figure(flow, "Residuals", residual_fig, styles)
    for title, fig in extra_figures or []:
        _add_figure(flow, title, fig, styles)
    return _finish(buffer, doc, flow, output_path)

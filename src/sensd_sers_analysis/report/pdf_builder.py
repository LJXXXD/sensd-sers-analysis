"""
PDF report generation for Sensor Assessment results.

Uses ReportLab to compile consistency metrics, degradation trends, batch
variance, and outlier impact into a professional PDF document. Design is
modular to support future metric injection.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _df_to_table_data(
    df: pd.DataFrame,
    *,
    max_cols: int = 12,
    float_fmt: str = "{:.4g}",
) -> list[list[str]]:
    """Convert DataFrame to list of lists for ReportLab Table."""
    df_str = df.copy()
    for c in df_str.select_dtypes(include=["float", "floating"]).columns:
        df_str[c] = df_str[c].apply(
            lambda x: float_fmt.format(x)
            if pd.notna(x) and isinstance(x, (int, float))
            else ""
        )
    df_str = df_str.fillna("—")
    headers = [str(c)[:20] for c in df_str.columns]
    rows = [headers] + df_str.astype(str).values.tolist()
    return rows


def _figure_to_image_bytes(fig, *, dpi: int = 150, format: str = "png") -> bytes:
    """Serialize matplotlib Figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_sensor_assessment_pdf(
    *,
    consistency_table: Optional[pd.DataFrame] = None,
    degradation_table: Optional[pd.DataFrame] = None,
    degradation_fig: Optional[Any] = None,
    batch_variance_table: Optional[pd.DataFrame] = None,
    batch_boxplot_fig: Optional[Any] = None,
    deviating_sensors_table: Optional[pd.DataFrame] = None,
    outlier_method: str = "iqr",
    report_title: str = "SERS Sensor Assessment Report",
    output_path: Optional[str | Path] = None,
) -> bytes:
    """
    Compile assessment results into a PDF report.

    All arguments are optional; only provided sections are included.
    Design is modular: add new tables/figures by extending the flow.

    Args:
        consistency_table: Summary of CV (raw vs filtered) per feature/group.
        degradation_table: Slope, R², p-value from degradation analysis.
        degradation_fig: matplotlib Figure for degradation trend plot.
        batch_variance_table: Per-sensor mean, std, z_from_batch.
        batch_boxplot_fig: matplotlib Figure for batch boxplot.
        deviating_sensors_table: Sensors with |z| > threshold.
        outlier_method: Method used for outlier filtering (for report text).
        report_title: Title on first page.
        output_path: If provided, also save PDF to this path.

    Returns:
        PDF file contents as bytes (for Streamlit download).
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=18,
        spaceAfter=8,
    )
    body_style = styles["Normal"]

    flow: list = []

    # Title and metadata
    flow.append(Paragraph(report_title, title_style))
    flow.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            body_style,
        )
    )
    flow.append(Spacer(1, 0.25 * inch))

    # 1. Consistency Metrics
    if consistency_table is not None and not consistency_table.empty:
        flow.append(Paragraph("1. Consistency Metrics (CV)", heading_style))
        flow.append(
            Paragraph(
                f"Coefficient of Variation (σ/μ) for key features. "
                f"Raw vs. outlier-filtered (method: {outlier_method}). "
                f"Lower CV indicates better consistency.",
                body_style,
            )
        )
        flow.append(Spacer(1, 0.1 * inch))

        table_data = _df_to_table_data(
            consistency_table,
            float_fmt="{:.4f}",
        )
        col_count = len(table_data[0])
        usable_width = 6.5 * inch
        col_widths = [usable_width / max(col_count, 1)] * col_count
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f5f5")],
                    ),
                ]
            )
        )
        flow.append(t)
        flow.append(Spacer(1, 0.3 * inch))

    # 2. Degradation Analysis
    if degradation_table is not None and not degradation_table.empty:
        flow.append(Paragraph("2. Degradation Analysis", heading_style))
        flow.append(
            Paragraph(
                "Linear regression of feature vs. sequence/timestamp. "
                "Negative slope indicates degradation.",
                body_style,
            )
        )
        flow.append(Spacer(1, 0.1 * inch))

        table_data = _df_to_table_data(degradation_table, float_fmt="{:.4g}")
        col_count = len(table_data[0])
        usable_width = 6.5 * inch
        col_widths = [usable_width / max(col_count, 1)] * col_count
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#70AD47")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#e8f4e4")],
                    ),
                ]
            )
        )
        flow.append(t)
        flow.append(Spacer(1, 0.15 * inch))

    if degradation_fig is not None:
        img_bytes = _figure_to_image_bytes(degradation_fig)
        img = Image(io.BytesIO(img_bytes), width=5.5 * inch, height=3.5 * inch)
        flow.append(img)
        flow.append(Spacer(1, 0.3 * inch))

    # 3. Batch Variance
    if batch_variance_table is not None and not batch_variance_table.empty:
        flow.append(Paragraph("3. Batch Variance", heading_style))
        flow.append(
            Paragraph(
                "Per-sensor statistics and deviation from batch mean. "
                "z_from_batch indicates how many standard deviations a sensor "
                "is from the population mean.",
                body_style,
            )
        )
        flow.append(Spacer(1, 0.1 * inch))

        # Limit to key columns if too wide
        key_cols = [
            c
            for c in [
                "sensor_id",
                "n_samples",
                "mean",
                "std",
                "cv",
                "z_from_batch",
                "deviation_pct",
            ]
            if c in batch_variance_table.columns
        ]
        if not key_cols:
            key_cols = batch_variance_table.columns[:8].tolist()
        sub = batch_variance_table[key_cols].head(30)
        table_data = _df_to_table_data(sub, float_fmt="{:.4g}")
        col_count = len(table_data[0])
        usable_width = 6.5 * inch
        col_widths = [usable_width / max(col_count, 1)] * col_count
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ED7D31")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#fde9db")],
                    ),
                ]
            )
        )
        flow.append(t)
        if len(batch_variance_table) > 30:
            flow.append(
                Paragraph(
                    f"<i>(Showing first 30 of {len(batch_variance_table)} sensors)</i>",
                    body_style,
                )
            )
        flow.append(Spacer(1, 0.15 * inch))

    if batch_boxplot_fig is not None:
        img_bytes = _figure_to_image_bytes(batch_boxplot_fig)
        img = Image(io.BytesIO(img_bytes), width=5.5 * inch, height=3.5 * inch)
        flow.append(img)
        flow.append(Spacer(1, 0.3 * inch))

    # 4. Deviating Sensors
    if deviating_sensors_table is not None and not deviating_sensors_table.empty:
        flow.append(Paragraph("4. Deviating Sensors", heading_style))
        flow.append(
            Paragraph(
                "Sensors with |z_from_batch| > 2.0 (or configured threshold).",
                body_style,
            )
        )
        flow.append(Spacer(1, 0.1 * inch))
        table_data = _df_to_table_data(deviating_sensors_table, float_fmt="{:.4g}")
        col_count = len(table_data[0])
        usable_width = 6.5 * inch
        col_widths = [usable_width / max(col_count, 1)] * col_count
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C55A11")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ]
            )
        )
        flow.append(t)

    doc.build(flow)
    pdf_bytes = buffer.getvalue()

    if output_path is not None:
        Path(output_path).write_bytes(pdf_bytes)

    return pdf_bytes

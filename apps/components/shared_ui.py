"""
Shared UI components for SERS Data Explorer.
"""

import logging
import hashlib
from collections.abc import Callable

import pandas as pd
import streamlit as st

from theme import (
    DATAFRAME_WIDTH,
    FIGURE_WIDTH,
    HIDE_INDEX,
)

logger = logging.getLogger(__name__)


def report_context_key(df: pd.DataFrame, *settings) -> str:
    """Fingerprint report inputs so a changed scope cannot download stale bytes."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes())
    digest.update(repr((tuple(df.columns), settings)).encode("utf-8"))
    return digest.hexdigest()


def render_pdf_download_section(
    session_key: str,
    filename: str,
    generate_callback: Callable[[], bytes],
    *,
    context_key: str,
    button_label: str = "Generate report",
    download_label: str = "Download PDF",
    mime: str = "application/pdf",
    container=None,
) -> None:
    """
    Render Generate button and Download button for PDF reports.

    When Generate is clicked, calls generate_callback() to produce PDF bytes,
    stores them in st.session_state[session_key], and shows success. The
    Download button appears ONLY after successful generation (never before).

    Args:
        session_key: Key in st.session_state for PDF bytes.
        context_key: Stable digest of the current report data and settings.
        filename: Download filename (e.g. "sensor_assessment_report.pdf").
        generate_callback: Callable that returns PDF bytes. Called when
            Generate button is clicked.
        button_label: Label for the Generate button.
        download_label: Label for the Download button.
        mime: MIME type for download.
        container: Streamlit container (default: st).
    """
    c = container if container is not None else st
    state_context_key = f"{session_key}_context"
    if st.session_state.get(state_context_key) != context_key:
        st.session_state.pop(session_key, None)
        st.session_state.pop(state_context_key, None)

    with c.container():
        if st.button(button_label, key=f"{session_key}_btn", type="primary"):
            try:
                pdf_bytes = generate_callback()
                if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
                    raise ValueError("Report generation returned no PDF bytes.")
                st.session_state[session_key] = pdf_bytes
                st.session_state[state_context_key] = context_key
                logger.info("PDF report generated successfully: %s", session_key)
                st.success("Report generated. Click Download below.")
            except Exception as e:
                logger.exception("Report generation failed (%s): %s", session_key, e)
                st.session_state.pop(session_key, None)
                st.session_state.pop(state_context_key, None)
                st.error(f"Report generation failed: {e}")

        _pdf_ready = (
            session_key in st.session_state
            and isinstance(st.session_state[session_key], bytes)
            and len(st.session_state[session_key]) > 0
        )
        if _pdf_ready:
            st.download_button(
                label=download_label,
                data=st.session_state[session_key],
                file_name=filename,
                mime=mime,
                key=f"{session_key}_download",
            )


def render_metrics_row(
    metrics: list[tuple[str, str]],
    *,
    container=None,
) -> None:
    """
    Render a row of metric (label, value) pairs in columns.

    Args:
        metrics: List of (label, value) tuples.
        container: Streamlit container (default: st).
    """
    c = container if container is not None else st
    if not metrics:
        return
    cols = c.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)


def render_dataframe_stretch(
    df: pd.DataFrame,
    *,
    hide_index: bool = HIDE_INDEX,
    column_config: dict | None = None,
    **kwargs,
) -> None:
    """
    Render a DataFrame with standard stretch width and hide_index.

    Args:
        df: DataFrame to display.
        hide_index: Whether to hide the index column.
        column_config: Optional Streamlit column_config dict.
        **kwargs: Additional arguments passed to st.dataframe.
    """
    combined = {
        "width": DATAFRAME_WIDTH,
        "hide_index": hide_index,
        **kwargs,
    }
    if column_config is not None:
        combined["column_config"] = column_config
    st.dataframe(df, **combined)


def render_figure_stretch(
    fig,
    *,
    close: bool = True,
) -> None:
    """
    Render a matplotlib figure with stretch width and optionally close it.

    Args:
        fig: matplotlib Figure to display.
        close: If True, close the figure after rendering to free memory.
    """
    try:
        st.pyplot(fig, width=FIGURE_WIDTH)
    finally:
        if close:
            import matplotlib.pyplot as plt

            plt.close(fig)

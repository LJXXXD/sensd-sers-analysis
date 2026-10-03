"""Report failures, figure ownership and full text are observable."""

import io
from contextlib import ExitStack
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pypdf import PdfReader

from sensd_sers_analysis.application import classification_service, sensor_assessment_service
from sensd_sers_analysis.report.figures import own_figure
from sensd_sers_analysis.report.pdf_builder import build_sensor_assessment_pdf
from sensd_sers_analysis.utils.availability import PlotUnavailableError
from sensd_sers_analysis.visualization.assessment_plots import plot_multi_sensor_regression


def test_constant_concentration_overlay_preserves_scatter_without_fake_fit():
    frame = pd.DataFrame(
        {
            "sensor_id": ["S1", "S1"],
            "serotype": ["SH", "SH"],
            "log_concentration": [2.0, 2.0],
            "f": [1.0, 3.0],
        }
    )
    before = plt.get_fignums()
    fig = plot_multi_sensor_regression(frame, "SH", "f")
    assert not fig.axes[0].lines
    assert len(fig.axes[0].collections[0].get_offsets()) == 2
    assert plt.get_fignums() == before
    fig.clear()


def test_owned_figures_cleaned_on_failure_and_other_figures_preserved():
    unrelated = plt.figure()
    owned = plt.figure()
    try:
        with pytest.raises(RuntimeError), ExitStack() as stack:
            own_figure(stack, owned)
            raise RuntimeError("serialize failure")
        assert plt.fignum_exists(unrelated.number)
        assert not plt.fignum_exists(owned.number)
    finally:
        plt.close(unrelated)


def test_qa_only_expected_unavailability_becomes_report_note(monkeypatch):
    qa = SimpleNamespace(table=pd.DataFrame())
    overlay = SimpleNamespace(
        serotype="ST", feature="f", excluded_sensors=set(), pass_sensors=set()
    )

    def absent(*a, **kw):
        raise PlotUnavailableError("no finite measurements")

    monkeypatch.setattr(sensor_assessment_service, "plot_multi_sensor_regression", absent)
    monkeypatch.setattr(sensor_assessment_service, "plot_macro_batch_regression", absent)
    captured = {}

    def builder(**kwargs):
        captured.update(kwargs)
        return b"pdf"

    monkeypatch.setattr(sensor_assessment_service, "build_sensor_assessment_qa_pdf", builder)
    assert (
        sensor_assessment_service.build_sensor_assessment_qa_pdf_bytes(
            pd.DataFrame(), qa, [overlay], report_title="QA"
        )
        == b"pdf"
    )
    assert len(captured["unavailable_sections"]) == 2

    def broken(*a, **kw):
        raise ValueError("implementation contract failed")

    monkeypatch.setattr(sensor_assessment_service, "plot_multi_sensor_regression", broken)
    with pytest.raises(ValueError, match="implementation contract"):
        sensor_assessment_service.build_sensor_assessment_qa_pdf_bytes(
            pd.DataFrame(), qa, [overlay], report_title="QA"
        )


def test_classification_later_plot_failure_releases_already_created_figure(monkeypatch):
    pca = Figure()
    pca.subplots().plot([0, 1])
    result = SimpleNamespace(feature_importances=[1])
    artifacts = SimpleNamespace(
        clean_classification_df=pd.DataFrame(), rf_result=result, svm_result=result
    )
    monkeypatch.setattr(classification_service, "plot_pca_classification", lambda *a: pca)

    def broken(*a):
        raise RuntimeError("second plot failed")

    monkeypatch.setattr(classification_service, "plot_feature_importance", broken)
    with pytest.raises(RuntimeError, match="second plot"):
        classification_service.build_classification_report_pdf_bytes(artifacts)
    assert not pca.axes


def test_pdf_wraps_complete_headers_and_special_text_without_omitting_sections():
    title = "Retrospective <scope> & measurements"
    header = "Complete descriptive header longer than twenty characters"
    frame = pd.DataFrame(
        {"sensor_id": ["source<&>" + "long-name-" * 10], header: [1.234], "Greek σ/μ": [2.0]}
    )
    pdf = build_sensor_assessment_pdf(consistency_table=frame, report_title=title)
    text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert title in text
    assert header in " ".join(text.split())
    assert "Response Trend Across Tests" in text
    assert "Unavailable: no assessable rows" in text
    assert "σ/μ" in text


def test_wide_pdf_panels_repeat_unambiguous_row_identity_and_every_column():
    frame = pd.DataFrame(
        {
            "sensor_id": ["same-sensor", "same-sensor"],
            **{f"metric_{i}": [i, i + 100] for i in range(12)},
        }
    )
    pdf = build_sensor_assessment_pdf(consistency_table=frame)
    text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert text.count("Report row") >= 2
    for i in range(12):
        assert f"metric_{i}" in text
        assert str(i + 100) in text

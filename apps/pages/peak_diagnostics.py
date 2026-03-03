"""
Peak Detection Diagnostics tab — serotype-specific peak verification.
"""

import numpy as np
import pandas as pd
import streamlit as st

from sensd_sers_analysis.data import get_raman_shift, get_signals_matrix
from sensd_sers_analysis.utils import order_concentration_labels


def render(filtered_features, wide_df):
    """Render the Peak Detection Diagnostics tab."""
    import matplotlib.pyplot as plt

    peak_by_sero = st.session_state.get("peak_infos_by_serotype", {})
    mean_by_sero = st.session_state.get("mean_spec_by_serotype", {})
    raman_x = st.session_state.get("raman_x", np.array([]))

    if not peak_by_sero or raman_x.size == 0:
        st.info(
            "Peak detection requires loaded data with Raman intensity columns. "
            "Adjust **Peaks per serotype** in the sidebar and ensure high-concentration "
            "samples are present (>0 CFU, serotype-specific)."
        )
        return

    st.markdown(
        "Visual verification of dynamic peak extraction: serotype-specific "
        "anchors, search windows, and detection success rates (0 CFU excluded from "
        "learning). Each serotype uses its own peak count from the sidebar."
    )

    for sel_serotype in sorted(peak_by_sero.keys()):
        peak_infos = peak_by_sero.get(sel_serotype, [])
        mean_spec = mean_by_sero.get(sel_serotype, np.array([]))

        if not peak_infos or mean_spec.size == 0:
            continue

        with st.expander(
            f"**{sel_serotype}** — Mean spectrum & diagnostics", expanded=True
        ):
            fig_anchor, ax_anchor = plt.subplots(figsize=(14, 4))
            ax_anchor.plot(
                raman_x,
                mean_spec,
                color="C0",
                linewidth=1.5,
                label=f"Mean ({sel_serotype}, high conc)",
            )
            for i, info in enumerate(peak_infos):
                ax_anchor.axvline(
                    info.center,
                    color=f"C{(i % 9) + 1}",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"{info.peak_name} @ {info.center:.0f} cm⁻¹",
                )
                ax_anchor.axvspan(
                    info.window_min,
                    info.window_max,
                    alpha=0.15,
                    color=f"C{(i % 9) + 1}",
                )
            ax_anchor.set_xlabel("Raman shift (cm⁻¹)")
            ax_anchor.set_ylabel("Intensity")
            ax_anchor.set_title(
                f"{sel_serotype} | Dashed = Voted Centers, Shaded = Search Windows"
            )
            ax_anchor.legend(loc="upper right", fontsize=8, ncol=2)
            ax_anchor.grid(True, alpha=0.3)
            fig_anchor.tight_layout()
            st.pyplot(fig_anchor, width="stretch")
            plt.close(fig_anchor)

            diag_data = [
                {
                    "Peak Name": p.peak_name,
                    "Center (cm⁻¹)": f"{p.center:.1f}",
                    "Window Range": f"[{p.window_min:.1f}, {p.window_max:.1f}]",
                    "Detection Success Rate (%)": f"{p.success_rate * 100:.1f}",
                }
                for p in peak_infos
            ]
            st.dataframe(pd.DataFrame(diag_data), width="stretch", hide_index=True)

    st.markdown("#### Signal-level verification")
    st.caption(
        "Inspect a single spectrum: shaded regions = serotype-specific search "
        "windows; green ★ = where the algorithm detected a peak (passed prominence "
        "check). Pick a serotype below to filter signals — this ties verification "
        "to the plots above."
    )
    wide_filtered = (
        wide_df.loc[filtered_features.index] if not filtered_features.empty else wide_df
    )
    sensor_col = "sensor_id" if "sensor_id" in filtered_features.columns else None
    conc_col = (
        "concentration_group"
        if "concentration_group" in filtered_features.columns
        else None
    )
    serotype_col = "serotype" if "serotype" in filtered_features.columns else None

    if not (sensor_col and conc_col and not wide_filtered.empty):
        st.caption(
            "Signal-level verification requires sensor_id and concentration_group "
            "in the data."
        )
        return

    sero_opts_ver = sorted(peak_by_sero.keys())
    sel_sero_ver = st.selectbox(
        "Serotype (filter signals)",
        options=sero_opts_ver,
        index=0,
        key="peak_diag_serotype_filter",
        help="Only show sensors/concentrations that have signals of this serotype.",
    )
    df_ver = (
        filtered_features[filtered_features[serotype_col].astype(str) == sel_sero_ver]
        if serotype_col
        else filtered_features
    )

    sensor_opts = (
        sorted(df_ver[sensor_col].dropna().unique().astype(str).tolist())
        if not df_ver.empty
        else []
    )
    conc_opts = (
        order_concentration_labels(
            df_ver[conc_col].dropna().unique().astype(str).tolist()
        )
        or ["(none)"]
        if not df_ver.empty
        else ["(none)"]
    )

    if not sensor_opts or not conc_opts:
        st.info(
            f"No signals for serotype **{sel_sero_ver}** in the current filters. "
            "Adjust filters or select another serotype."
        )
        return

    col_s, col_c, col_sig = st.columns(3)
    with col_s:
        sel_sensor = st.selectbox(
            "Sensor ID",
            options=sensor_opts,
            index=0,
            key="peak_diag_sensor",
        )
    with col_c:
        sel_conc = st.selectbox(
            "Concentration",
            options=conc_opts,
            index=0,
            key="peak_diag_conc",
        )
    matches = df_ver[
        (df_ver[sensor_col].astype(str) == sel_sensor)
        & (df_ver[conc_col].astype(str) == sel_conc)
    ]
    signal_labels = []
    for idx in matches.index:
        fn = (
            wide_filtered.loc[idx, "filename"]
            if "filename" in wide_filtered.columns
            else str(idx)
        )
        signal_labels.append(f"{fn} (idx {idx})")

    with col_sig:
        if len(signal_labels) > 1:
            sel_idx = st.selectbox(
                "Signal",
                options=range(len(signal_labels)),
                format_func=lambda i: signal_labels[i],
                key="peak_diag_signal",
            )
            row_idx = matches.index[sel_idx]
        else:
            row_idx = matches.index[0] if len(matches) > 0 else None

    if row_idx is None:
        st.warning("No matching signal for selected sensor and concentration.")
        return

    row_sero = (
        str(wide_filtered.loc[row_idx, serotype_col])
        if serotype_col and serotype_col in wide_filtered.columns
        else sel_sero_ver
    )
    row_peak_infos = peak_by_sero.get(row_sero)
    if not row_peak_infos:
        row_peak_infos = peak_by_sero.get(st.session_state.get("peak_default_serotype"))
    if not row_peak_infos:
        row_peak_infos = next(iter(peak_by_sero.values()), [])

    spec_row = wide_filtered.loc[[row_idx]]
    sig_mat = get_signals_matrix(spec_row)
    raman = get_raman_shift(spec_row)
    y_spec = sig_mat[0]
    x_spec = np.asarray(raman, dtype=float)

    valid = np.isfinite(y_spec.astype(float))
    x_plot = x_spec[valid]
    y_plot = np.asarray(y_spec, dtype=float)[valid]
    sort_idx = np.argsort(x_plot)
    x_plot = x_plot[sort_idx]
    y_plot = y_plot[sort_idx]

    fig_sig, ax_sig = plt.subplots(figsize=(14, 5))
    ax_sig.plot(x_plot, y_plot, color="C0", linewidth=1.2, label="Raw spectrum")

    for i, info in enumerate(row_peak_infos):
        ax_sig.axvspan(
            info.window_min,
            info.window_max,
            alpha=0.12,
            color=f"C{(i % 9) + 1}",
        )
        peak_col = info.peak_name
        if peak_col in filtered_features.columns:
            val = filtered_features.loc[row_idx, peak_col]
            if pd.notna(val) and np.isfinite(val):
                mask = (x_spec >= info.window_min) & (x_spec <= info.window_max)
                window_y = np.where(mask, y_spec.astype(float), np.nan)
                if mask.any() and np.any(np.isfinite(window_y)):
                    local_idx = int(np.nanargmax(window_y))
                    peak_x = float(x_spec[local_idx])
                    peak_y = float(y_spec[local_idx])
                    ax_sig.scatter(
                        peak_x,
                        peak_y,
                        marker="*",
                        s=200,
                        color="green",
                        edgecolors="darkgreen",
                        linewidths=1.5,
                        zorder=5,
                        label="Detected" if i == 0 else None,
                    )

    ax_sig.set_xlabel("Raman shift (cm⁻¹)")
    ax_sig.set_ylabel("Intensity")
    ax_sig.set_title(
        f"Signal: {sel_sensor} @ {sel_conc} ({row_sero}) | Green ★ = detected peaks"
    )
    ax_sig.legend(loc="upper right", fontsize=8)
    ax_sig.grid(True, alpha=0.3)
    fig_sig.tight_layout()
    st.pyplot(fig_sig, width="stretch")
    plt.close(fig_sig)

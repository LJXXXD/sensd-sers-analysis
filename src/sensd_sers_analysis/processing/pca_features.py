"""
Principal Component Analysis (PCA) feature extraction from SERS spectra.

Extracts PC1 and PC2 from raw spectral intensity columns (rs_*) for use as
assessable features alongside max_intensity, mean_intensity, and integral_area.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sensd_sers_analysis.data import get_signals_matrix


def add_pca_features(df_wide: pd.DataFrame, *, n_components: int = 2) -> pd.DataFrame:
    """
    Extract PCA features from spectral intensity columns and append to DataFrame.

    Isolates numerical spectral columns (rs_*), standardizes them, fits PCA
    with n_components=2, and appends PC1, PC2, PC1_var_ratio, PC2_var_ratio.
    Output is a DataFrame with one row per sample (matching df_wide index) and
    the four new columns. Use this result to merge PCA features into the
    feature DataFrame from extract_basic_features.

    Args:
        df_wide: Wide-format SERS DataFrame with rs_* intensity columns.
        n_components: Number of principal components (default 2).

    Returns:
        DataFrame with index matching df_wide and columns: PC1, PC2,
        PC1_var_ratio, PC2_var_ratio. Empty DataFrame if insufficient data.

    Example:
        >>> wide = load_sers_data("example_data/")
        >>> pca_df = add_pca_features(wide)
        >>> features = extract_basic_features(wide)
        >>> features = features.join(pca_df)
    """
    if df_wide.empty:
        return pd.DataFrame(
            columns=["PC1", "PC2", "PC1_var_ratio", "PC2_var_ratio"],
            index=df_wide.index,
        )

    signals = get_signals_matrix(df_wide)
    if signals.size == 0 or signals.shape[1] < 2:
        return pd.DataFrame(
            columns=["PC1", "PC2", "PC1_var_ratio", "PC2_var_ratio"],
            index=df_wide.index,
        )

    # Replace inf/nan with 0 for scaling (or drop rows with too many NaN)
    X = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_eff = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    if n_eff < 1:
        return pd.DataFrame(
            columns=["PC1", "PC2", "PC1_var_ratio", "PC2_var_ratio"],
            index=df_wide.index,
        )

    pca = PCA(n_components=n_eff)
    pca_scores = pca.fit_transform(X_scaled)

    var_ratios = pca.explained_variance_ratio_
    var1 = float(var_ratios[0]) if len(var_ratios) > 0 else np.nan
    var2 = float(var_ratios[1]) if len(var_ratios) > 1 else np.nan

    pc1 = (
        pca_scores[:, 0] if pca_scores.shape[1] >= 1 else np.full(len(df_wide), np.nan)
    )
    pc2 = (
        pca_scores[:, 1] if pca_scores.shape[1] >= 2 else np.full(len(df_wide), np.nan)
    )

    out = pd.DataFrame(
        index=df_wide.index,
        data={
            "PC1": pc1,
            "PC2": pc2,
            "PC1_var_ratio": np.full(len(df_wide), var1),
            "PC2_var_ratio": np.full(len(df_wide), var2),
        },
    )
    return out

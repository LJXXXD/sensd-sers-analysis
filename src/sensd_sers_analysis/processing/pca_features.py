"""Cohort PCA for exploratory plots, independent of model eligibility."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sensd_sers_analysis.data import get_signals_matrix


def add_pca_features(df_wide: pd.DataFrame, *, n_components: int = 2) -> pd.DataFrame:
    """Return aligned exploratory scores fitted only on complete finite spectra.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Spectral rows with ``rs_*`` intensities and original source indices.
    n_components : int
        Requested components, either 1 or 2.

    Returns
    -------
    pd.DataFrame
        PC1/PC2 and explained-variance ratios, aligned to every input row. Rows
        with incomplete spectra have NaN scores. Fewer than two complete rows
        leaves all scores unavailable. Constant spectra have zero scores and
        undefined variance ratios. This full-cohort transformation is exploratory
        and must not supply predictors to held-out model evaluation.
    """
    if n_components not in (1, 2):
        raise ValueError("Exploratory PCA supports one or two components.")
    out = pd.DataFrame(
        np.nan, index=df_wide.index, columns=["PC1", "PC2", "PC1_var_ratio", "PC2_var_ratio"]
    )
    signals = get_signals_matrix(df_wide)
    if not signals.size or signals.shape[1] < 2:
        return out
    complete = np.isfinite(signals).all(axis=1)
    if complete.sum() < 2:
        return out
    scaled = StandardScaler().fit_transform(signals[complete])
    n_eff = min(n_components, scaled.shape[0], scaled.shape[1])
    if not np.any(scaled):
        scores = np.zeros((complete.sum(), n_eff))
        ratios = np.full(n_eff, np.nan)
    else:
        pca = PCA(n_components=n_eff, svd_solver="full")
        scores = pca.fit_transform(scaled)
        ratios = pca.explained_variance_ratio_
    for component in range(n_eff):
        out.loc[complete, f"PC{component + 1}"] = scores[:, component]
        out.loc[complete, f"PC{component + 1}_var_ratio"] = ratios[component]
    return out

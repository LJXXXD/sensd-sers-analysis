"""
Concentration regression paradigms (global, two-stage, multi-task).
"""

from sensd_sers_analysis.regression.data_prep import prepare_concentration_regression_data
from sensd_sers_analysis.regression.metrics import regression_metrics
from sensd_sers_analysis.regression.models_global import (
    SingleRegressorResult,
    train_global_regressors,
)
from sensd_sers_analysis.regression.models_mtl import (
    MtlRegressionOutputs,
    MtlSpectralNet,
    encode_serotype_class_ids,
    serotype_class_labels_from_column,
    train_mtl_regressor,
)
from sensd_sers_analysis.regression.models_two_stage import (
    TwoStageRegressionOutputs,
    train_two_stage_regressors,
)
from sensd_sers_analysis.regression.plots import (
    plot_actual_vs_predicted,
    plot_regression_feature_importance,
    plot_residuals,
)
from sensd_sers_analysis.regression.splits import (
    assert_disjoint_group_split,
    group_train_test_indices,
)

__all__ = [
    "encode_serotype_class_ids",
    "MtlRegressionOutputs",
    "MtlSpectralNet",
    "serotype_class_labels_from_column",
    "SingleRegressorResult",
    "TwoStageRegressionOutputs",
    "assert_disjoint_group_split",
    "group_train_test_indices",
    "plot_actual_vs_predicted",
    "plot_regression_feature_importance",
    "plot_residuals",
    "prepare_concentration_regression_data",
    "regression_metrics",
    "train_global_regressors",
    "train_mtl_regressor",
    "train_two_stage_regressors",
]

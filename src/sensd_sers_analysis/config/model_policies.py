"""
Shared analysis-policy constants.

These constants preserve the current analytical behavior while centralizing
cross-module policy values that were previously duplicated in multiple files.
"""

GLOBAL_QA_REJECTION_MULTIPLIER = 2.0
GLOBAL_QA_R2_MIN_THRESHOLD = 0.80
GLOBAL_QA_IQR_WHIS = 1.5

BATCH_DEVIATION_Z_THRESHOLD = 2.0

PHASE2_INLIER_FEATURE = "integral_area"
PHASE2_QA_FEATURES = (PHASE2_INLIER_FEATURE,)
PHASE2_TEST_SIZE = 0.2
PHASE2_RANDOM_STATE = 42
PHASE2_RF_N_ESTIMATORS = 100

__all__ = [
    "BATCH_DEVIATION_Z_THRESHOLD",
    "GLOBAL_QA_IQR_WHIS",
    "GLOBAL_QA_R2_MIN_THRESHOLD",
    "GLOBAL_QA_REJECTION_MULTIPLIER",
    "PHASE2_INLIER_FEATURE",
    "PHASE2_QA_FEATURES",
    "PHASE2_RANDOM_STATE",
    "PHASE2_RF_N_ESTIMATORS",
    "PHASE2_TEST_SIZE",
]

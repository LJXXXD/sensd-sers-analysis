"""
Spectral preprocessing policy constants.
"""

# Tolerance when merging nearly-coincident Raman-shift columns on one spectrum
# before master-grid snapping (cm⁻¹ scale).
SNAP_SHIFT_DEDUPE_RTOL: float = 1e-9
SNAP_SHIFT_DEDUPE_ATOL: float = 1e-5

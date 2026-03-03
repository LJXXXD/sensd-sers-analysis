"""
Parsing utilities for user input and numeric values.
"""


def parse_raman_shift_bound(s: str) -> float | None:
    """
    Parse user input to float; return None if blank or invalid.

    Args:
        s: String input (e.g., from a text input field).

    Returns:
        Parsed float or None if blank, invalid, or non-numeric.
    """
    if not s or not str(s).strip():
        return None
    try:
        return float(str(s).strip())
    except ValueError:
        return None

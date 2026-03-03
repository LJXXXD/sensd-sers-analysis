"""Natural (human-friendly) sorting for text with embedded numbers.

Sorts strings the way people expect: "item2" before "item10", "0 CFU" before
"1 CFU" before "10 CFU". Pure text (no digits) sorts after strings that contain
numbers (e.g., "Unknown" after "1000 CFU"). Works for labels, filenames, etc.
"""

import re
from typing import Callable, List, Optional


def natural_sort_key(text: str) -> tuple:
    """Build a key for natural sorting of strings with embedded numbers.

    Strings containing digits sort before pure text. Within each group, numeric
    parts are ordered numerically and text lexicographically.

    Examples:
        - "file10" is ordered after "file2".
        - "0 CFU" is ordered before "1 CFU" before "10 CFU".
        - "1000 CFU" is ordered before "Unknown" (text-only goes last).

    Args:
        text: Input string.

    Returns:
        Tuple (has_digit, parts) used by ``sorted``.
    """
    has_digit = bool(re.search(r"\d", text))

    def convert(part: str) -> tuple:
        if part.isdigit():
            return (0, int(part))
        return (1, part.lower())

    parts = [convert(c) for c in re.split(r"([0-9]+)", text) if c]
    return (0 if has_digit else 1, parts)


def natural_sort(
    items: List[str],
    key_func: Optional[Callable[[str], tuple]] = None,
) -> List[str]:
    """Sort strings using natural (human-friendly) order.

    Examples:
        - ["item10", "item2", "item1"] -> ["item1", "item2", "item10"].
        - ["10 CFU", "1 CFU", "0 CFU"] -> ["0 CFU", "1 CFU", "10 CFU"].

    Args:
        items: Strings to sort.
        key_func: Optional custom key; defaults to :func:`natural_sort_key`.

    Returns:
        New list sorted in natural order.
    """
    if not items:
        return items
    key = key_func if key_func is not None else natural_sort_key
    return sorted(items, key=key)


def order_concentration_labels(values: List[str]) -> List[str]:
    """Order concentration_group labels in natural order.

    Strings with numbers (e.g., "0 CFU", "10 CFU") sort first; pure text
    (e.g., "Unknown") sorts last. Convenience alias for natural_sort.

    Args:
        values: Labels such as "0 CFU", "1 CFU", "10 CFU", "Unknown".

    Returns:
        Same values sorted in natural order.
    """
    return natural_sort(values)

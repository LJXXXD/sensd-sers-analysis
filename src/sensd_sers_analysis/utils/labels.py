"""String formatting utilities for column names and display labels."""

# Acronyms to display in all caps when they appear as whole words.
_LABEL_ACRONYMS = ("id", "cfu", "uuid", "url")


def format_column_label(col: str) -> str:
    """
    Convert a column name to a human-readable display label.

    Example: "concentration_group" -> "Concentration Group".
    Example: "sensor_id" -> "Sensor ID" (acronyms like id kept in caps).

    Args:
        col: Snake_case or similar column name.

    Returns:
        Title-style label with underscores replaced by spaces.
    """
    label = col.replace("_", " ").title()
    for acronym in _LABEL_ACRONYMS:
        # Replace whole-word occurrences (e.g., "Id" -> "ID")
        cap = acronym.upper()
        for variant in (acronym.title(), acronym.capitalize()):
            if variant != cap:
                label = label.replace(f" {variant} ", f" {cap} ")
                label = label.replace(f" {variant}", f" {cap}")
                label = label.replace(f"{variant} ", f"{cap} ")
                if label == variant:
                    label = cap
    return label

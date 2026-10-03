"""Expected absence of assessable plot data, distinct from implementation failures."""


class PlotUnavailableError(ValueError):
    """A requested diagnostic has no assessable measurements in the selected scope."""

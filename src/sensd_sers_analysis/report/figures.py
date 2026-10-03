"""Ownership of figures created for a single report request."""

from contextlib import ExitStack

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def own_figure(stack: ExitStack, figure: Figure) -> Figure:
    """Release this request's figure on success or failure; preserve other figures."""

    def release() -> None:
        plt.close(figure)
        figure.clear()

    stack.callback(release)
    return figure

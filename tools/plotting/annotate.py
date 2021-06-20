from functools import singledispatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..outliers import tukey_fences


def add_tukey_marks(
    data: pd.Series,
    ax: plt.Axes,
    annot: bool = True,
    iqr_color: str = "r",
    fence_color: str = "k",
    fence_style: str = "--",
    annot_quarts: bool = False,
) -> plt.Axes:
    """Add IQR box and fences to a histogram-like plot.

    Args:
        data (pd.Series): Data for calculating IQR and fences.
        ax (plt.Axes): Axes to annotate.
        iqr_color (str, optional): Color of shaded IQR box. Defaults to "r".
        fence_color (str, optional): Fence line color. Defaults to "k".
        fence_style (str, optional): Fence line style. Defaults to "--".
        annot_quarts (bool, optional): Annotate Q1 and Q3. Defaults to False.

    Returns:
        plt.Axes: Annotated Axes object.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    ax.axvspan(q1, q3, color=iqr_color, alpha=0.2)
    iqr_mp = q1 + ((q3 - q1) / 2)
    lower, upper = tukey_fences(data)
    ax.axvline(lower, c=fence_color, ls=fence_style)
    ax.axvline(upper, c=fence_color, ls=fence_style)
    text_yval = ax.get_ylim()[1]
    text_yval *= 1.01
    if annot:
        ax.text(iqr_mp, text_yval, "IQR", ha="center")
        if annot_quarts:
            ax.text(q1, text_yval, "Q1", ha="center")
            ax.text(q3, text_yval, "Q3", ha="center")
        ax.text(upper, text_yval, "Fence", ha="center")
        ax.text(lower, text_yval, "Fence", ha="center")
    return ax


@singledispatch
def annot_bars(
    ax: plt.Axes,
    dist: float = 0.15,
    color: str = "k",
    compact: bool = False,
    orient: str = "h",
    format_spec: str = "{x:.2f}",
    fontsize: int = 12,
    alpha: float = 0.5,
    drop_last: int = 0,
    **kwargs,
) -> None:
    """Annotate a bar graph with the bar values.

    Parameters
    ----------
    ax : Axes
        Axes object to annotate.
    dist : float, optional
        Distance from ends as fraction of max bar. Defaults to 0.15.
    color : str, optional
        Text color. Defaults to "k".
    compact : bool, optional
        Annotate inside the bars. Defaults to False.
    orient : str, optional
        Bar orientation. Defaults to "h".
    format_spec : str, optional
        Format string for annotations. Defaults to "{x:.2f}".
    fontsize : int, optional
        Font size. Defaults to 12.
    alpha : float, optional
        Opacity of text. Defaults to 0.5.
    drop_last : int, optional
        Number of bars to ignore on tail end. Defaults to 0.
    """
    if not compact:
        dist = -dist

    xb = np.array(ax.get_xbound()) * (1 + abs(2 * dist))
    ax.set_xbound(*xb)

    max_bar = np.abs([b.get_width() for b in ax.patches]).max()
    dist = dist * max_bar
    for bar in ax.patches[: -drop_last or len(ax.patches)]:
        if orient.lower() == "h":
            x = bar.get_width()
            x = x + dist if x < 0 else x - dist
            y = bar.get_y() + bar.get_height() / 2
        elif orient.lower() == "v":
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            y = y + dist if y < 0 else y - dist
        else:
            raise ValueError("`orient` must be 'h' or 'v'")

        text = format_spec.format(x=bar.get_width())
        ax.annotate(
            text,
            (x, y),
            ha="center",
            va="center",
            c=color,
            fontsize=fontsize,
            alpha=alpha,
            **kwargs,
        )


@annot_bars.register
def _(
    ax: np.ndarray,
    dist: float = 0.15,
    color: str = "k",
    compact: bool = False,
    orient: str = "h",
    format_spec: str = "{x:.2f}",
    fontsize: int = 12,
    alpha: float = 0.5,
    drop_last: int = 0,
    **kwargs,
) -> None:
    """Dispatch for ndarrays"""
    params = locals()
    for ax in params.pop("ax").flat:
        annot_bars(ax, **params)

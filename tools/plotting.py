from functools import partial, singledispatch
from types import MappingProxyType
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wordcloud as wc
from matplotlib import ticker
from sklearn.preprocessing import minmax_scale

from . import outliers, utils

# Default style settings for heatmaps
HEATMAP_STYLE = MappingProxyType(
    {
        "square": True,
        "annot": True,
        "fmt": ".2f",
        "cbar": False,
        "center": 0,
        "cmap": sns.color_palette("coolwarm", n_colors=100, desat=0.6),
        "linewidths": 0.1,
        "linecolor": "k",
        "annot_kws": MappingProxyType({"fontsize": 10}),
    }
)

# Matplotlib rcParams to (optionally) set
MPL_DEFAULTS = MappingProxyType({"axes.labelpad": 10, "axes.titlepad": 5})

_rng = np.random.default_rng(31)


def _format_big_number(num, dec):
    """Format large number using abreviations like 'K' and 'M'."""
    abb = ""
    if num != 0:
        mag = np.log10(np.abs(num))
        if mag >= 12:
            num = num / 10 ** 12
            abb = "T"
        elif mag >= 9:
            num = num / 10 ** 9
            abb = "B"
        elif mag >= 6:
            num = num / 10 ** 6
            abb = "M"
        elif mag >= 3:
            num = num / 10 ** 3
            abb = "K"
        num = round(num, dec)
    return f"{num:,.{dec}f}{abb}"


def big_number_formatter(dec: int = 0) -> ticker.FuncFormatter:
    """Formatter for large numbers; uses abbreviations like 'K' and 'M'.

    Parameters
    ----------
    dec : int, optional
        Decimal precision, by default 0.

    Returns
    -------
    FuncFormatter
        Tick formatter.
    """

    @ticker.FuncFormatter
    def formatter(num, pos):
        return _format_big_number(num, dec)

    return formatter


def big_money_formatter(dec: int = 0) -> ticker.FuncFormatter:
    """Formatter for large monetary numbers; uses abbreviations like 'K' and 'M'.

    Parameters
    ----------
    dec : int, optional
        Decimal precision, by default 0.

    Returns
    -------
    FuncFormatter
        Tick formatter.
    """

    @ticker.FuncFormatter
    def formatter(num, pos):
        return f"${_format_big_number(num, dec)}"

    return formatter


def figsize_like(data: pd.DataFrame, scale: float = 0.85) -> np.ndarray:
    """Calculate figure size based on the shape of data.

    Args:
        data (pd.DataFrame): Ndarray, Series, or Dataframe for figsize.
        scale (float, optional): Scale multiplier for figsize. Defaults to 0.85.

    Returns:
        [np.ndarray]: array([width, height]).
    """
    return np.array(data.shape)[::-1] * scale


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
    lower, upper = outliers.tukey_fences(data)
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
def rotate_ticks(ax: plt.Axes, deg: float, axis: str = "x"):
    """Rotate ticks on `axis` by `deg`.

    Parameters
    ----------
    ax : Axes or ndarray of Axes
        Axes object or objects to rotate ticks on.
    deg : float
        Degree of rotation.
    axis : str, optional
        Axis on which to rotate ticks, 'x' (default) or 'y'.
    """
    get_labels = getattr(ax, f"get_{axis}ticklabels")
    for label in get_labels():
        label.set_rotation(deg)


@rotate_ticks.register
def _(ax: np.ndarray, deg: float, axis: str = "x"):
    """Process ndarrays"""
    axs = ax
    for ax in axs:
        rotate_ticks(ax, deg=deg, axis=axis)


def map_ticklabels(ax: plt.Axes, mapper: Callable, axis: str = "x") -> None:
    """Apply callable to tick labels.

    Parameters
    ----------
    ax : Axes
        Axes object to apply function on.
    mapper : Callable
        Callable to apply to tick labels.
    axis : str, optional
        Axis on which to apply callable, 'x' (default) or 'y'.
    """
    axis = getattr(ax, f"{axis}axis")
    labels = [x.get_text() for x in axis.get_ticklabels()]
    labels = list(map(mapper, labels))
    axis.set_ticklabels(labels)


def pair_corr_heatmap(
    *,
    data: pd.DataFrame,
    ignore: Union[str, list] = None,
    annot: bool = True,
    high_corr: float = None,
    scale: float = 0.5,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a heatmap of the pairwise correlations in `data`.

    Parameters
    ----------
    data : DataFrame
        Data for pairwise correlations.
    ignore : str or list, optional
        Column or columns to ignore, by default None.
    annot : bool, optional
        Whether to annotate cells, by default True.
    high_corr : float, optional
        Threshold for high correlations, by default None. Causes cells
        to be colored in all-or-nothing fashion.
    scale : float, optional
        Scale multiplier for figure size, by default 0.5.
    ax : Axes, optional
        Axes to plot on, by default None.

    Returns
    -------
    Axes
        The heatmap.
    """
    if not ignore:
        ignore = []
    corr_df = data.drop(columns=ignore).corr()
    title = "Correlations Between Features"
    if ax is None:
        figsize = figsize_like(corr_df, scale)
        fig, ax = plt.subplots(figsize=figsize)
    if high_corr is not None:
        if annot:
            annot = corr_df.values
        corr_df = corr_df.abs() > high_corr
        kwargs["center"] = None
        title = f"High {title}"
    mask = np.triu(np.ones_like(corr_df, dtype="int64"), k=0)
    style = dict(HEATMAP_STYLE)
    style.update(kwargs)
    style.update({"annot": annot})
    ax = sns.heatmap(
        data=corr_df,
        mask=mask,
        ax=ax,
        **style,
    )
    ax.set_title(title, pad=10)
    return ax


def calc_subplots_size(nplots: int, ncols: int, height: int) -> tuple:
    """Calculate number of rows and figsize for subplots.

    Parameters
    ----------
    nplots : int
        Number of subplots.
    ncols : int
        Number of columns in figure.
    height : int
        Height of each subplot.

    Returns
    -------
    nrows: int
        Number of rows in figure.
    figsize: tuple
        Width and height of figure.

    """
    raise DeprecationWarning("Deprecated. Use `smart_subplots` instead.")
    nrows = int(np.ceil(nplots / ncols))
    figsize = (ncols * height, nrows * height)
    return nrows, figsize


def smart_subplots(
    *,
    nplots: int,
    size: Tuple[float, float],
    ncols: int = None,
    nrows: int = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Wrapper for `plt.subplots` which calculates the specifications.

    Parameters
    ----------
    nplots : int
        Number of subplots.
    size : Tuple[float, float]
        Size of each subplot in format (width, height).
    ncols : int, optional
        Number of columns in figure. Derived from `nplots` and `nrows` if not specified.
    nrows : int, optional
        Number of rows in figure. Derived from `nplots` and `ncols` if not specified.
    **kwargs
        Keyword arguments passed to `plt.subplots`.
    Returns
    -------
    fig: Figure
        Figure for the plot.
    axs: Axes or array of Axes
        Axes for the plot.
    """
    if ncols and not nrows:
        nrows = int(np.ceil(nplots / ncols))
    elif nrows and not ncols:
        ncols = int(np.ceil(nplots / nrows))
    elif not (nrows or ncols):
        raise ValueError("Must pass either `ncols` or `nrows`")

    figsize = (ncols * size[0], nrows * size[1])
    kwargs.update(nrows=nrows, ncols=ncols, figsize=figsize)
    fig, axs = plt.subplots(**kwargs)
    return fig, axs


def set_invisible(axs: np.ndarray) -> None:
    """Sets all axes to invisible."""
    for ax in axs.flat:
        ax.set_visible(False)


def flip_axis(ax: plt.Axes, axis: str = "x") -> None:
    """Flip axis so it extends in the opposite direction.

    Parameters
    ----------
    ax : Axes
        Axes object with axis to flip.
    axis : str, optional
        Which axis to flip, by default "x".
    """
    if axis.lower() == "x":
        ax.set_xlim(reversed(ax.get_xlim()))
    elif axis.lower() == "y":
        ax.set_ylim(reversed(ax.get_ylim()))
    else:
        raise ValueError("`axis` must be 'x' or 'y'")


def mirror_plot(
    *,
    data: pd.DataFrame,
    x: str,
    y: str,
    left_estimator: Callable = np.sum,
    right_estimator: Callable = np.mean,
    sort_side="right",
    sort_dir="desc",
    size: Tuple[float, float] = (4, 8),
    **kwargs,
) -> plt.Figure:
    """Plot two horizontal bar graphs aligned back-to-back on the vertical axis.

    Parameters
    ----------
    data : DataFrame
        Data for plotting.
    x : str
        Variable for horizontal axis.
    y : str
        Variable for vertical axis.
    left_estimator : Callable, optional
        Estimator for left graph, by default np.sum.
    right_estimator : Callable, optional
        Estimator for right graph, by default np.mean.
    sort_side : str, optional
        Side to sort on, 'left' or 'right' (default).
    sort_dir : str, optional
        Sort direction, 'asc' or 'desc' (default).
    size : Tuple[float, float], optional
        Size of each subplot, by default (4, 8).

    Returns
    -------
    Figure
        Figure for plot.
    """
    if sort_side.lower() not in {"right", "left"}:
        raise ValueError("`sort_side` must be 'right' or 'left'")
    sort_est = left_estimator if sort_side.lower() == "left" else right_estimator
    order = data.groupby(y)[x].agg(sort_est).sort_values().index.to_numpy()
    if sort_dir.lower() == "desc":
        order = order[::-1]

    palette = cat_palette("deep", data.loc[:, y])
    barplot = partial(
        sns.barplot, data=data, y=y, x=x, order=order, palette=palette, **kwargs
    )
    fig, (ax1, ax2) = smart_subplots(nplots=2, size=size, ncols=2, sharey=True)
    barplot(ax=ax1, estimator=left_estimator)
    barplot(ax=ax2, estimator=right_estimator)

    ax1.set_ylabel(None)
    ax2.set_ylabel(None)
    flip_axis(ax1)
    fig.tight_layout()
    return fig


def grouper_plot(
    *,
    data: pd.DataFrame,
    grouper: str = None,
    x: str = None,
    y: str = None,
    kind: str = "line",
    ncols: int = 3,
    height: int = 4,
    **kwargs,
) -> plt.Figure:
    """Plot data by group---one subplot per group.

    Parameters
    ----------
    data : DataFrame
        Data to plot, by default None.
    grouper : str, optional
        Column to group by, by default None.
    x : str, optional
        Variable for x-axis, by default None.
    y : str, optional
        Variable for y-axis, by default None.
    kind : str, optional
        Kind of plot for Dataframe.plot().
        Options: 'line' (default), 'bar', 'barh', 'hist', 'box',
        'kde', 'density', 'area', 'pie', 'scatter', 'hexbin'.
    ncols : int, optional
        Number of subplot columns, by default 3
    height : int, optional
        Height of square subplots, by default 4

    Returns
    -------
    Figure
        The figure.
    """
    data.sort_values(x, inplace=True)
    grouped = data.groupby(grouper)
    fig, axs = smart_subplots(nplots=len(grouped), ncols=ncols, size=(height, height))
    set_invisible(axs)

    for ax, (label, group) in zip(axs.flat, grouped):
        group.plot(x=x, y=y, ax=ax, kind=kind, **kwargs)
        ax.set_title(label)
        ax.set_visible(True)

    fig.tight_layout()
    return fig


def multi_rel(
    *,
    data: pd.DataFrame,
    x: Union[str, list],
    y: str,
    kind="line",
    ncols: int = 3,
    size: Tuple[float, float] = (5.0, 5.0),
    sharey: bool = True,
    **kwargs,
) -> plt.Figure:
    """Plot each `x`-value against `y` on line graphs.
    Parameters
    ----------
    data : DataFrame
        Data with distributions to plot.
    x : str or list-like of str
        Dependent variable(s).
    y : str
        Independent variable.
    kind : str, optional
        Kind of plot: 'line' (default), 'scatter', 'reg', 'bar'.
    ncols : int, optional
        Number of columns for subplots, defaults to 3.
    size : Tuple[float, float], optional.
        Size of each subpot, by default (5.0, 5.0).
    sharey: bool, optional
        Share the y axis between subplots. Defaults to True.
    Returns
    -------
    Figure
        Multiple relational plots.
    """
    fig, axs = smart_subplots(
        nplots=data.columns.size,
        ncols=ncols,
        size=size,
        sharey=sharey,
    )
    set_invisible(axs)
    kinds = dict(
        line=sns.lineplot, scatter=sns.scatterplot, reg=sns.regplot, bar=sns.barplot
    )
    plot = kinds[kind.lower()]

    for ax, column in zip(axs.flat, x):
        ax.set_visible(True)
        ax = plot(data=data, x=column, y=y, ax=ax, **kwargs)

    if axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.set_ylabel(None)
    elif axs.size > 1:
        for ax in axs[1:]:
            ax.set_ylabel(None)
    fig.tight_layout()
    return fig


def multi_dist(
    *,
    data: pd.DataFrame,
    tukey_marks: bool = False,
    ncols: int = 3,
    height: int = 5,
    **kwargs,
) -> plt.Figure:
    """Plot histograms for all numeric variables in `data`.

    Parameters
    ----------
    data : DataFrame
        Data with distributions to plot.
    tukey_marks : bool, optional
        Annotate histograms with IQR and Tukey's fences, by default False.
    ncols : int, optional
        Number of columns for subplots, by default 3.
    height : int, optional
        Subpot height, by default 5.

    Returns
    -------
    Figure
        Multiple histograms.
    """
    data = data.select_dtypes("number")
    fig, axs = smart_subplots(
        nplots=data.columns.size,
        ncols=ncols,
        size=(height, height),
    )
    set_invisible(axs)

    for ax, column in zip(axs.flat, data.columns):
        ax.set_visible(True)
        ax = sns.histplot(data=data, x=column, ax=ax, **kwargs)
        if tukey_marks:
            add_tukey_marks(data[column], ax, annot=False)
        ax.set_title(f"Distribution of `{column}`")

    if axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.set_ylabel(None)
    elif axs.size > 1:
        for ax in axs[1:]:
            ax.set_ylabel(None)
    fig.tight_layout()
    return fig


def countplot(
    *,
    data: pd.Series,
    normalize: bool = False,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    orient: str = "h",
    sort: str = "desc",
    figsize: Tuple[float, float] = (5, 5),
    annot: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot value counts of every feature in `data`.

    Parameters
    ----------
    data : Series
        Data to plot.
    normalize : bool, optional
        Show fractions instead of counts, by default False.
    heat : str, optional
        Color palette for heat, by default "coolwarm".
    heat_desat : float, optional
        Saturation of heat color palette, by default 0.6.
    ncols : int, optional
        Number of columns for subplots, by default 3.
    figsize : int, optional
        Figure size, defaults to (5, 5). Ignored if `ax` is passed.
    orient : str, optional
        Bar orientation, by default "h".
    sort : str, optional
        Direction for sorting bars. Can be 'asc' or 'desc' (default).
    annot : bool, optional
        Annotate bars, True by default.
    ax : Axes, optional
        Axes to plot on.

    Returns
    -------
    Figure
        Value count plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sort = sort.lower()

    df = data.value_counts(normalize=normalize).to_frame("Count")
    df.index.name = data.name or "Series"
    df.reset_index(inplace=True)
    pal = heat_palette(df["Count"], heat, desat=heat_desat)
    ax = barplot(
        data=df,
        x=data.name or "Series",
        y="Count",
        ax=ax,
        orient=orient,
        sort=sort,
        palette=pal,
        **kwargs,
    )
    title = f"`{data.name}` Value Counts" if data.name else "Value Counts"
    ax.set(title=title)
    format_spec = "{x:.0%}" if normalize else "{x:,.0f}"
    if annot:
        annot_bars(ax, orient=orient, format_spec=format_spec)

    orient = orient.lower()
    if orient == "h":
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(format_spec))
        ax.set(ylabel=None)
    else:
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(format_spec))
        ax.set(xlabel=None)
    return ax


def multi_countplot(
    *,
    data: pd.DataFrame,
    normalize: bool = False,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    ncols: int = 3,
    height: int = 5,
    orient: str = "h",
    sort: str = "desc",
    annot: bool = True,
    **kwargs,
) -> plt.Figure:
    """Plot value counts of every feature in `data`.

    Parameters
    ----------
    data : DataFrame
        Data for plotting.
    normalize : bool, optional
        Show fractions instead of counts, by default False.
    heat : str, optional
        Color palette for heat, by default "coolwarm".
    heat_desat : float, optional
        Saturation of heat color palette, by default 0.6.
    ncols : int, optional
        Number of columns for subplots, by default 3.
    height : int, optional
        Subplot height, by default 5.
    orient : str, optional
        Bar orientation, by default "h".
    sort : str, optional
        Direction for sorting bars. Can be 'asc' or 'desc' (default).
    annot : bool, optional
        Annotate bars, True by default.

    Returns
    -------
    Figure
        Multiple value count plots.
    """
    fig, axs = smart_subplots(
        nplots=data.columns.size,
        ncols=ncols,
        size=(height, height),
    )
    sort = sort.lower()
    format_spec = "{x:.0%}" if normalize else "{x:,.0f}"
    data = data.loc[:, data.nunique().sort_values(ascending=(sort == "asc")).index]
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, column in zip(axs.flat, data.columns):
        ax.set_visible(True)
        col_df = data[column].value_counts(normalize=normalize).to_frame("Count")
        col_df.index.name = column
        col_df.reset_index(inplace=True)
        pal = heat_palette(col_df["Count"], heat, desat=heat_desat)
        ax = barplot(
            data=col_df,
            x=column,
            y="Count",
            ax=ax,
            orient=orient,
            sort=sort,
            palette=pal,
            **kwargs,
        )
        ax.set_title(f"`{column}` Value Counts")
        if annot:
            annot_bars(ax, orient=orient, format_spec=format_spec)
        count_axis = ax.xaxis if orient.lower() == "h" else ax.yaxis
        count_axis.set_major_formatter(ticker.StrMethodFormatter(format_spec))
    if axs.size > 1:
        for ax in axs:
            ax.set_ylabel(None)
    fig.tight_layout()
    return fig


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
        Format string for annotations. Defaults to ".2f".
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
    """Process ndarrays"""
    params = locals()
    for ax in params.pop(ax).flat:
        annot_bars(ax, **params)


def heat_palette(data: pd.Series, palette_name: str, desat: float = 0.6) -> np.ndarray:
    """Return Series of heat-colors corresponding to values in `data`.

    Parameters
    ----------
    data : Series
        Series of numeric values to associate with heat colors.
    palette_name : str
        Name of Seaborn color palette.
    desat : float, optional
        Saturation of Seaborn color palette, by default 0.6.

    Returns
    -------
    ndarray
        Heat colors aligned with `data`.
    """
    heat = pd.Series(
        sns.color_palette(palette_name, desat=desat, n_colors=201),
        index=pd.RangeIndex(-100, 101),
    )
    idx = np.around(minmax_scale(data, feature_range=(-100, 100))).astype(np.int64)
    return heat.loc[idx].to_numpy()


def heated_barplot(
    *,
    data: pd.Series,
    heat: str = "coolwarm",
    heat_desat: float = 0.6,
    figsize: tuple = (6, 8),
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a sharply divided barplot ranking positive and negative values.

    Args:
        data (pd.Series): Data to plot.
        heat (str): Name of color palette to be passed to Seaborn.
        heat_desat (float, optional): Saturation of color palette. Defaults to 0.6.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.

    Returns:
        plt.Axes: Axes for the plot.
    """
    data.index = data.index.astype(str)
    data.sort_values(ascending=False, inplace=True)
    palette = heat_palette(data, heat, desat=heat_desat)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.barplot(
        x=data.values, y=data.index, palette=palette, orient="h", ax=ax, **kwargs
    )
    ax.axvline(0.0, color="k", lw=1, ls="-", alpha=0.33)
    return ax


def cat_palette(
    name: str, keys: list, shuffle: bool = False, offset: int = 0, **kwargs
) -> dict:
    """Create a color palette dictionary for a categorical variable.

    Args:
        name (str): Color palette name to be passed to Seaborn.
        keys (list): Keys for mapping to colors.
        shuffle (bool, optional): Shuffle the palette. Defaults to False.
        offset (int, optional): Number of initial colors to skip over. Defaults to 0.

    Returns:
        dict: Categorical-style color mapping.
    """
    n_colors = len(keys) + offset
    pal = sns.color_palette(name, n_colors=n_colors, **kwargs)[offset:]
    if shuffle:
        _rng.shuffle(pal)
    return dict(zip(keys, pal))


def barplot(
    *,
    data: pd.DataFrame,
    x: str,
    y: str,
    sort="asc",
    orient="v",
    estimator: Callable = np.mean,
    figsize: tuple = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a barplot with sorted bars and switchable orientation.

    Parameters
    ----------
    data : DataFrame
        Data for plotting.
    x : str
        Variable for x-axis.
    y : str
        Variable for y-axis.
    sort : str, optional
        Sort direction, by default "asc".
    orient : str, optional
        Bar orientation: 'h' or 'v' (default).
    estimator : Callable, optional
        Estimator for calculating bar heights, by default np.mean.
    figsize : tuple, optional
        Figure size. Defaults to (8, 5) if not specified.
    ax : Axes, optional
        Axes to plot on, by default None.

    Returns
    -------
    Axes
        Barplot.
    """

    if ax is None:
        if figsize is None:
            width, height = (8, 5)
            figsize = (width, height) if orient == "v" else (height, width)
        fig, ax = plt.subplots(figsize=figsize)
    if sort:
        if sort.lower() in ("asc", "desc"):
            asc = sort.lower() == "asc"
        else:
            raise ValueError("`sort` must be 'asc', 'desc', or None")
        order = data.groupby(x)[y].agg(estimator)
        order = order.sort_values(ascending=asc).index.to_list()
    else:
        order = None

    # titles = {
    #     "y": utils.to_title(y),
    #     "x": utils.to_title(x),
    #     "est": utils.to_title(estimator.__name__),
    # }

    if orient.lower() == "h":
        x, y = y, x
    elif orient.lower() != "v":
        raise ValueError("`orient` must be 'v' or 'h'")
    ax = sns.barplot(
        data=data,
        x=x,
        y=y,
        estimator=estimator,
        orient=orient,
        order=order,
        ax=ax,
        **kwargs,
    )

    # ax.set_title("{est} {y} by {x}".format(**titles), pad=10)
    # ax.set_xlabel(titles["x" if orient.lower() == "v" else "y"], labelpad=10)
    # ax.set_ylabel(titles["y" if orient.lower() == "v" else "x"], labelpad=15)
    return ax


def cat_corr_heatmap(
    *,
    data: pd.DataFrame,
    categorical: str,
    transpose: bool = False,
    high_corr: float = None,
    scale: float = 0.5,
    no_prefix: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a correlation heatmap of categorical vs. numeric features.

    Args:
        data (DataFrame): Frame containing categorical and numeric data.
        categorical (str): Name or list of names of categorical features.
        high_corr (float): Threshold for high correlation. Defaults to None.
        scale (float, optional): Multiplier for determining figsize. Defaults to 0.5.
        no_prefix (bool, optional): If only one cat, do not prefix dummies. Defaults to True.
        ax (Axes, optional): Axes to plot on. Defaults to None.

    Returns:
        Axes: Axes of the plot.
    """
    if isinstance(categorical, str):
        ylabel = utils.to_title(categorical)
        categorical = [categorical]
        single_cat = True
    else:
        ylabel = "Categorical Features"
        single_cat = False
    title = "Correlation with Numeric Features"
    cat_df = data.filter(categorical, axis=1)
    if no_prefix and single_cat:
        dummies = pd.get_dummies(cat_df, prefix="", prefix_sep="")
    else:
        dummies = pd.get_dummies(cat_df)
    corr_df = dummies.apply(lambda x: data.corrwith(x))
    if not transpose:
        corr_df = corr_df.T
    if high_corr is not None:
        if "annot" not in kwargs or kwargs.get("annot"):
            kwargs["annot"] = corr_df.values
        corr_df = corr_df.abs() > high_corr
        kwargs["center"] = None
        title = f"High {title}"
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize_like(corr_df, scale=scale))
    style = dict(HEATMAP_STYLE)
    style.update(kwargs)
    ax = sns.heatmap(corr_df, ax=ax, **style)
    xlabel = "Numeric Features"
    if transpose:
        xlabel, ylabel = ylabel, xlabel
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title, pad=10)
    return ax

def _validate_orient(orient):
    if orient.lower() not in {"h", "v"}:
        raise ValueError(f"`orient` must be 'h' or 'v', not {orient}")


def emo_wordclouds(emo_vecs, size=(5, 3), orient="h", **kwargs):
    _validate_orient(orient)
    if orient.lower() == "v":
        ncols, nrows = (0, 3)
    else:
        ncols, nrows = (3, 0)

    fig, axs = smart_subplots(nplots=3, size=size, ncols=ncols, nrows=nrows)

    if emo_vecs.shape[0] == 3 and emo_vecs.shape[1] > 3:
        emo_vecs = emo_vecs.T
    if emo_vecs.shape[1] != 3:
        raise ValueError("Expected `emo_vecs` to have exactly 3 columns")

    emo_vecs.columns = emo_vecs.columns.str.lower()
    color_funcs = {
        "neutral": wc.get_single_color_func("gray"),
        "positive": wc.get_single_color_func("green"),
        "negative": wc.get_single_color_func("red"),
    }

    for ax, column in zip(axs.flat, emo_vecs.columns):
        width, height = np.array(size) * 100
        cloud = wc.WordCloud(
            color_func=color_funcs[column], width=width, height=height, **kwargs
        )
        cloud = cloud.generate_from_frequencies(emo_vecs["neutral"])
        ax.imshow(cloud.to_image(), interpolation="bilinear", aspect="equal")

        # Hide grid lines
        ax.grid(False)

        # Hide ticks
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig
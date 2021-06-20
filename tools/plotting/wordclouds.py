from functools import singledispatch
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import seaborn as sns
import wordcloud as wc
import matplotlib.pyplot as plt
from .._validation import _validate_orient
from .utils import smart_subplots


@singledispatch
def wordcloud(
    word_scores: pd.Series,
    *,
    cmap: str = "Greys",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    ax: plt.Axes = None,
    **kwargs
):
    if ax is None:
        _, ax = plt.subplots(figsize=size)

    width, height = np.array(size) * 100
    cloud = wc.WordCloud(
        colormap=cmap,
        width=width,
        height=height,
        **kwargs,
    )
    cloud = cloud.generate_from_frequencies(word_scores)
    ax.imshow(cloud.to_image(), interpolation="bilinear", aspect="equal")

    if word_scores.name is not None:
        ax.set(title=word_scores.name)

    # Hide grid lines and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


@wordcloud.register
def _(
    word_scores: pd.DataFrame,
    *,
    cmap: Union[str, List, Dict] = "Greys",
    size: Tuple[float, float] = (5, 3),
    ncols: int = 3,
    ax: plt.Axes = None,
    **kwargs
):
    if ax is not None:
        raise ValueError("`ax` not supported for DataFrame input")
    fig, axs = smart_subplots(nplots=word_scores.shape[1], size=size, ncols=ncols)

    if isinstance(cmap, str):
        cmap = dict.fromkeys(word_scores.columns, cmap)
    elif isinstance(cmap, list):
        cmap = dict(zip(word_scores.columns, cmap))
    elif not isinstance(cmap, dict):
        raise TypeError("`cmap` must be str, list, or dict {cols -> cmaps}")

    for ax, column in zip(axs.flat, word_scores.columns):
        wordcloud(
            word_scores.loc[:, column],
            cmap=cmap[column],
            size=size,
            ncols=ncols,
            ax=ax,
            **kwargs,
        )
    fig.tight_layout()
    return fig
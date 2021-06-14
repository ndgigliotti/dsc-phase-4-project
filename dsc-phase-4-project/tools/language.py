import enum
from functools import partial
from operator import itemgetter
import re
from typing import Union
import fuzzywuzzy
import numpy as np
import pandas as pd
from fuzzywuzzy.process import extractOne as extract_one
from IPython.core.display import Markdown, display
from pandas._typing import ArrayLike


def readable_sample(data: pd.Series, n: int = 10, random_state=None) -> None:
    data = data.sample(n=n, random_state=random_state)
    display(Markdown(data.to_markdown()))


def identify_brands(
    *pats: str,
    docs: pd.Series,
    exclusive: bool = True,
    flags: re.RegexFlag = re.I,
):
    brands = [docs.str.findall(x, flags=flags) for x in pats]

    # Explode the lists
    for i, series in enumerate(brands):
        brands[i] = series.explode().dropna()

    if exclusive:
        # Drop tweets with conflicting brands
        conflicting = set()
        for series in brands:
            other_indices = set().union(
                *[set(x.index) for x in brands if x is not series]
            )
            conflicting.update(other_indices & set(series.index))
        for i, series in enumerate(brands):
            brands[i] = series.drop(conflicting)

    # Combine into one Series
    findings = pd.concat(brands, axis=0).rename("brands")
    # Sort and return
    return findings.sort_index()

def fuzzy_match(data: pd.Series, options:ArrayLike, **kwargs):
    extract_option = partial(extract_one, choices=options, **kwargs)
    scores = data.map(extract_option, "ignore")
    data = data.to_frame("original")
    data["extracted"] = scores.map(itemgetter(0), "ignore")
    data["score"] = scores.map(itemgetter(1), "ignore")
    return data
from typing import Iterable
import numpy as np
import pandas as pd
from pandas.core.dtypes.inference import is_list_like, is_nested_list_like
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def _validate_orient(orient):
    if orient.lower() not in {"h", "v"}:
        raise ValueError(f"`orient` must be 'h' or 'v', not {orient}")


def _validate_sort(sort):
    if sort is None:
        pass
    elif sort.lower() not in {"asc", "desc"}:
        raise ValueError("`sort` must be 'asc', 'desc', or None")


def _check_if_tagged(docs: pd.Series):
    """Check if `docs` are POS tagged."""
    list_like = docs.map(is_list_like)
    nested = docs.map(is_nested_list_like)
    tagged = list_like.all() and nested.any()
    return tagged


def _validate_transformer(obj):
    if obj is None:
        raise ValueError("Transformer is None")
    est = isinstance(obj, BaseEstimator)
    trans = isinstance(obj, TransformerMixin)
    pipe = isinstance(obj, Pipeline)
    if not ((est and trans) or pipe):
        raise TypeError(
            "Transformer must be Sklearn transformer or Pipeline"
        )


def _validate_raw_docs(X: Iterable):
    if isinstance(X, str):
        raise TypeError("Expected iterable over raw documents, string object received.")
    if isinstance(X, np.ndarray) and X.ndim > 1:
        raise ValueError(
            f"Expected iterable over raw documents, received {X.ndim}darray"
        )
    if isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected iterable over raw documents, received DataFrame")

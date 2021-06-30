from typing import Iterable
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from pandas.core.generic import NDFrame
from . import utils


def _validate_orient(orient):
    if orient.lower() not in {"h", "v"}:
        raise ValueError(f"`orient` must be 'h' or 'v', not {orient}")


def _validate_sort(sort):
    if sort is None:
        pass
    elif sort.lower() not in {"asc", "desc"}:
        raise ValueError("`sort` must be 'asc', 'desc', or None")


def _validate_train_test_split(X_train, X_test, y_train, y_test):
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    if X_train.ndim > 1:
        assert X_train.shape[1] == X_test.shape[1]
    if y_train.ndim > 1:
        assert y_train.shape[1] == y_test.shape[1]


def _check_1dlike(data):
    msg = "Data must be shape (n_samples,) or (n_samples, 1)."
    if data.ndim == 2 and data.shape[1] > 1:
        raise ValueError(msg)
    elif data.ndim > 2:
        raise ValueError(msg)


def _validate_transformer(obj):
    if obj is None:
        raise ValueError("Transformer is None")
    est = isinstance(obj, BaseEstimator)
    trans = isinstance(obj, TransformerMixin)
    pipe = isinstance(obj, Pipeline)
    if not ((est and trans) or pipe):
        raise TypeError("Transformer must be Sklearn transformer or Pipeline")


def _validate_raw_docs(X: Iterable[str]):
    """Used for text vectorizers. Makes sure X is iterable over raw documents."""
    if not isinstance(X, Iterable) or isinstance(X, str):
        raise TypeError(
            f"Expected iterable over raw documents, {type(X)} object received."
        )
    if isinstance(X, (np.ndarray, NDFrame)):
        _check_1dlike(X)

    if isinstance(X, np.ndarray) and X.ndim > 1:
        raise ValueError(
            f"Expected iterable over raw documents, received {X.ndim}darray"
        )
    if isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected iterable over raw documents, received DataFrame")


def _validate_docs(docs):
    """Makes sure `docs` is either str or iterable of str."""
    if not isinstance(docs, (str, Iterable)):
        raise TypeError(f"Expected str or iterable of str, got {type(docs)}.")
    elif isinstance(docs, Iterable):
        docs = np.asarray(docs)
        _check_1dlike(docs)
        types = utils.flat_map(type, docs.squeeze())
        if not (types == str).all():
            non_str = types[types != str]
            raise TypeError(
                f"Expected iterable of str, but iterable contains {non_str[0]}."
            )

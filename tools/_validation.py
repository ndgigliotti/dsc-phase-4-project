from typing import Iterable
import numpy as np
import pandas as pd
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


def _validate_train_test_split(X_train, X_test, y_train, y_test):
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    if X_train.ndim > 1:
        assert X_train.shape[1] == X_test.shape[1]
    if y_train.ndim > 1:
        assert y_train.shape[1] == y_test.shape[1]


def _check_array_1dlike(array):
    msg = "Array must be shape (n_samples,) or (n_samples, 1)."
    if array.ndim == 2 and array.shape[1] > 1:
        raise ValueError(msg)
    elif array.ndim > 2:
        raise ValueError(msg)


def _validate_transformer(obj):
    if obj is None:
        raise ValueError("Transformer is None")
    est = isinstance(obj, BaseEstimator)
    trans = isinstance(obj, TransformerMixin)
    pipe = isinstance(obj, Pipeline)
    if not ((est and trans) or pipe):
        raise TypeError("Transformer must be Sklearn transformer or Pipeline")


def _validate_raw_docs(X: Iterable):
    if isinstance(X, str):
        raise TypeError("Expected iterable over raw documents, string object received.")
    if isinstance(X, np.ndarray) and X.ndim > 1:
        raise ValueError(
            f"Expected iterable over raw documents, received {X.ndim}darray"
        )
    if isinstance(X, pd.DataFrame):
        raise TypeError(f"Expected iterable over raw documents, received DataFrame")

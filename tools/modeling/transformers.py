from functools import partial, singledispatch, singledispatchmethod
from typing import Union

import numpy as np
import pandas as pd
from nltk.tokenize.casual import TweetTokenizer as NLTKTweetTokenizer
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import as_float_array
from sklearn.utils.validation import check_is_fitted

# The following partial objects are shorthand callables
# for constructing commonly used estimators.

LogTransformer = partial(
    FunctionTransformer,
    func=np.log,
    inverse_func=np.exp,
)

Log10Transformer = partial(
    FunctionTransformer,
    func=np.log10,
    inverse_func=partial(np.power, 10),
)


class DummyEncoder(BaseEstimator, TransformerMixin):
    """Transformer wrapper for pd.get_dummies."""

    def __init__(
        self,
        prefix=None,
        prefix_sep="_",
        dummy_na=False,
        columns=None,
        sparse=False,
        drop_first=False,
        dtype=np.float64,
    ):
        for key, value in locals().items():
            if key == "self":
                continue
            else:
                setattr(self, key, value)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dummies = pd.get_dummies(X, **self.get_params())
        self.feature_names_ = dummies.columns.to_numpy()
        return dummies


class ArrayForcer(BaseEstimator, TransformerMixin):
    def __init__(self, force_all_finite=True, force_dense=False) -> None:
        self.force_all_finite = force_all_finite
        self.force_dense = force_dense
        super().__init__()

    @property
    def feature_names_(self):
        check_is_fitted(self)
        return self.columns_.to_numpy() if self.columns_ is not None else None

    @singledispatchmethod
    def fit(self, X: np.ndarray, y=None):
        self.columns_ = None
        self.index_ = None
        self.dtypes_ = X.dtype
        self.input_type_ = np.ndarray
        self.input_shape_ = X.shape
        return self

    @fit.register
    def _(self, X: csr_matrix, y=None):
        self.columns_ = None
        self.index_ = None
        self.dtypes_ = X.dtype
        self.input_type_ = csr_matrix
        self.input_shape_ = X.shape
        return self

    @fit.register
    def _(self, X: pd.DataFrame, y=None):
        self.columns_ = X.columns
        self.index_ = X.index
        self.dtypes_ = X.dtypes
        self.input_type_ = pd.DataFrame
        self.input_shape_ = X.shape
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = as_float_array(X, force_all_finite=self.force_all_finite)
        if isinstance(X, csr_matrix) and self.force_dense:
            X = X.todense()
        return X

    def inverse_transform(self, X):
        check_is_fitted(self)
        if self.input_type_ == pd.DataFrame:
            if X.shape == (self.index_.size, self.columns_.size):
                result = pd.DataFrame(data=X, index=self.index_, columns=self.columns_)
                for column, dtype in self.dtypes_.items():
                    result[column] = result[column].astype(dtype)
            else:
                raise ValueError(
                    "`X` must be same shape as input if input was DataFrame"
                )
        else:
            result = X.astype(self.dtypes_)
        return result
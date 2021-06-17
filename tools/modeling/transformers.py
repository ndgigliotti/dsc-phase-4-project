from functools import partial, reduce, singledispatchmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import as_float_array
from sklearn.utils.validation import check_is_fitted
from nltk.tokenize.casual import TweetTokenizer as NLTKTweetTokenizer

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


class PandasWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None):
        super().__init__()
        self.transformer = transformer

    def reconst_frame(self, X: np.ndarray):
        X = pd.DataFrame(X, self.index_, self.columns_)
        for column, dtype in self.dtypes_.items():
            X[column] = X[column].astype(dtype)
        return X

    @singledispatchmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.index_ = X.index
        self.columns_ = X.columns
        self.dtypes_ = X.dtypes
        if self.transformer is not None:
            if y is not None:
                y = y.to_numpy()
            self.transformer.fit(X.to_numpy(), y)
        return self

    @fit.register
    def _(self, X: pd.Series, y: pd.Series = None):
        self.fit(X.to_frame(), y)
        return self

    @singledispatchmethod
    def transform(self, X: pd.DataFrame):
        check_is_fitted(self)
        if self.transformer is not None:
            init_shape = X.shape
            X = self.transformer.transform(X.to_numpy())
            if X.shape != init_shape:
                raise RuntimeError("Transformation must preserve shape and order")
            X = self.reconst_frame(X)
        return X

    @transform.register
    def _(self, X: pd.Series):
        return self.transform(X.to_frame()).squeeze()

    @singledispatchmethod
    def inverse_transform(self, X: pd.DataFrame):
        check_is_fitted(self)
        if self.transformer is not None:
            init_shape = X.shape
            X = self.transformer.inverse_transform(X.to_numpy())
            if X.shape != init_shape:
                raise RuntimeError(
                    "Inverse transformation must preserve shape and order"
                )
            X = self.reconst_frame(X)
        return X

    @inverse_transform.register
    def _(self, X: pd.Series):
        return self.inverse_transform(X.to_frame()).squeeze()


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


class FloatArrayForcer(BaseEstimator, TransformerMixin):
    def __init__(self, force_all_finite=True) -> None:
        self.force_all_finite = force_all_finite
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
    def _(self, X: pd.DataFrame, y=None):
        self.columns_ = X.columns
        self.index_ = X.index
        self.dtypes_ = X.dtypes
        self.input_type_ = pd.DataFrame
        self.input_shape_ = X.shape
        return self

    def transform(self, X):
        check_is_fitted(self)
        return as_float_array(X, force_all_finite=self.force_all_finite)

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


class TweetTokenizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        preserve_case=True,
        reduce_len=False,
        strip_handles=False,
        subset=None,
    ):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles
        self.subset = subset

        self.tokenizer = NLTKTweetTokenizer(
            preserve_case=preserve_case,
            reduce_len=reduce_len,
            strip_handles=strip_handles,
        )
        super().__init__()

    def fit(self, X: Union[pd.DataFrame, pd.Series], y=None):
        return self

    @singledispatchmethod
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X[self.subset] = X.loc[:, self.subset].applymap(self.tokenizer.tokenize)
        return X

    @transform.register
    def _(self, X: pd.Series):
        return X.map(self.tokenizer.tokenize)

    @singledispatchmethod
    def inverse_transform(self, X: pd.DataFrame):
        return X.apply(lambda x: x.str.join(" "))

    @inverse_transform.register
    def _(self, X: pd.Series):
        return X.str.join(" ")
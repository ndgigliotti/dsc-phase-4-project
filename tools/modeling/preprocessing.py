from functools import partial, singledispatchmethod

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import as_float_array
from sklearn.utils.validation import check_is_fitted

# The following partial objects are shorthand callables
# for constructing commonly used estimators.

log_transformer = partial(
    FunctionTransformer,
    func=np.log,
    inverse_func=np.exp,
)

log10_transformer = partial(
    FunctionTransformer,
    func=np.log10,
    inverse_func=partial(np.power, 10),
) 


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Simple quantile-based Winsorizer."""

    def __init__(self, inner: float = None) -> None:
        self.inner = inner

    @property
    def limits(self):
        return (1 - np.array([self.inner] * 2)) / 2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = as_float_array(X, force_all_finite="allow-nan")
        return winsorize(X, limits=self.limits, axis=0, nan_policy="propagate")


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


def binary_features(X: np.ndarray, as_mask: bool = False) -> np.array:
    """Returns column indices of binary features.

    Parameters
    ----------
    X : np.ndarray
        Array to get binary feature indices from.
    as_mask : bool, optional
        Return boolean mask instead of indices, by default False.

    Returns
    -------
    np.ndarray
        Flat array of feature indices (or booleans).
    """
    df = pd.DataFrame(X)
    mask = (df.nunique() == 2).to_numpy()
    return mask if as_mask else df.columns[mask].to_numpy()

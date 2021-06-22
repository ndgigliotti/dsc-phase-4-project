from functools import singledispatch, singledispatchmethod
from typing import Any, Union
import numpy as np

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .. import utils
from .._validation import _validate_transformer
from ..language import frame_doc_vecs


# @singledispatch
# def _wrap(obj: BaseEstimator):
#     if obj is None or isinstance(obj, str):
#         pass
#     elif not isinstance(obj, PandasWrapper):
#         obj = PandasWrapper(obj)
#     return obj


# @_wrap.register
# def _(obj: Pipeline):
#     new_pipe = [(x, _wrap(y)) for x, y in obj.steps]
#     return Pipeline(new_pipe, memory=obj.memory, verbose=obj.verbose)


# @_wrap.register
# def _(obj: ColumnTransformer):
#     new_ct = [(x, _wrap(y), z) for x, y, z in obj.transformers_]
#     return ColumnTransformer(
#         new_ct,
#         remainder=obj.remainder,
#         sparse_threshold=obj.sparse_threshold,
#         n_jobs=obj.n_jobs,
#         transformer_weights=obj.transformer_weights,
#         verbose=obj.verbose,
#     )


class PandasWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None):
        super().__init__()
        self.transformer = transformer
        if not self.passthrough:
            _validate_transformer(self.transformer)

    @property
    def is_vectorizer(self):
        if self.transformer is None:
            result = False
        elif isinstance(self.transformer, _VectorizerMixin):
            result = True
        elif "vectorizer" in self.transformer.__class__.__name__.lower():
            result = True
        elif "raw_documents" in utils.get_param_names(self.transformer.fit):
            result = True
        else:
            result = False
        return result

    @property
    def passthrough(self):
        return self.transformer is None or self.transformer == "passthrough"

    def unwrap(self):
        return self.transformer

    def _reconst_frame(self, X: Union[np.ndarray, csr_matrix]):
        if isinstance(X, csr_matrix):
            X = X.todense()
        X = pd.DataFrame(X, self.index_, self.columns_)
        for column, dtype in self.dtypes_.items():
            X[column] = X[column].astype(dtype)
        return X

    def _vocab_frame(self, X: Union[np.ndarray, csr_matrix]):
        if isinstance(X, csr_matrix):
            X = frame_doc_vecs(X, self.transformer.vocabulary_, self.index_)
        else:
            raise NotImplementedError()
        return X

    @singledispatchmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **fit_params):
        if self.is_vectorizer:
            raise TypeError("Cannot pass DataFrame to vectorizer")
        self.index_ = X.index
        self.columns_ = X.columns
        self.dtypes_ = X.dtypes
        if not self.passthrough:
            if y is not None:
                y = y.to_numpy()
            self.transformer.fit(X.to_numpy(), y, **fit_params)
        return self

    @fit.register
    def _(self, X: pd.Series, y: pd.Series = None, **fit_params):
        if self.is_vectorizer and not self.passthrough:
            self.index_ = X.index
            self.transformer.fit(X.to_list(), y, **fit_params)
        else:
            self.fit(X.to_frame(), y, **fit_params)
        return self

    def _transform_preserve_struct(self, X: pd.DataFrame):
        init_shape = X.shape
        X = self.transformer.transform(X.to_numpy())
        if X.shape != init_shape:
            raise RuntimeError("Transformation must preserve shape and order")
        return self._reconst_frame(X)

    def _itransform_preserve_struct(self, X: pd.DataFrame):
        init_shape = X.shape
        X = self.transformer.inverse_transform(X.to_numpy())
        if X.shape != init_shape:
            raise RuntimeError("Inverse transformation must preserve shape and order")
        return self._reconst_frame(X)

    @singledispatchmethod
    def transform(self, X: pd.DataFrame):
        check_is_fitted(self)
        if self.is_vectorizer:
            raise TypeError("Cannot pass DataFrame to vectorizer")
        if not self.passthrough:
            X = self._transform_preserve_struct(X)
        return X

    @transform.register
    def _(self, X: pd.Series):
        check_is_fitted(self)
        if not self.passthrough:
            if self.is_vectorizer:
                X = self.transformer.transform(X.to_list())
                X = self._vocab_frame(X)
            else:
                X = self._transform_preserve_struct(X.to_frame()).squeeze()
        return X

    @singledispatchmethod
    def inverse_transform(self, X: pd.DataFrame):
        check_is_fitted(self)
        if self.is_vectorizer:
            raise NotImplementedError(
                "Inverse transform not implemented for vectorizers"
            )
        if not self.passthrough:
            X = self._itransform_preserve_struct(X)
        return X

    @inverse_transform.register
    def _(self, X: pd.Series):
        check_is_fitted(self)
        if self.is_vectorizer:
            raise NotImplementedError(
                "Inverse transform not implemented for vectorizers"
            )
        if not self.passthrough:
            X = self._itransform_preserve_struct(X.to_frame()).squeeze()
        return X

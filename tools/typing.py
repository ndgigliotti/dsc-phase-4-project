from re import Pattern
from typing import Any, Callable, Iterable, List, Tuple, TypeVar, Union

from numpy import ndarray
from numpy.random import BitGenerator, RandomState
from pandas.core.generic import NDFrame
from pandas.core.series import Series
from scipy.sparse.base import spmatrix
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# RandomState-related types
SeedLike = Union[int, ndarray, BitGenerator, RandomState]
LegacySeedLike = Union[int, ndarray, RandomState]

# String or compiled regex
PatternLike = Union[str, Pattern]

# Series or NDArray
SeriesOrArray = Union[Series, ndarray]
NDStruct = Union[NDFrame, ndarray, spmatrix]
# Estimator or Pipeline
EstimatorLike = Union[BaseEstimator, Pipeline]

# One or more strings
Documents = Union[str, Iterable[str]]

# List of word tokens
TokenList = List[str]

# List of tokens with POS tags
TaggedTokenList = List[Tuple[str, str]]

# Function which takes a string
CallableOnStr = Callable[[str], Any]

# Function which tokenizes a string
Tokenizer = Callable[[str], TokenList]

from re import Pattern
from typing import Any, Callable, Iterable, List, Tuple, TypeVar, Union

import numpy as np
from pandas.core.series import Series
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# RandomState-related types
RandomSeed = Union[int, np.ndarray, np.random.BitGenerator, np.random.RandomState]
LegacyRandomSeed = Union[int, np.ndarray, np.random.RandomState]

# String or compiled regex
PatternLike = Union[str, Pattern]

# Series or NDArray
SeriesOrArray = TypeVar("SeriesOrArray", Series, np.ndarray)

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

from re import Pattern
from typing import List, Sequence, TypeVar, Union
import numpy as np
from pandas._typing import ArrayLike
from pandas.core.series import Series
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# RandomState-related types accepted by Pandas
RandomSeed = Union[int, ArrayLike, np.random.BitGenerator, np.random.RandomState]

# String or compiled regex
StrOrPattern = Union[str, Pattern]

# Series or NDArray
SeriesOrArray = TypeVar("SeriesOrArray", Series, np.ndarray)

# List-like
ListLike = TypeVar("ListLike", Series, np.ndarray, Sequence)

# Estimator or Pipeline
EstimatorLike = Union[BaseEstimator, Pipeline]
from re import Pattern
from typing import List, TypeVar, Union
import numpy as np
from pandas._typing import ArrayLike
from pandas.core.series import Series

# RandomState-related types accepted by Pandas
RandomSeed = Union[int, ArrayLike, np.random.BitGenerator, np.random.RandomState]

# String or compiled regex
StrOrPattern = Union[str, Pattern]

# Series or NDArray
SeriesOrArray = TypeVar("SeriesOrArray", Series, np.ndarray)
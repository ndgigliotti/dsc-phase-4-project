import pandas as pd
from pandas.core.dtypes.inference import is_list_like, is_nested_list_like


def _validate_orient(orient):
    if orient.lower() not in {"h", "v"}:
        raise ValueError(f"`orient` must be 'h' or 'v', not {orient}")


def _validate_sort(sort):
    if sort is None:
        pass
    elif sort.lower() not in {"asc", "desc"}:
        raise ValueError("`sort` must be 'asc', 'desc', or None")


def _check_if_tagged(docs: pd.Series):
    """Check if `docs` are POS tagged."""
    list_like = docs.map(is_list_like)
    nested = docs.map(is_nested_list_like)
    tagged = list_like.all() and nested.any()
    return tagged
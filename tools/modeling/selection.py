import os
import textwrap
from typing import List, Union

import numpy as np
import pandas as pd
from feature_engine.selection import (
    SmartCorrelatedSelection as SmartCorrelatedSelectionFE,
)
from IPython.core.display import HTML
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import joblib
from .. import utils

Variables = Union[None, int, str, List[Union[str, int]]]


def _drop_splits(cv_results: dict) -> pd.DataFrame:
    """Clean up messy grid search results and return DataFrame.

    Converts `cv_results` to DataFrame and removes the numerous
    "splitX_" columns which make the table hard to read.

    Parameters
    ----------
    cv_results : dict
        Dict of results from GridSearchCV or RandomizedSearchCV.

    Returns
    -------
    DataFrame
        Cleaned up results.
    """
    cv_results = pd.DataFrame(cv_results)
    splits = cv_results.filter(regex=r"split[0-9]+_").columns
    cv_results.drop(columns=splits, inplace=True)
    cv_results.sort_values("rank_test_score", inplace=True)
    return cv_results


def grid_search(
    estimator,
    param_grid,
    X,
    y,
    name,
    dirname="grid_search",
    scoring=None,
    n_jobs=None,
    cv=None,
    verbose=1,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
    return_train_score=False,
):
    search = GridSearchCV(
        estimator,
        param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=False,
        cv=cv,
        verbose=verbose,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
        return_train_score=return_train_score,
    )
    search.fit(X, y)
    os.makedirs(dirname, exist_ok=True)
    filename = name if name.endswith(".joblib") else f"{name}.joblib"
    filename = os.path.join(dirname, filename)
    joblib.dump(search, filename)
    return filename

def load_results(path):
    if not path.endswith(".joblib"):
        path = f"{path}.joblib"
    search = joblib.load(os.path.normpath(path))
    return _drop_splits(search.cv_results_)

class SmartCorrelatedSelection(SmartCorrelatedSelectionFE):
    """Wrapper for feature_engine.selection.SmartCorrelatedSelection."""

    def __init__(
        self,
        variables: Variables = None,
        method: str = "pearson",
        threshold: float = 0.8,
        missing_values: str = "ignore",
        selection_method: str = "missing_values",
        estimator=None,
        scoring: str = "roc_auc",
        cv: int = 3,
        verbose: bool = False,
    ):
        super().__init__(
            variables=variables,
            method=method,
            threshold=threshold,
            missing_values=missing_values,
            selection_method=selection_method,
            estimator=estimator,
            scoring=scoring,
            cv=cv,
        )
        self.verbose = verbose

    @property
    def selected_features_(self):
        check_is_fitted(self)
        corr_superset = set().union(*self.correlated_feature_sets_)
        return list(corr_superset.difference(self.features_to_drop_))

    def show_report(self):
        check_is_fitted(self)
        name = self.__class__.__name__
        info = [
            pd.Series(self.selected_features_, name="Selected"),
            pd.Series(self.features_to_drop_, name="Rejected"),
        ]
        info = pd.concat(info, axis=1)
        # info = info.applymap(lambda x: f"'{x}'", na_action="ignore")
        name = self.__class__.__name__
        # info = info.T.to_string(na_rep="", header=False, justify="left", max_cols=6)
        info = info.T.to_html(na_rep="", header=False, max_cols=6, notebook=True)
        info = f"<h4>{name}</h4>{info}"
        display(HTML(info))

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        super().fit(X, y=y)
        if self.verbose:
            self.show_report()
        return self

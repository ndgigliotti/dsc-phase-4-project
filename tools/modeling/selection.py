import textwrap
from typing import List, Union

import numpy as np
import pandas as pd
from feature_engine.selection import \
    SmartCorrelatedSelection as SmartCorrelatedSelectionFE
from IPython.core.display import HTML
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .. import utils

Variables = Union[None, int, str, List[Union[str, int]]]


def tidy_results(cv_results: dict) -> pd.DataFrame:
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
    return cv_results


def make_search_pipe(
    estimator_pipe: Pipeline,
    param_grid: dict,
    kind: str = "grid",
    step_name: str = "search",
    n_jobs: int = None,
    **kwargs,
) -> Pipeline:
    """Construct a parameter search pipeline from another pipeline.

    Creates a pipeline for conducting a parameter search on the
    final estimator of another pipeline. The search pipeline will be
    a clone of the original with the final estimator swapped out
    for a search estimator. The search estimator contains a clone
    of the original final estimator with the search parameters reset.

    Parameters
    ----------
    estimator_pipe : Pipeline
        Pipeline with final estimator to be used in search.
    param_grid : dict
        Parameter grid or distributions for search.
    kind : str, optional
        Search type: "grid" (default) or "randomized".
    step_name : str, optional
        Name of search step in new pipeline, by default "search".
    n_jobs: int, optional
        Number of jobs to run in parallel.
    **kwargs
        Additional keyword arguments for search estimator.

    Returns
    -------
    Pipeline
        Pipeline with search as the final estimator.
    """
    # Clone estimator from `estimator_pipe`
    estimator = clone(estimator_pipe[-1])

    # Reset parameters in `estimator` which are in `param_grid`
    defaults = pd.Series(utils.get_defaults(estimator.__class__))
    to_reset = defaults.loc[defaults.index.isin(param_grid)]
    estimator.set_params(**to_reset)

    # Create search estimator
    if kind.lower() == "grid":
        search = GridSearchCV(estimator, param_grid, n_jobs=n_jobs, **kwargs)
    elif kind.lower() == "randomized":
        search = RandomizedSearchCV(estimator, param_grid, n_jobs=n_jobs, **kwargs)

    # Construct search pipeline with search as last step
    search_pipe = list(clone(estimator_pipe[:-1]).named_steps.items())
    search_pipe.append((step_name, search))
    return Pipeline(search_pipe)


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

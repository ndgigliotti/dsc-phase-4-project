import os
import tempfile
from operator import itemgetter
from typing import Callable, Dict, List, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from feature_engine.selection import \
    SmartCorrelatedSelection as SmartCorrelatedSelectionFE
from IPython.core.display import HTML
from IPython.display import display
from numpy import ndarray
from numpy.random import RandomState
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (GridSearchCV, HalvingGridSearchCV,
                                     HalvingRandomSearchCV, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .. import utils

Variables = Union[None, int, str, List[Union[str, int]]]


def _to_pickle(obj: object, dst: str, test: bool = False) -> str:
    """Pickle an object via Joblib.

    Parameters
    ----------
    obj : object
        Object to serialize and save to disk.
    dst : str
        Filepath of file to create. Directories will also be created
        if necessary.
    test : bool, optional
        Pickles object to a temporary file to see if pickling errors
        are raised. The file is immediately removed. By default False.

    Returns
    -------
    str
        Filepath of object.
    """

    if test:
        # Pickle object to tempfile
        with tempfile.TemporaryFile() as f:
            # Deleted when closed
            joblib.dump(obj, f)
            dst = "success"
    else:
        # Pickle object to `dst`
        dst = os.path.normpath(dst)

        if os.path.dirname(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)

        if not os.path.basename(dst):
            raise ValueError(f"Invalid file path: {dst}")

        # Add extension if missing
        if ".joblib" not in os.path.basename(dst):
            dst = f"{dst}.joblib"
        joblib.dump(obj, dst)
    return dst


def sweep(
    estimator: Union[BaseEstimator, Pipeline],
    param_space: Union[Dict, List[Dict], Series],
    *,
    X: Union[DataFrame, Series, ndarray],
    y: Union[Series, ndarray],
    dst: str,
    scoring: Union[str, Callable, List, Tuple, Dict] = None,
    n_jobs: int = None,
    n_iter: int = 10,
    refit: bool = False,
    cv: int = None,
    kind: str = "grid",
    verbose: int = 1,
    pre_dispatch: str = "2*n_jobs",
    error_score: float = np.nan,
    return_train_score: bool = False,
    random_state: Union[int, RandomState] = None,
    factor: int = 3,
    resource: str = "n_samples",
    max_resources: Union[int, str] = "auto",
    min_resources: Union[int, str] = "exhaust",
    aggressive_elimination: bool = False,
    **kwargs,
) -> str:
    """Fit and pickle any Scikit-Learn search estimator.

    Fit and pickle a GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV,
    or HalvingRandomSearchCV object. Immediately saving the search estimator
    helps prevent losing the results. See the Scikit-Learn documentation on
    the aforementioned search estimators for more details on their parameters.

    Parameters
    ----------
    estimator : Union[BaseEstimator, Pipeline]
        Estimator or pipeline ending with estimator.
    param_space : Union[Dict, List[Dict], Series]
        Specification of the parameter search space.
    X : Union[DataFrame, Series, ndarray]
        Independent variables.
    y : Union[Series, ndarray]
        Target variable.
    dst : str
        Output filepath.
    scoring : Union[str, Callable, List, Tuple, Dict], optional
        Metric name(s) or callable(s) to be passed to search estimator.
    n_jobs : int, optional
        Number of tasks to run in parallel. Defaults to 1 if not specified.
        Pass -1 to use all available CPU cores.
    n_iter : int, optional
        Number of iterations for randomized search, by default 10.
        Irrelevant for non-randomized searches.
    refit : bool, optional
        Whether to refit the estimator with the best parameters from the
        search. False by default.
    cv : int, optional
        Number of cross validation folds, or cross validator object.
        Defaults to 5 if not specified.
    kind : str, optional
        String specifying search type:
            * 'grid' - GridSearchCV
            * 'hgrid' - HalvingGridSearchCV
            * 'rand' - RandomizedSearchCV
            * 'hrand' - HalvingRandomSearchCV
    verbose : int, optional
        Print out details about the search, by default 1.
    pre_dispatch : str, optional
        Controls the number of jobs that get dispatched during parallel
        execution, by default "2*n_jobs".
    error_score : float, optional
        Score if an error occurs in estimator fitting, by default np.nan.
    return_train_score : bool, optional
        Whether to include training scores in `cv_results_`, by default False.
    random_state : int or RandomState, optional
        Seed for random number generator, or RandomState, by default None.
        Only relevant for randomized searches.
    factor : int, optional
        Proportion of candidates that are selected for each subsequent iteration,
        by default 3. Only relevant for halving searches.
    resource : str, optional
        Defines the resource that increases with each iteration, 'n_samples' by default.
        Only relevant for halving searches.
    max_resources : Union[int, str], optional
        The maximum amount of resource that any candidate is allowed to use
        for a given iteration, by default 'auto'. Only relevant for halving searches.
    min_resources : Union[int, str], optional
        The minimum amount of resource that any candidate is allowed to use
        for a given iteration. Can be integer, 'smallest' or 'exhaust' (default).
        Only relevant for halving searches.
    aggressive_elimination : bool, optional
        Replay the first iteration to weed out candidates until enough are eliminated
        such that only `factor` candidates are evaluated in the final iteration.
        False by default. Only relevant for halving searches.

    Returns
    -------
    str
        Filename of pickled search estimator.

    """

    # Select search class
    kinds = dict(
        grid=GridSearchCV,
        hgrid=HalvingGridSearchCV,
        rand=RandomizedSearchCV,
        hrand=HalvingRandomSearchCV,
    )
    try:
        cls = kinds[kind.lower()]
    except KeyError:
        raise ValueError("Valid kinds are 'grid', 'hgrid', 'rand', and 'hrand'.")

    # Convert param space to dict
    if isinstance(param_space, pd.Series):
        param_space = param_space.to_dict()

    # Filter out the relevant parameters
    relevant = pd.Series(locals()).drop("kwargs")
    relevant["param_grid"] = param_space
    relevant["param_distributions"] = param_space
    relevant = relevant.loc[utils.get_param_names(cls.__init__)]
    relevant.update(kwargs)

    search = cls(**relevant)

    # Test pickling search estimator before fitting
    _to_pickle(search, dst, test=True)

    search.fit(X, y)

    # Pickle search estimator
    filename = _to_pickle(search, dst)

    return filename


def load_results(
    path: str,
    *,
    drop_splits: bool = True,
    short_names: bool = True,
    drop_dicts: bool = True,
    stats: Sequence[str] = ("mean_test_score", "rank_test_score"),
    rank_index: bool = False,
) -> DataFrame:
    """Load stripped-down version of search results from pickle.

    Retrieves the `cv_results_` from a pickled, fitted, search estimator
    and optionally trims it down for quick readability.

    Parameters
    ----------
    path : str
        Filename of pickled search estimator.
    drop_splits : bool, optional
        Drop the columns of individual cross validation splits. By default True.
    short_names : bool, optional
        Strip pipeline prefixes and extra words like 'test' from column labels.
        By default True.
    drop_dicts : bool, optional
        Drop parameter dictionaries to make the DataFrame prettier. By default True.
    stats : Sequence[str], optional
        Stats to include in the report, by default 'mean_test_score'
        and 'rank_test_score'. Pass `None` to for all the available stats.
    rank_index : bool, optional
        Set the index to 'rank_test_score' and sort by index. There may be duplicate
        indices. By default False.

    Returns
    -------
    DataFrame
        Table of cross validation results.
    """
    # Load search estimator
    if ".joblib" not in path:
        path = f"{path}.joblib"
    search = joblib.load(os.path.normpath(path))

    # Construct DataFrame
    df = pd.DataFrame(search.cv_results_)

    # Identify param columns and stat columns
    par_cols = df.columns[df.columns.str.startswith("param_")].to_list()
    par_cols.sort()
    if not drop_dicts:
        par_cols += ["params"]
    stat_cols = df.columns[~df.columns.isin(par_cols)].to_list()

    # Put param columns on the left and stat columns on the right
    df = utils.explicit_sort(df, order=(par_cols + stat_cols), mode="index", axis=1)

    # Drop stats not specified in `stats` argument
    if stats is not None:
        df.drop(set(stat_cols) - set(stats), axis=1, inplace=True)

    # Prune columns of individual splits
    if drop_splits:
        splits = df.filter(regex=r"split[0-9]+_").columns
        df.drop(columns=splits, inplace=True)

    # Set index to 'rank_test_score' (if applicable) and sort
    if rank_index:
        if "rank_test_score" not in df.columns:
            raise RuntimeWarning("Could not set index to 'rank_test_score'")
        else:
            df.set_index("rank_test_score", drop=True, inplace=True)
            df.sort_index(inplace=True)
    # Sort by 'rank_test_score' no matter what (unless dropped)
    elif "rank_test_score" in df.columns:
        df.sort_values("rank_test_score", inplace=True)

    # Cut out pipeline prefixes and the word 'test'
    if short_names:
        df.columns = df.columns.str.split("__").map(itemgetter(-1))
        df.columns = df.columns.str.replace("test_score", "score", regex=False)
        if df.index.name is not None:
            df.index.name = df.index.name.replace("test_score", "score")
    return df


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
        name = self.__class__.__name__
        info = info.T.to_html(na_rep="", header=False, max_cols=6, notebook=True)
        info = f"<h4>{name}</h4>{info}"
        display(HTML(info))

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        super().fit(X, y=y)
        if self.verbose:
            self.show_report()
        return self

import os
from operator import itemgetter
from types import MappingProxyType
from typing import List, Union

import numpy as np
import pandas as pd
from feature_engine.selection import (
    SmartCorrelatedSelection as SmartCorrelatedSelectionFE,
)
from IPython.core.display import HTML
from IPython.display import display
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import joblib
from .. import utils

Variables = Union[None, int, str, List[Union[str, int]]]

def _to_pickle(search, name, dirname, test=False):
    os.makedirs(dirname, exist_ok=True)
    filename = name if name.endswith(".joblib") else f"{name}.joblib"
    filename = os.path.join(dirname, filename)
    joblib.dump(search, filename)
    if test:
        os.remove(filename)
    return filename

def sweep(
    estimator,
    param_space,
    X,
    y,
    name,
    dirname="sweeps",
    scoring=None,
    n_jobs=None,
    n_iter=10,
    refit=False,
    cv=None,
    kind="grid",
    verbose=1,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
    return_train_score=False,
    random_state=None,
    n_candidates="exhaust",
    factor=3,
    resource="n_samples",
    max_resources="auto",
    min_resources="smallest",
    aggressive_elimination=False,
    **kwargs,
):
    """Flexible parameter search function.
    
    Fit and pickle a GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV,
    or HalvingRandomSearchCV object. Immediately saving the search estimator
    prevents you from losing the results of a broad sweep."""

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
    _to_pickle(search, name, dirname, test=True)

    search.fit(X, y)

    # Pickle search estimator
    filename = _to_pickle(search, name, dirname)

    return filename


def load_results(
    path,
    drop_splits=True,
    short_names=True,
    drop_dicts=True,
    stats=("mean_test_score", "rank_test_score"),
    rank_index=False,
):
    if not path.endswith(".joblib"):
        path = f"{path}.joblib"
    search = joblib.load(os.path.normpath(path))
    df = pd.DataFrame(search.cv_results_)
    par_cols = df.columns[df.columns.str.startswith("param_")].to_list()
    par_cols.sort()
    if not drop_dicts:
        par_cols += ["params"]
    stat_cols = df.columns[~df.columns.isin(par_cols)].to_list()
    df = utils.explicit_sort(df, order=(par_cols + stat_cols), mode="index", axis=1)
    if stats is not None:
        df.drop(set(stat_cols) - set(stats), axis=1, inplace=True)
    if drop_splits:
        splits = df.filter(regex=r"split[0-9]+_").columns
        df.drop(columns=splits, inplace=True)
    if rank_index:
        if "rank_test_score" not in df.columns:
            raise RuntimeWarning("Could not set index to 'rank_test_score'")
        else:
            df.set_index("rank_test_score", drop=True, inplace=True)
            df.sort_index(inplace=True)
    elif "rank_test_score" in df.columns:
            df.sort_values("rank_test_score", inplace=True)
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

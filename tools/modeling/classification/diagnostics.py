from functools import partial
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.metrics import classification_report as sk_report
from sklearn.metrics import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ...plotting import smart_subplots
from ...utils import pandas_heatmap


def _get_estimator_name(estimator: Union[BaseEstimator, Pipeline]) -> str:
    """Returns estimator class name.

    If a Pipeline is passed, returns the class name of the final estimator.

    Parameters
    ----------
    estimator : Estimator or Pipeline
        Estimator to get class name for.

    Returns
    -------
    str
        Class name.
    """
    if isinstance(estimator, Pipeline):
        name = estimator[-1].__class__.__name__
    else:
        name = estimator.__class__.__name__
    return name


def high_correlations(data: pd.DataFrame, thresh: float = 0.75) -> pd.Series:
    """Get non-trivial feature correlations at or above `thresh`.

    Parameters
    ----------
    data : DataFrame
        Data for finding high correlations.
    thresh : float, optional
        High correlation threshold, by default 0.75.

    Returns
    -------
    Series
        High correlations.
    """
    corr_df = pd.get_dummies(data).corr()
    mask = np.tril(np.ones_like(corr_df, dtype=np.bool_))
    corr_df = corr_df.mask(mask).stack()
    high = corr_df >= thresh
    return corr_df[high]


def classification_report(
    y_test: Union[pd.Series, np.ndarray],
    y_pred: np.ndarray,
    zero_division: str = "warn",
    heatmap: bool = False,
) -> Union[pd.DataFrame, Styler]:
    """Return diagnostic report for classification, optionally styled as a heatmap.

    Parameters
    ----------
    y_test : Series or ndarray of shape (n_samples,)
        Target test set.
    y_pred :  ndarray of shape (n_samples,)
        Values predicted from predictor test set.
    zero_division : str, optional
        Value to return for division by zero: 0, 1, or 'warn'.
    heatmap : bool, optional
        Style report as a heatmap, by default False.

    Returns
    -------
    DataFrame or Styler (if `heatmap = True`)
        Diagnostic report table.
    """
    report = pd.DataFrame(
        sk_report(y_test, y_pred, output_dict=True, zero_division=zero_division)
    )

    order = report.columns.to_list()[:2] + [
        "macro avg",
        "weighted avg",
        "accuracy",
    ]
    report = report.loc[:, order]

    support = report.loc["support"].iloc[:2]
    support /= report.loc["support", "macro avg"]
    report.loc["support"] = support

    report["bal accuracy"] = balanced_accuracy_score(y_test, y_pred)
    mask = np.array([[0, 1, 1, 1], [0, 1, 1, 1]]).T.astype(np.bool_)
    report[["accuracy", "bal accuracy"]] = report.filter(like="accuracy", axis=1).mask(
        mask
    )

    return (
        pandas_heatmap(report, subset=["0.0", "1.0"], axis=1, vmin=0, vmax=1)
        if heatmap
        else report
    )


def compare_scores(estimator_1, estimator_2, X_test, y_test, prec=3, heatmap=True):
    scores_1 = classification_report(
        y_test, estimator_1.predict(X_test), precision=prec
    )
    scores_2 = classification_report(
        y_test, estimator_2.predict(X_test), precision=prec
    )
    result = scores_1.compare(scores_2, keep_equal=True, keep_shape=True)
    name_1 = _get_estimator_name(estimator_1)
    name_2 = _get_estimator_name(estimator_2)
    result.rename(columns=dict(self=name_1, other=name_2), inplace=True)
    result = result.T
    return pandas_heatmap(result) if heatmap else result


def classification_plots(
    estimator: Union[BaseEstimator, Pipeline],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    average: str = "weighted",
    size: Tuple[float, float] = (5, 5),
) -> plt.Figure:
    """Plot confusion matrix, ROC curve, and precision-recall curve.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Fitted classification estimator or pipeline with fitted
        final estimator to evaluate.
    X_test : DataFrame or ndarray of shape (n_samples, n_features)
        Predictor test set.
    y_test : Series or ndarray of shape (n_samples,)
        target test set.
    average : str, optional
        Method of averaging: 'micro', 'macro', 'weighted' (default), 'samples'.
    size: tuple (float, float), optional
        Size of each subplot; (5, 5) by default.

    Returns
    -------
    Figure
        Figure containing three subplots.
    """
    fig, (ax1, ax2, ax3) = smart_subplots(nplots=3, ncols=3, size=size)
    plot_confusion_matrix(
        estimator,
        X_test,
        y_test,
        cmap="Blues",
        normalize="true",
        colorbar=False,
        ax=ax1,
    )
    plot_roc_curve(estimator, X_test, y_test, ax=ax2)
    plot_precision_recall_curve(estimator, X_test, y_test, ax=ax3)

    baseline_style = dict(lw=2, linestyle=":", color="r", alpha=1)
    ax2.plot([0, 1], [0, 1], **baseline_style)
    ax3.plot([0, 1], [y_test.mean()] * 2, **baseline_style)
    ax3.plot([0, 0], [y_test.mean(), 1], **baseline_style)

    try:
        y_score = estimator.predict_proba(X_test)[:, estimator.classes_.argmax()]
    except AttributeError:
        y_score = estimator.decision_function(X_test)
    auc_score = roc_auc_score(y_test, y_score, average=average).round(2)
    ap_score = average_precision_score(y_test, y_score, average=average).round(2)

    ax1.set_title("Normalized Confusion Matrix")
    ax2.set_title(f"Receiver Operating Characteristic Curve: AUC = {auc_score}")
    ax3.set_title(f"Precision-Recall Curve: AP = {ap_score}")
    ax2.get_legend().set_visible(False)
    ax3.get_legend().set_visible(False)
    fig.tight_layout()
    return fig


def plot_double_confusion_matrices(
    estimator: Union[BaseEstimator, Pipeline],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    cmap: str = "Blues",
    colorbar: bool = False,
    size: Tuple[float, float] = (5, 5),
    **kwargs,
) -> plt.Figure:
    """Plot normalized and raw confusion matrices side by side.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Fitted classification estimator or pipeline with fitted
        final estimator to evaluate.
    X_test : DataFrame or ndarray of shape (n_samples, n_features)
        Predictor test set.
    y_test : Series or ndarray of shape (n_samples,)
        target test set.
    cmap : str, optional
        Matplotlib colormap for the matrices, by default "Blues".
    colorbar : bool, optional
        Show colorbars, by default False.
    size: tuple (float, float), optional
        Size of each subplot; (5, 5) by default.
    **kwargs:
        Keyword arguments passed to `sklearn.metrics.plot_confusion_matrix`
        for both plots.

    Returns
    -------
    Figure
        Two confusion matrices.
    """
    fig, (ax1, ax2) = smart_subplots(nplots=2, size=size, ncols=2, sharey=True)

    # Generic partial function for both plots
    plot_matrix = partial(
        plot_confusion_matrix,
        estimator=estimator,
        X=X_test,
        y_true=y_test,
        cmap=cmap,
        colorbar=colorbar,
        **kwargs,
    )

    # Plot normalized matrix
    plot_matrix(ax=ax1, normalize="true")

    # Plot raw matrix
    plot_matrix(ax=ax2)

    # Set titles
    ax1.set(title="Normalized Confusion Matrix")
    ax2.set(title="Raw Confusion Matrix")

    fig.tight_layout()
    return fig


def standard_report(
    estimator: BaseEstimator,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    zero_division: str = "warn",
) -> None:
    """Display standard report of diagnostic metrics and plots for classification.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted classification estimator for evaluation.
    X_test : DataFrame or ndarray of shape (n_samples, n_features)
        Predictor test set.
    y_test : Series or ndarray of shape (n_samples,)
        Target test set.
    zero_division : str, optional
        Value to return for division by zero: 0, 1, or 'warn'.
    """
    table = classification_report(
        y_test, estimator.predict(X_test), zero_division=zero_division, heatmap=True
    )
    classification_plots(estimator, X_test, y_test)
    display(table)

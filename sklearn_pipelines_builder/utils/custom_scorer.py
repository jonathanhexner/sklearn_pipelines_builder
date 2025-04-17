from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, get_scorer, make_scorer
)
import numpy as np
import functools


# ✅ Decorator to negate scorer results while preserving `_score_func`
def negate_scorer(scorer_func):
    """Decorator that negates the result of a scoring function while preserving `_score_func`."""

    @functools.wraps(scorer_func)
    def wrapper(y_true, y_pred, sample_weight=None):
        return -scorer_func(y_true, y_pred, sample_weight)

    wrapper._score_func = scorer_func  # Preserve `_score_func`
    return wrapper


# ✅ Scoring functions
def weighted_mae(y_true, y_pred, sample_weight=None):
    """Computes weighted Mean Absolute Error (MAE)."""
    if sample_weight is None:
        return mean_absolute_error(y_true, y_pred)
    return np.average(np.abs(y_true - y_pred), weights=sample_weight)


def weighted_mse(y_true, y_pred, sample_weight=None):
    """Computes weighted Mean Squared Error (MSE)."""
    if sample_weight is None:
        return mean_squared_error(y_true, y_pred)
    return np.average((y_true - y_pred) ** 2, weights=sample_weight)


def weighted_rmse(y_true, y_pred, sample_weight=None):
    """Computes weighted Root Mean Squared Error (RMSE)."""
    return np.sqrt(weighted_mse(y_true, y_pred, sample_weight))


# ✅ Wrapper for model-based scoring that supports `sample_weight`
def model_based_scorer(scorer_func):
    """
    Wraps a scorer function to support model-based evaluation.
    - Calls `estimator.predict(X)` to get predictions.
    - Supports `sample_weight` in both `fit()` and evaluation.

    Returns:
    - A function that can be used in `cross_val_score` and `GridSearchCV`.
    """

    @functools.wraps(scorer_func)
    def wrapper(estimator, X, y_true, sample_weight=None):
        y_pred = estimator.predict(X)  # Get predictions from model
        return scorer_func(y_true, y_pred, sample_weight)

    wrapper._score_func = scorer_func  # Preserve `_score_func`
    return wrapper


# ✅ Automatically fetches Scikit-learn scorers and wraps them
def generic_scorer_wrapper(scorer_name):
    """
    Wraps any Scikit-learn scorer to accept `sample_weight`, even if it ignores it.
    Calls `get_scorer(scorer_name)` internally.
    """
    sklearn_scorer = get_scorer(scorer_name)

    @functools.wraps(sklearn_scorer._score_func)
    def wrapped_scorer(y_true, y_pred, sample_weight=None):
        return sklearn_scorer._score_func(y_true, y_pred)  # sample_weight ignored

    wrapped_scorer._score_func = sklearn_scorer._score_func  # Maintain `_score_func`
    return wrapped_scorer


# ✅ Define a dynamic scorer registry
SCORER_REGISTRY = {
    "mse": model_based_scorer(negate_scorer(weighted_mse)),
    "rmse": model_based_scorer(negate_scorer(weighted_rmse)),
    "mae": model_based_scorer(negate_scorer(weighted_mae)),  # ✅ Uses decorator
    "weighted_mae": model_based_scorer(negate_scorer(weighted_mae)),
}


def get_custom_scorer(scorer_name):
    """
    Returns a scorer function that:
    - Supports both `_score_func` (direct usage) and model-based scoring.
    - Uses Scikit-learn's implementation where possible.
    - Wraps standard scorers to accept `sample_weight`, even if ignored.

    Parameters:
    - scorer_name (str): Name of the metric (e.g., "mse", "r2", "mae", "weighted_mae").

    Returns:
    - A scorer function that can be used with both models and direct vectors.
    """
    if scorer_name in SCORER_REGISTRY:
        return SCORER_REGISTRY[scorer_name]

    # ✅ If the scorer is not in our custom registry, use Scikit-learn’s version
    try:
        return generic_scorer_wrapper(scorer_name)
    except KeyError:
        raise ValueError(
            f"Scorer '{scorer_name}' is not supported. Choose from {list(SCORER_REGISTRY.keys()) + list(get_scorer_names())}")


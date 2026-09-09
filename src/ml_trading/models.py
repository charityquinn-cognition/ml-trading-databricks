"""Model factory.

Every model is a scikit-learn ``Pipeline`` that starts with imputation and scaling so
that a fold with a degenerate feature (all-NaN, zero variance) cannot silently poison
the fit, and so the same estimator interface works for linear and tree models.

The families here span the axes that matter for a noisy, low-signal cross-section: a
regularised linear model (hard to beat when the signal-to-noise ratio is this low), two
bagged tree families, two boosted tree families, a small neural net, and a soft-vote
ensemble of the three families that make different errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from ml_trading.config import ModelConfig

LINEAR_MODELS = {"logistic", "ridge"}
SCALED_MODELS = LINEAR_MODELS | {"mlp", "ensemble"}
"""Models fed standardised inputs. The ensemble is scaled once in the outer pipeline so
its linear member behaves; the tree members are indifferent to a monotone rescaling."""

UNWEIGHTED_MODELS = {"mlp"}
"""Families whose ``fit`` takes no ``sample_weight``."""

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic": {"C": 1.0, "max_iter": 500, "solver": "lbfgs"},
    "ridge": {"alpha": 100.0},
    "random_forest": {
        "n_estimators": 400,
        "max_depth": 5,
        "min_samples_leaf": 200,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 400,
        "max_depth": 8,
        "min_samples_leaf": 200,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
    "gbm": {
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    },
    "hist_gbm": {
        "max_iter": 300,
        "learning_rate": 0.03,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
        "early_stopping": False,
    },
    "mlp": {
        "hidden_layer_sizes": (32, 16),
        "alpha": 1e-2,
        "learning_rate_init": 1e-3,
        "max_iter": 60,
        "batch_size": 4096,
        "early_stopping": False,
    },
    "ensemble": {},
}

DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic": {"C": [0.01, 0.1, 1.0]},
    "ridge": {"alpha": [10.0, 100.0, 1000.0]},
    "random_forest": {"max_depth": [3, 5, 8], "min_samples_leaf": [100, 200, 500]},
    "extra_trees": {"max_depth": [5, 8, 12], "min_samples_leaf": [100, 200, 500]},
    "gbm": {
        "learning_rate": [0.01, 0.03, 0.1],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [100, 200, 500],
    },
    "hist_gbm": {"learning_rate": [0.01, 0.03, 0.1], "max_leaf_nodes": [7, 15, 31]},
    "mlp": {"alpha": [1e-3, 1e-2, 1e-1]},
}

ENSEMBLE_MEMBERS = ("logistic", "random_forest", "gbm")
"""Ensemble members, chosen to be a linear model, a bagged tree and a boosted tree: the
errors of the three are much less correlated than three variants of one family."""


def build_model(
    config: ModelConfig,
    *,
    regression: bool = False,
    seed: int = 0,
    overrides: dict[str, Any] | None = None,
) -> Pipeline:
    """Build the estimator pipeline named by ``config``.

    ``regression`` selects the regressor flavour of the same family, used when the label
    is a continuous (volatility-normalised) forward return rather than a sign.
    ``overrides`` wins over the config, and is how the per-fold tuner injects candidates.
    """
    name = config.name
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"unknown model: {name}. Choose from {sorted(DEFAULT_PARAMS)}")
    if (name == "ridge") != regression and name in LINEAR_MODELS:
        expected = "ridge" if regression else "logistic"
        raise ValueError(f"model '{name}' does not support this label kind; use '{expected}'")

    params = {**DEFAULT_PARAMS[name], **config.params, **(overrides or {})}
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if name in SCALED_MODELS:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", _estimator(name, params, regression=regression, seed=seed)))
    return Pipeline(steps)


def _estimator(name: str, params: dict[str, Any], *, regression: bool, seed: int) -> Any:
    if name == "ridge":
        return Ridge(random_state=seed, **params)
    if name == "logistic":
        return LogisticRegression(random_state=seed, **params)
    if name == "random_forest":
        forest = RandomForestRegressor if regression else RandomForestClassifier
        return forest(random_state=seed, **params)
    if name == "extra_trees":
        trees = ExtraTreesRegressor if regression else ExtraTreesClassifier
        return trees(random_state=seed, **params)
    if name == "hist_gbm":
        hist = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
        return hist(random_state=seed, **params)
    if name == "mlp":
        net = MLPRegressor if regression else MLPClassifier
        return net(random_state=seed, **params)
    if name == "ensemble":
        return _voting_estimator(params, regression=regression, seed=seed)

    from lightgbm import LGBMClassifier, LGBMRegressor

    gbm = LGBMRegressor if regression else LGBMClassifier
    return gbm(random_state=seed, **params)


def _voting_estimator(params: dict[str, Any], *, regression: bool, seed: int) -> Any:
    """Average the members' predictions; each member keeps its own defaults.

    Members are bare estimators rather than nested pipelines so that a ``sample_weight``
    passed to the ensemble reaches all of them.
    """
    members = []
    for member in ENSEMBLE_MEMBERS:
        name = "ridge" if regression and member == "logistic" else member
        member_params = {**DEFAULT_PARAMS[name], **params.get(name, {})}
        members.append((member, _estimator(name, member_params, regression=regression, seed=seed)))
    if regression:
        return VotingRegressor(members)
    return VotingClassifier(members, voting="soft")


def parameter_grid(config: ModelConfig) -> list[dict[str, Any]]:
    """Candidate parameter dicts for the per-fold tuner, as a full cross product."""
    if not config.tune:
        return []
    grids = {**DEFAULT_GRIDS.get(config.name, {}), **config.tune_grid}
    missing = [name for name in config.tune if name not in grids]
    if missing:
        raise ValueError(f"no tuning grid for {missing} on model '{config.name}'")

    candidates: list[dict[str, Any]] = [{}]
    for name in config.tune:
        candidates = [
            {**candidate, name: value} for candidate in candidates for value in grids[name]
        ]
    return candidates


def supports_sample_weight(config: ModelConfig) -> bool:
    """Whether this family's ``fit`` accepts per-row weights."""
    return config.name not in UNWEIGHTED_MODELS


def feature_importances(pipeline: Pipeline, names: list[str]) -> dict[str, float]:
    """Best-effort importance extraction, normalised to sum to one."""
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        raw = list(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        raw = [abs(value) for value in estimator.coef_.ravel()]
    else:
        return {}

    total = sum(raw)
    if total <= 0:
        return dict.fromkeys(names, 0.0)
    return {name: float(value) / float(total) for name, value in zip(names, raw, strict=True)}

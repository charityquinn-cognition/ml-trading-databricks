"""Model factory.

Every model is a scikit-learn ``Pipeline`` that starts with imputation and scaling so
that a fold with a degenerate feature (all-NaN, zero variance) cannot silently poison
the fit, and so the same estimator interface works for linear and tree models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from ml_trading.config import ModelConfig

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic": {"C": 1.0, "max_iter": 500, "solver": "lbfgs"},
    "ridge": {"alpha": 100.0},
    "random_forest": {
        "n_estimators": 400,
        "max_depth": 5,
        "min_samples_leaf": 200,
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
}


def build_model(config: ModelConfig, *, regression: bool = False, seed: int = 0) -> Pipeline:
    """Build the estimator pipeline named by ``config``.

    ``regression`` selects the regressor flavour of the same family, used when the label
    is a continuous (volatility-normalised) forward return rather than a sign.
    """
    name = config.name
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"unknown model: {name}. Choose from {sorted(DEFAULT_PARAMS)}")
    if (name == "ridge") != regression and name in {"ridge", "logistic"}:
        expected = "ridge" if regression else "logistic"
        raise ValueError(f"model '{name}' does not support this label kind; use '{expected}'")

    params = {**DEFAULT_PARAMS[name], **config.params}
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]

    if name in {"logistic", "ridge"}:
        steps.append(("scale", StandardScaler()))
        estimator: Any = (
            Ridge(random_state=seed, **params)
            if name == "ridge"
            else LogisticRegression(random_state=seed, **params)
        )
    elif name == "random_forest":
        forest = RandomForestRegressor if regression else RandomForestClassifier
        estimator = forest(random_state=seed, **params)
    else:
        from lightgbm import LGBMClassifier, LGBMRegressor

        gbm = LGBMRegressor if regression else LGBMClassifier
        estimator = gbm(random_state=seed, **params)

    steps.append(("model", estimator))
    return Pipeline(steps)


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

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_trading.config import ModelConfig
from ml_trading.models import build_model, feature_importances, supports_sample_weight


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.Series]:
    # The shipped tree defaults are heavily regularised (hundreds of samples per leaf),
    # so the fixture needs a realistic number of rows for them to split at all.
    rng = np.random.default_rng(0)
    rows = 4000
    features = pd.DataFrame(rng.normal(size=(rows, 4)), columns=list("abcd"))
    target = features["a"] + 0.2 * rng.normal(size=rows)
    return features, target


CLASSIFIERS = ["logistic", "random_forest", "extra_trees", "gbm", "hist_gbm", "mlp", "ensemble"]
REGRESSORS = ["ridge", "random_forest", "extra_trees", "gbm", "hist_gbm", "ensemble"]


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_classifiers_fit_and_predict_probabilities(
    name: str, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    features, target = xy
    model = build_model(ModelConfig(name=name), regression=False, seed=0)
    model.fit(features, (target > 0).astype(int))
    proba = model.predict_proba(features)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("name", REGRESSORS)
def test_regressors_fit_and_predict(name: str, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = xy
    model = build_model(ModelConfig(name=name), regression=True, seed=0)
    model.fit(features, target)
    assert np.corrcoef(model.predict(features), target)[0, 1] > 0.5


def test_missing_values_are_imputed(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = xy
    holed = features.copy()
    holed.iloc[::10, 0] = np.nan
    model = build_model(ModelConfig(name="logistic"), seed=0)
    model.fit(holed, (target > 0).astype(int))
    assert np.isfinite(model.predict_proba(holed)[:, 1]).all()


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model(ModelConfig(name="transformer"))


def test_linear_family_must_match_the_label_kind() -> None:
    with pytest.raises(ValueError, match="does not support"):
        build_model(ModelConfig(name="logistic"), regression=True)
    with pytest.raises(ValueError, match="does not support"):
        build_model(ModelConfig(name="ridge"), regression=False)


def test_config_params_override_defaults() -> None:
    model = build_model(ModelConfig(name="logistic", params={"C": 0.01}))
    assert model.named_steps["model"].C == 0.01


def test_seed_makes_fits_reproducible(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = xy
    labels = (target > 0).astype(int)
    first = build_model(ModelConfig(name="random_forest"), seed=3).fit(features, labels)
    second = build_model(ModelConfig(name="random_forest"), seed=3).fit(features, labels)
    np.testing.assert_allclose(
        first.predict_proba(features)[:, 1], second.predict_proba(features)[:, 1]
    )


@pytest.mark.parametrize("name", [name for name in CLASSIFIERS if name != "mlp"])
def test_weighted_fits_are_accepted_by_every_weighted_family(
    name: str, xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    """Sample weights must reach the estimator - including each member of the ensemble."""
    features, target = xy
    config = ModelConfig(name=name)
    assert supports_sample_weight(config)
    weights = np.linspace(0.5, 1.5, len(features))
    model = build_model(config, seed=0)
    model.fit(features, (target > 0).astype(int), model__sample_weight=weights)
    assert np.isfinite(model.predict_proba(features)[:, 1]).all()


def test_the_neural_net_is_declared_unweighted() -> None:
    assert not supports_sample_weight(ModelConfig(name="mlp"))


def test_feature_importances_sum_to_one(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    features, target = xy
    model = build_model(ModelConfig(name="logistic"), seed=0)
    model.fit(features, (target > 0).astype(int))
    importances = feature_importances(model, list(features.columns))
    assert set(importances) == set(features.columns)
    assert sum(importances.values()) == pytest.approx(1.0)
    assert max(importances, key=importances.__getitem__) == "a"

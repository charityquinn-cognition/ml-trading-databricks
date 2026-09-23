from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_trading.config import ExperimentConfig, FeatureConfig, LabelConfig, ModelConfig
from ml_trading.features import build_dataset, feature_columns
from ml_trading.models import parameter_grid
from ml_trading.training import (
    fit_fold_model,
    purged_split,
    sample_weights,
    tuned_parameters,
)
from tests.conftest import make_prices


@pytest.fixture
def dataset() -> pd.DataFrame:
    frame = build_dataset(
        make_prices(days=1200, symbols=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")),
        FeatureConfig(momentum_windows=(5, 21), skip_momentum_windows=(63,), beta_window=63),
        LabelConfig(horizon=5),
    )
    return frame.dropna(subset=["label", "forward_return"]).reset_index(drop=True)


def test_sample_weights_are_none_by_default(dataset: pd.DataFrame) -> None:
    assert sample_weights(dataset, LabelConfig()) is None


def test_return_attribution_weights_track_absolute_target(dataset: pd.DataFrame) -> None:
    weights = sample_weights(dataset, LabelConfig(sample_weight="return_attribution"))
    assert weights is not None
    assert weights.mean() == pytest.approx(1.0)
    target = dataset["forward_excess_return"].abs()
    assert np.corrcoef(weights, target)[0, 1] > 0.99


def test_time_decay_weights_favour_recent_rows(dataset: pd.DataFrame) -> None:
    weights = sample_weights(dataset, LabelConfig(sample_weight="time_decay"))
    assert weights is not None
    ordered = pd.Series(weights, index=dataset["date"]).groupby(level=0).mean().sort_index()
    assert ordered.iloc[-1] > ordered.iloc[0]
    assert ordered.is_monotonic_increasing


def test_purged_split_leaves_a_horizon_sized_gap(dataset: pd.DataFrame) -> None:
    front, back = purged_split(dataset, 0.7, horizon=5)
    assert not front.empty and not back.empty
    gap = (back["date"].min() - front["date"].max()).days
    assert gap > 5
    assert front["date"].max() < back["date"].min()


def test_purged_split_of_a_tiny_block_is_empty() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    front, back = purged_split(frame, 0.7, horizon=5)
    assert front.empty and back.empty


def test_parameter_grid_is_the_cross_product() -> None:
    grid = parameter_grid(ModelConfig(name="logistic", tune=("C",)))
    assert [candidate["C"] for candidate in grid] == [0.01, 0.1, 1.0]
    assert parameter_grid(ModelConfig(name="logistic")) == []


def test_parameter_grid_rejects_unknown_parameters() -> None:
    with pytest.raises(ValueError, match="no tuning grid"):
        parameter_grid(ModelConfig(name="logistic", tune=("nonsense",)))


def test_tuning_picks_a_candidate_from_the_grid(dataset: pd.DataFrame) -> None:
    config = ExperimentConfig(
        model=ModelConfig(name="logistic", tune=("C",), tune_grid={"C": [0.001, 1.0]})
    )
    chosen = tuned_parameters(dataset, feature_columns(dataset), config)
    assert chosen["C"] in {0.001, 1.0}


def test_meta_labelling_produces_scores_bounded_by_the_primary(dataset: pd.DataFrame) -> None:
    """Sizing by P(the bet is right) can only shrink a bet towards neutral, never flip it."""
    columns = feature_columns(dataset)
    cut = dataset["date"].quantile(0.8)
    train = dataset[dataset["date"] <= cut]
    test = dataset[dataset["date"] > cut]

    plain = ExperimentConfig(model=ModelConfig(name="logistic"))
    meta = ExperimentConfig(model=ModelConfig(name="logistic", meta_label=True))
    primary_scores, _, _ = fit_fold_model(train, test, columns, plain)
    meta_scores, _, _ = fit_fold_model(train, test, columns, meta)

    assert len(meta_scores) == len(test)
    assert np.all(np.sign(meta_scores - 0.5) == np.sign(primary_scores - 0.5))
    assert np.all(np.abs(meta_scores - 0.5) <= np.abs(primary_scores - 0.5) + 1e-9)


def test_meta_labelling_requires_a_binary_label() -> None:
    with pytest.raises(ValueError, match="binary primary label"):
        ExperimentConfig(
            label=LabelConfig(kind="continuous"),
            model=ModelConfig(name="ridge", meta_label=True),
        )

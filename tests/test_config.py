from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml_trading.config import (
    BacktestConfig,
    ExperimentConfig,
    LabelConfig,
    ModelConfig,
    SplitConfig,
)


def test_execution_lag_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="execution_lag"):
        ExperimentConfig(backtest=BacktestConfig(execution_lag=0))


def test_horizon_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        ExperimentConfig(label=LabelConfig(horizon=0))


@pytest.mark.parametrize("field", ["train_years", "test_years", "step_years"])
def test_nonpositive_split_years_are_rejected(field: str) -> None:
    """A step of zero would leave the walk-forward loop advancing nowhere, forever."""
    with pytest.raises(ValueError, match="step_years"):
        ExperimentConfig(splits=SplitConfig(**{field: 0.0}))


def test_inverted_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="short_threshold"):
        ExperimentConfig(backtest=BacktestConfig(long_threshold=0.4, short_threshold=0.6))


def test_unknown_label_kind_and_weighting_are_rejected() -> None:
    with pytest.raises(ValueError, match="label kind"):
        ExperimentConfig(label=LabelConfig(kind="ternary"))
    with pytest.raises(ValueError, match="weighting"):
        ExperimentConfig(backtest=BacktestConfig(weighting="equal"))


def test_unknown_sample_weighting_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample weighting"):
        ExperimentConfig(label=LabelConfig(sample_weight="uniformish"))


def test_meta_labelling_needs_a_binary_label() -> None:
    with pytest.raises(ValueError, match="binary primary label"):
        ExperimentConfig(
            label=LabelConfig(kind="continuous"),
            model=ModelConfig(name="ridge", meta_label=True),
        )


def test_tuning_fields_round_trip_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "tuned.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "label": {"sample_weight": "both", "time_decay_floor": 0.1},
                "features": {"anomaly_signals": False, "seasonality_years": 3},
                "model": {
                    "name": "gbm",
                    "tune": ["learning_rate"],
                    "tune_grid": {"learning_rate": [0.01, 0.05]},
                },
            }
        )
    )
    config = ExperimentConfig.from_yaml(path)
    assert config.label.sample_weight == "both"
    assert config.features.anomaly_signals is False
    assert config.model.tune == ("learning_rate",)
    assert config.model.tune_grid == {"learning_rate": [0.01, 0.05]}


def test_from_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "data": {"symbols": ["SPY", "QQQ"], "start": "2010-01-01"},
                "label": {"horizon": 10, "kind": "continuous"},
                "model": {"name": "ridge", "params": {"alpha": 1.0}},
            }
        )
    )
    config = ExperimentConfig.from_yaml(path)
    assert config.name == "demo"
    assert config.data.symbols == ("SPY", "QQQ")  # YAML lists become tuples
    assert config.label.horizon == 10
    assert config.model.params == {"alpha": 1.0}


def test_from_dict_rejects_unknown_sections() -> None:
    with pytest.raises(ValueError, match="unknown config key"):
        ExperimentConfig.from_dict({"nonsense": 1})


def test_flat_params_are_mlflow_friendly() -> None:
    flat = ExperimentConfig().flat_params()
    assert flat["label.horizon"] == 5
    assert flat["backtest.cost_bps"] == 5.0
    assert all("." in key or key in {"name", "seed"} for key in flat)


def test_shipped_configs_are_valid() -> None:
    for path in sorted(Path("configs").glob("*.yaml")):
        ExperimentConfig.from_yaml(path)

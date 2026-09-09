from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml_trading.config import BacktestConfig, ExperimentConfig, LabelConfig


def test_execution_lag_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="execution_lag"):
        ExperimentConfig(backtest=BacktestConfig(execution_lag=0))


def test_horizon_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        ExperimentConfig(label=LabelConfig(horizon=0))


def test_inverted_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="short_threshold"):
        ExperimentConfig(backtest=BacktestConfig(long_threshold=0.4, short_threshold=0.6))


def test_unknown_label_kind_and_weighting_are_rejected() -> None:
    with pytest.raises(ValueError, match="label kind"):
        ExperimentConfig(label=LabelConfig(kind="ternary"))
    with pytest.raises(ValueError, match="weighting"):
        ExperimentConfig(backtest=BacktestConfig(weighting="equal"))


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

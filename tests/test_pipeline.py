from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_trading.cli import main
from ml_trading.config import (
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    LabelConfig,
    ModelConfig,
    SplitConfig,
)
from ml_trading.pipeline import run_experiment, sweep
from tests.conftest import make_prices

FEATURES = FeatureConfig(
    momentum_windows=(5, 21),
    skip_momentum_windows=(63,),
    volatility_windows=(21,),
    ma_ratio_windows=(20,),
    beta_window=63,
)
SPLITS = SplitConfig(train_years=1.5, test_years=0.5, step_years=0.5)


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        features=FEATURES,
        label=LabelConfig(horizon=5),
        splits=SPLITS,
        model=ModelConfig(name="logistic"),
        backtest=BacktestConfig(cost_bps=1.0, volatility_target=None),
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_prices(symbols=("AAA", "BBB", "CCC", "DDD", "EEE"), days=1200)


def test_run_experiment_produces_out_of_sample_predictions(
    config: ExperimentConfig, panel: pd.DataFrame
) -> None:
    result = run_experiment(config, panel)
    assert len(result.folds) >= 2
    assert set(result.predictions.columns) >= {"date", "symbol", "score", "fold"}
    assert not result.predictions.duplicated(subset=["date", "symbol"]).any()
    # Every prediction must come from a model fitted strictly before it.
    assert result.predictions["date"].min() > panel["date"].min()
    assert result.backtest.daily.index.is_monotonic_increasing


def test_metrics_and_report_are_populated(config: ExperimentConfig, panel: pd.DataFrame) -> None:
    result = run_experiment(config, panel)
    metrics = result.metrics()
    assert metrics["n_folds"] == len(result.folds)
    assert np.isfinite(metrics["strategy_sharpe"])
    assert np.isfinite(metrics["benchmark_sharpe"])
    report = result.report()
    assert "sharpe" in report and result.config.name in report
    assert result.importances and sum(result.importances.values()) == pytest.approx(1.0)


def test_continuous_labels_run_end_to_end(config: ExperimentConfig, panel: pd.DataFrame) -> None:
    regression = replace(
        config,
        label=replace(config.label, kind="continuous"),
        model=ModelConfig(name="ridge"),
    )
    result = run_experiment(regression, panel)
    assert result.predictions["score"].nunique() > 10
    # A regression score is not a probability, so the rank book must still work.
    assert result.backtest.positions.abs().sum(axis=1).max() > 0


def test_predictions_are_dated_after_their_training_fold(
    config: ExperimentConfig, panel: pd.DataFrame
) -> None:
    result = run_experiment(config, panel)
    for fold_label, group in result.predictions.groupby("fold"):
        test_start = pd.Timestamp(str(fold_label).split("_")[1])
        assert group["date"].min() >= test_start


def test_sweep_ranks_by_sharpe(config: ExperimentConfig, panel: pd.DataFrame) -> None:
    configs = [
        replace(config, name="a"),
        replace(config, name="b", backtest=replace(config.backtest, cost_bps=50.0)),
    ]
    table, results = sweep(configs, panel)
    assert list(table["name"]) == list(
        table.sort_values("strategy_sharpe", ascending=False)["name"]
    )
    assert len(results) == 2
    # Higher costs cannot improve net performance for identical signals.
    sharpes = table.set_index("name")["strategy_sharpe"]
    assert sharpes["a"] > sharpes["b"]


def test_experiment_without_folds_raises(config: ExperimentConfig) -> None:
    short = make_prices(days=400)
    with pytest.raises(ValueError, match=r"no walk-forward folds|every fold was skipped"):
        run_experiment(replace(config, splits=SplitConfig(train_years=6)), short)


def test_cli_run_writes_artifacts(tmp_path: Path, panel: pd.DataFrame) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    data = DataConfig(
        symbols=tuple(sorted(panel["symbol"].unique())),
        start="2015-01-01",
        end="2020-12-31",
        cache_dir=str(cache),
    )
    from ml_trading.data import _cache_path

    path = _cache_path(data)
    assert path is not None
    panel.to_parquet(path, index=False)

    config = ExperimentConfig(
        name="cli",
        data=data,
        features=FEATURES,
        splits=SPLITS,
        backtest=BacktestConfig(volatility_target=None),
    )
    config_path = tmp_path / "cli.yaml"
    config_path.write_text(json.dumps(config.to_dict()))  # YAML is a superset of JSON

    output = tmp_path / "artifacts"
    assert main(["run", "--config", str(config_path), "--output", str(output)]) == 0

    directory = output / "cli"
    assert {"daily.csv", "predictions.csv", "report.txt", "metrics.json", "config.json"} <= {
        entry.name for entry in directory.iterdir()
    }
    metrics = json.loads((directory / "metrics.json").read_text())
    assert "strategy_sharpe" in metrics


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--config", "x.yaml", "--log-level", "WARNING"],
        ["--log-level", "WARNING", "run", "--config", "x.yaml"],
    ],
)
def test_cli_accepts_log_level_on_either_side_of_the_subcommand(argv: list[str]) -> None:
    from ml_trading.cli import build_parser

    assert build_parser().parse_args(argv).log_level == "WARNING"

"""Command line entry point: ``ml-trading run|sweep|data``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ml_trading.config import ExperimentConfig
from ml_trading.data import coverage_report, load_prices
from ml_trading.metrics import deflated_sharpe_ratio
from ml_trading.pipeline import ExperimentResult, run_experiment, sweep

logger = logging.getLogger(__name__)

SWEEP_COLUMNS = [
    "name",
    "model",
    "strategy_sharpe",
    "strategy_annual_return",
    "strategy_max_drawdown",
    "strategy_sharpe_pvalue",
    "deflated_sharpe",
    "mean_fold_auc",
    "mean_fold_ic",
    "benchmark_sharpe",
]


def build_parser() -> argparse.ArgumentParser:
    # SUPPRESS keeps the subcommand's copy of --log-level from overwriting a value
    # the user passed before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(prog="ml-trading", description=__doc__, parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", parents=[common], help="run one walk-forward experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("artifacts"))
    run.add_argument("--refresh-data", action="store_true")
    run.add_argument("--mlflow", action="store_true", help="log params/metrics/artifacts to MLflow")
    run.add_argument("--mlflow-experiment", default="/Shared/ml_trading")

    sweep_parser = subparsers.add_parser(
        "sweep", parents=[common], help="run every config in a directory"
    )
    sweep_parser.add_argument("--config-dir", type=Path, required=True)
    sweep_parser.add_argument("--output", type=Path, default=Path("artifacts"))
    sweep_parser.add_argument("--refresh-data", action="store_true")
    sweep_parser.add_argument("--mlflow", action="store_true")
    sweep_parser.add_argument("--mlflow-experiment", default="/Shared/ml_trading")

    data = subparsers.add_parser("data", parents=[common], help="download and cache prices only")
    data.add_argument("--config", type=Path, required=True)
    data.add_argument("--refresh-data", action="store_true", default=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(args, "log_level", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "data":
        config = ExperimentConfig.from_yaml(args.config)
        prices = load_prices(config.data, refresh=args.refresh_data)
        report = coverage_report(prices, config.data.symbols)
        print(f"cached {len(prices):,} rows for {prices['symbol'].nunique()} symbols")
        missing = report[report["missing"]]
        if not missing.empty:
            print(f"no data for {len(missing)} requested symbols: {', '.join(missing.index)}")
        return 0

    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        prices = load_prices(config.data, refresh=args.refresh_data)
        result = run_experiment(config, prices)
        print(result.report())
        directory = args.output / config.name
        _write_artifacts(directory, result)
        if args.mlflow:
            _log_mlflow(result, directory, args.mlflow_experiment)
        return 0

    configs = [ExperimentConfig.from_yaml(path) for path in sorted(args.config_dir.glob("*.yaml"))]
    if not configs:
        raise SystemExit(f"no *.yaml configs found in {args.config_dir}")
    table, results = sweep(configs, refresh_data=args.refresh_data)

    # A sweep is a multiple-testing exercise: the best Sharpe of N tries is biased upward,
    # so report each row against the expected maximum of N zero-skill trials.
    table["deflated_sharpe"] = [
        deflated_sharpe_ratio(
            row["strategy_sharpe"],
            n_trials=len(configs),
            n_days=int(row["strategy_n_days"]),
            skew=row["strategy_skew"],
            kurtosis=row["strategy_kurtosis"] + 3.0,
        )
        for _, row in table.iterrows()
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "sweep.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(table[SWEEP_COLUMNS].to_string(index=False))
    for result in results:
        directory = args.output / result.config.name
        _write_artifacts(directory, result)
        if args.mlflow:
            _log_mlflow(result, directory, args.mlflow_experiment)
    return 0


def _write_artifacts(directory: Path, result: ExperimentResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result.backtest.daily.to_csv(directory / "daily.csv")
    result.predictions.to_csv(directory / "predictions.csv", index=False)
    (directory / "report.txt").write_text(result.report() + "\n")
    (directory / "metrics.json").write_text(json.dumps(result.metrics(), indent=2, default=float))
    (directory / "config.json").write_text(
        json.dumps(result.config.to_dict(), indent=2, default=str)
    )
    logger.info("wrote artifacts to %s", directory)


def _log_mlflow(result: ExperimentResult, directory: Path, experiment: str) -> None:
    from ml_trading.tracking import log_experiment

    log_experiment(result, artifacts_dir=directory, experiment_name=experiment)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

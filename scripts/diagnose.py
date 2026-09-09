"""Diagnose a single config beyond the headline Sharpe.

Reports the walk-forward experiment, the daily cross-sectional information
coefficient with its t-statistic (signal quality, independent of portfolio
construction), a transaction-cost sensitivity curve, and per-year returns
against the benchmark.

    python scripts/diagnose.py configs/logistic_h21.yaml
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ml_trading.backtest import run_backtest, score_matrix
from ml_trading.config import ExperimentConfig
from ml_trading.data import load_prices
from ml_trading.pipeline import ExperimentResult


def cross_sectional_ic(result: ExperimentResult) -> pd.Series:
    """Daily rank correlation between scores and demeaned forward returns."""
    scores = score_matrix(result.predictions)
    forward = result.predictions.pivot(
        index="date", columns="symbol", values="forward_return"
    ).sort_index()
    return scores.corrwith(forward.sub(forward.mean(axis=1), axis=0), axis=1, method="spearman")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--costs-bps", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)

    config = ExperimentConfig.from_yaml(args.config)
    prices = load_prices(config.data)
    from ml_trading.pipeline import run_experiment

    result = run_experiment(config, prices)
    print(result.report())

    ic = cross_sectional_ic(result).dropna()
    tstat = ic.mean() / ic.std() * np.sqrt(len(ic))
    print(f"\ndaily cross-sectional IC: mean={ic.mean():.4f} t={tstat:.2f}")

    print("\ncost sensitivity")
    scores = score_matrix(result.predictions)
    for cost in args.costs_bps:
        stats = run_backtest(
            scores,
            prices,
            replace(config.backtest, cost_bps=cost),
            horizon=config.label.horizon,
        ).stats
        print(
            f"  {cost:>5.1f}bps sharpe={stats.sharpe:6.2f} ann={stats.annual_return:7.2%} "
            f"dd={stats.max_drawdown:7.2%} turnover={stats.avg_turnover:.3f} "
            f"p={stats.sharpe_pvalue:.3f}"
        )

    daily = result.backtest.daily
    columns = ["strategy_return", "benchmark_return"]
    yearly = daily.groupby(daily.index.year)[columns].apply(
        lambda block: (1.0 + block).prod() - 1.0
    )
    print("\nper-year returns\n" + yearly.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

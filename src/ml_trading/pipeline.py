"""End-to-end walk-forward experiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml_trading.backtest import BacktestResult, run_backtest, score_matrix
from ml_trading.config import ExperimentConfig
from ml_trading.features import build_dataset, feature_columns
from ml_trading.metrics import information_coefficient
from ml_trading.models import build_model, feature_importances
from ml_trading.splits import Fold, walk_forward_folds

logger = logging.getLogger(__name__)


@dataclass
class FoldReport:
    label: str
    train_rows: int
    test_rows: int
    auc: float
    """AUC of the sign of the forward excess return; NaN for a regression label."""

    information_coefficient: float


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    predictions: pd.DataFrame
    backtest: BacktestResult
    folds: list[FoldReport]
    importances: dict[str, float] = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        out = self.backtest.summary()
        out["mean_fold_auc"] = float(np.nanmean([fold.auc for fold in self.folds]))
        out["mean_fold_ic"] = float(
            np.nanmean([fold.information_coefficient for fold in self.folds])
        )
        out["n_folds"] = float(len(self.folds))
        return out

    def report(self) -> str:
        stats = self.backtest.stats
        bench = self.backtest.benchmark_stats
        lines = [
            f"experiment: {self.config.name}",
            f"model: {self.config.model.name}  horizon: {self.config.label.horizon}d  "
            f"folds: {len(self.folds)}",
            f"out-of-sample window: {self.backtest.daily.index.min().date()} -> "
            f"{self.backtest.daily.index.max().date()}",
            "",
            f"{'metric':<22}{'strategy':>12}{'benchmark':>12}",
            f"{'annual return':<22}{stats.annual_return:>11.2%}{bench.annual_return:>12.2%}",
            f"{'annual volatility':<22}{stats.annual_volatility:>11.2%}"
            f"{bench.annual_volatility:>12.2%}",
            f"{'sharpe':<22}{stats.sharpe:>12.2f}{bench.sharpe:>12.2f}",
            f"{'max drawdown':<22}{stats.max_drawdown:>11.2%}{bench.max_drawdown:>12.2%}",
            f"{'hit rate (daily)':<22}{stats.hit_rate:>11.2%}{bench.hit_rate:>12.2%}",
            "",
            f"sharpe p-value (one-sided): {stats.sharpe_pvalue:.3f}",
            f"average daily turnover: {stats.avg_turnover:.3f}",
            f"mean fold AUC: {self.metrics()['mean_fold_auc']:.4f}   "
            f"mean fold IC: {self.metrics()['mean_fold_ic']:.4f}",
        ]
        if self.importances:
            top = sorted(self.importances.items(), key=lambda item: -item[1])[:10]
            lines.append("")
            lines.append("top features: " + ", ".join(f"{name} {value:.3f}" for name, value in top))
        return "\n".join(lines)


def run_experiment(
    config: ExperimentConfig,
    prices: pd.DataFrame,
    *,
    dataset: pd.DataFrame | None = None,
) -> ExperimentResult:
    """Fit and evaluate the configured strategy walk-forward over the whole sample."""
    if dataset is None:
        dataset = build_dataset(prices, config.features, config.label)

    labelled = dataset.dropna(subset=["label", "forward_return"]).reset_index(drop=True)
    columns = feature_columns(labelled)
    if not columns:
        raise ValueError("dataset contains no feature columns")

    folds = walk_forward_folds(labelled["date"], config.splits, horizon=config.label.horizon)
    if not folds:
        raise ValueError(
            "no walk-forward folds were produced; shorten splits.train_years "
            "or widen data.start/end"
        )

    predictions: list[pd.DataFrame] = []
    reports: list[FoldReport] = []
    importances: dict[str, list[float]] = {name: [] for name in columns}

    for fold in folds:
        report, fold_predictions, fold_importances = _run_fold(fold, labelled, columns, config)
        if fold_predictions is None:
            continue
        reports.append(report)
        predictions.append(fold_predictions)
        for name, value in fold_importances.items():
            importances[name].append(value)

    if not predictions:
        raise ValueError("every fold was skipped; check the data range and label configuration")

    all_predictions = pd.concat(predictions, ignore_index=True)
    # Overlapping folds can re-predict a date; keep the earliest model's view of it.
    all_predictions = all_predictions.drop_duplicates(subset=["date", "symbol"], keep="first")

    backtest = run_backtest(
        score_matrix(all_predictions),
        prices,
        config.backtest,
        horizon=config.label.horizon,
    )

    return ExperimentResult(
        config=config,
        predictions=all_predictions,
        backtest=backtest,
        folds=reports,
        importances={
            name: float(np.mean(values)) for name, values in importances.items() if values
        },
    )


def _run_fold(
    fold: Fold,
    labelled: pd.DataFrame,
    columns: list[str],
    config: ExperimentConfig,
) -> tuple[FoldReport, pd.DataFrame | None, dict[str, float]]:
    train = labelled.loc[fold.train_mask]
    test = labelled.loc[fold.test_mask]

    regression = config.label.kind == "continuous"
    if train.empty or test.empty or (not regression and train["label"].nunique() < 2):
        logger.warning("skipping %s: degenerate training block", fold.label)
        empty = FoldReport(fold.label, len(train), len(test), np.nan, np.nan)
        return empty, None, {}

    model = build_model(config.model, regression=regression, seed=config.seed)

    if regression:
        model.fit(train[columns], train["label"].astype(float))
        score = model.predict(test[columns])
    else:
        model.fit(train[columns], train["label"].astype(int))
        score = model.predict_proba(test[columns])[:, 1]

    frame = test.loc[:, ["date", "symbol", "forward_return"]].copy()
    if "forward_excess_return" in test.columns:
        frame["forward_excess_return"] = test["forward_excess_return"]
    frame["score"] = score
    frame["fold"] = fold.label

    truth = (test.get("forward_excess_return", test["forward_return"]) > 0).astype(int)
    target_column = (
        "forward_excess_return" if "forward_excess_return" in frame.columns else "forward_return"
    )

    report = FoldReport(
        label=fold.label,
        train_rows=len(train),
        test_rows=len(test),
        auc=float(roc_auc_score(truth, score)) if truth.nunique() > 1 else float("nan"),
        information_coefficient=information_coefficient(frame["score"], frame[target_column]),
    )
    logger.info(
        "%s train=%d test=%d auc=%.4f ic=%.4f",
        fold.label,
        len(train),
        len(test),
        report.auc,
        report.information_coefficient,
    )
    return report, frame, feature_importances(model, columns)


def sweep(
    configs: list[ExperimentConfig],
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, list[ExperimentResult]]:
    """Run several configurations over the same prices and rank them by net Sharpe."""
    results: list[ExperimentResult] = []
    rows: list[dict[str, Any]] = []
    datasets: dict[tuple[Any, Any], pd.DataFrame] = {}
    for config in configs:
        logger.info("running experiment %s", config.name)
        key = (config.features, config.label)
        if key not in datasets:
            datasets[key] = build_dataset(prices, config.features, config.label)
        result = run_experiment(config, prices, dataset=datasets[key])
        results.append(result)
        row: dict[str, Any] = {"name": config.name, "model": config.model.name}
        row.update(result.metrics())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("strategy_sharpe", ascending=False)
    return table.reset_index(drop=True), results

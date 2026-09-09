"""Portfolio construction and backtesting.

The original notebook multiplied a signal by the *same day's* return, which both
looks ahead and mismatches the 5-day label horizon. This module fixes both:

* positions are held for the full label horizon by averaging the last ``horizon``
  daily target books, so exactly ``1 / horizon`` of the portfolio turns over per day;
* positions are shifted forward by ``execution_lag`` days before they earn anything,
  so a signal computed from the close of ``t`` earns the return of ``t+1`` onwards;
* trading costs are charged on realised turnover, and leverage is set from *trailing*
  realised volatility only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ml_trading.metrics import TRADING_DAYS, PerformanceStats, performance_stats

if TYPE_CHECKING:
    from ml_trading.config import BacktestConfig

MAX_LEVERAGE = 3.0
VOL_LOOKBACK = 63


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    """Indexed by date: strategy/benchmark returns, turnover, leverage, equity curves."""

    positions: pd.DataFrame
    """Date x symbol matrix of the weights actually held."""

    stats: PerformanceStats
    benchmark_stats: PerformanceStats

    def summary(self) -> dict[str, float]:
        out = {f"strategy_{key}": value for key, value in self.stats.to_dict().items()}
        out.update(
            {f"benchmark_{key}": value for key, value in self.benchmark_stats.to_dict().items()}
        )
        return out


def signal_to_weights(
    scores: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Turn a date x symbol matrix of model scores into gross-constrained target weights.

    ``rank`` weighting only uses the within-day ordering of the scores, so it works for
    both a classifier's P(up) and a regressor's predicted return. ``threshold`` weighting
    compares the score against absolute cutoffs and therefore assumes a probability.
    """
    if config.weighting == "rank":
        raw = _rank_weights(scores, allow_short=config.allow_short)
    else:
        long_leg = (scores > config.long_threshold).astype(float)
        short_leg = (scores < config.short_threshold).astype(float)
        raw = long_leg - (short_leg if config.allow_short else 0.0)
    raw = raw.where(scores.notna(), 0.0)

    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    scale = config.max_gross_exposure / gross
    return raw.mul(scale, axis=0).fillna(0.0)


def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig,
    *,
    horizon: int,
) -> BacktestResult:
    """Backtest a date x symbol matrix of model scores against tidy ``prices``."""
    returns = daily_returns_matrix(prices).reindex(index=scores.index)
    returns = returns.reindex(columns=scores.columns)

    target = signal_to_weights(scores, config)
    # Hold each book for the label horizon: 1/horizon of the portfolio rolls each day.
    held = target.rolling(horizon, min_periods=1).mean()
    # Execution lag: a signal from the close of t can only earn returns from t+lag.
    held = held.shift(config.execution_lag).fillna(0.0)

    gross_return = (held * returns).sum(axis=1)

    leverage = _volatility_scalar(gross_return, config)
    positions = held.mul(leverage, axis=0)

    strategy_gross = (positions * returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = positions.iloc[0].abs().sum()
    cost = turnover * config.cost_bps / 10_000.0
    strategy_net = strategy_gross - cost

    benchmark = returns.mean(axis=1)

    daily = pd.DataFrame(
        {
            "strategy_return_gross": strategy_gross,
            "cost": cost,
            "strategy_return": strategy_net,
            "benchmark_return": benchmark,
            "turnover": turnover,
            "leverage": leverage,
            "gross_exposure": positions.abs().sum(axis=1),
            "net_exposure": positions.sum(axis=1),
        }
    ).dropna(subset=["strategy_return", "benchmark_return"])

    daily["strategy_equity"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["benchmark_equity"] = (1.0 + daily["benchmark_return"]).cumprod()

    return BacktestResult(
        daily=daily,
        positions=positions.loc[daily.index],
        stats=performance_stats(daily["strategy_return"], daily["turnover"]),
        benchmark_stats=performance_stats(daily["benchmark_return"]),
    )


def _rank_weights(scores: pd.DataFrame, *, allow_short: bool) -> pd.DataFrame:
    """Cross-sectional rank weights: long the top of each day's ranking, short the bottom.

    Ranking within the day rather than thresholding a raw probability makes the book
    insensitive to the model's calibration drifting between folds, and matches the
    market-excess label the model is trained on.
    """
    ranks = scores.rank(axis=1, pct=True)
    centred = ranks.sub(ranks.mean(axis=1), axis=0)
    if not allow_short:
        centred = centred.clip(lower=0.0)
    return centred


def daily_returns_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Date x symbol matrix of simple close-to-close returns."""
    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    return wide.pct_change(fill_method=None)


def score_matrix(predictions: pd.DataFrame, column: str = "score") -> pd.DataFrame:
    """Pivot tidy per-row predictions into the date x symbol matrix the backtest wants."""
    return predictions.pivot(index="date", columns="symbol", values=column).sort_index()


def _volatility_scalar(gross_return: pd.Series, config: BacktestConfig) -> pd.Series:
    """Leverage from *trailing* realised volatility, so it is knowable at trade time."""
    if config.volatility_target is None:
        return pd.Series(1.0, index=gross_return.index)

    realised = gross_return.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std() * np.sqrt(
        TRADING_DAYS
    )
    scalar = (config.volatility_target / realised.shift(1)).clip(upper=MAX_LEVERAGE)
    return scalar.fillna(1.0)

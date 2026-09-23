"""Performance and significance statistics for a daily return stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


@dataclass(frozen=True)
class PerformanceStats:
    n_days: int
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sharpe_pvalue: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    skew: float
    kurtosis: float
    avg_turnover: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def performance_stats(returns: pd.Series, turnover: pd.Series | None = None) -> PerformanceStats:
    """Summarise a series of daily *simple* returns."""
    clean = pd.Series(returns).dropna().astype(float)
    if clean.empty:
        raise ValueError("cannot compute performance statistics on an empty return series")

    mean = clean.mean()
    std = clean.std(ddof=1)
    sharpe = float(mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0

    equity = (1.0 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(clean) / TRADING_DAYS
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    max_drawdown = float((equity / equity.cummax() - 1.0).min())

    downside = clean[clean < 0].std(ddof=1)
    sortino = float(mean / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0

    return PerformanceStats(
        n_days=len(clean),
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=float(std * np.sqrt(TRADING_DAYS)),
        sharpe=sharpe,
        sharpe_pvalue=_sharpe_pvalue(clean),
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
        hit_rate=float((clean > 0).mean()),
        skew=float(stats.skew(clean)) if std > 0 else 0.0,
        kurtosis=float(stats.kurtosis(clean)) if std > 0 else 0.0,
        avg_turnover=float(pd.Series(turnover).dropna().mean()) if turnover is not None else 0.0,
    )


def _sharpe_pvalue(returns: pd.Series) -> float:
    """One-sided p-value for the mean daily return being positive."""
    if returns.std(ddof=1) == 0:
        return 1.0
    _, two_sided = stats.ttest_1samp(returns, 0.0)
    p = float(two_sided) / 2.0
    return p if returns.mean() > 0 else 1.0 - p


def deflated_sharpe_ratio(
    sharpe: float, n_trials: int, n_days: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Probability the observed Sharpe survives the multiple-testing correction.

    Implements Bailey & Lopez de Prado's deflated Sharpe ratio: after trying
    ``n_trials`` configurations, the best in-sample Sharpe is inflated by selection,
    so it must be compared against the expected maximum of ``n_trials`` draws of a
    zero-skill Sharpe rather than against zero.
    """
    if n_days < 2 or n_trials < 1:
        return float("nan")

    euler = 0.5772156649015329
    if n_trials == 1:
        expected_max = 0.0
    else:
        expected_max = (1 - euler) * stats.norm.ppf(1 - 1 / n_trials) + euler * stats.norm.ppf(
            1 - 1 / (n_trials * np.e)
        )

    daily_sharpe = sharpe / np.sqrt(TRADING_DAYS)
    expected_max_daily = expected_max / np.sqrt(n_days)
    denominator = np.sqrt(1 - skew * daily_sharpe + (kurtosis - 1) / 4 * daily_sharpe**2)
    if denominator <= 0:
        return float("nan")
    statistic = (daily_sharpe - expected_max_daily) * np.sqrt(n_days - 1) / denominator
    return float(stats.norm.cdf(statistic))


def information_coefficient(predictions: pd.Series, realised: pd.Series) -> float:
    """Spearman rank correlation between the signal and the realised forward return."""
    frame = pd.DataFrame({"p": predictions, "r": realised}).dropna()
    if len(frame) < 3 or frame["p"].nunique() < 2:
        return float("nan")
    return float(stats.spearmanr(frame["p"], frame["r"]).statistic)


def cross_sectional_ic(
    frame: pd.DataFrame,
    *,
    score: str,
    target: str,
    date: str = "date",
) -> pd.Series:
    """Spearman correlation between signal and forward return *within each date*.

    Pooling a panel into one correlation measures something else: a signal that only
    knows which days were good for everything scores well pooled while picking no
    winners on any given day, which is the only skill a cross-sectional book can trade.
    """
    grouped = frame[[score, target]].groupby(frame[date], sort=True)
    return grouped.apply(lambda block: information_coefficient(block[score], block[target]))


def mean_cross_sectional_ic(frame: pd.DataFrame, *, score: str, target: str) -> float:
    daily = cross_sectional_ic(frame, score=score, target=target)
    return float(np.nanmean(daily)) if len(daily) else float("nan")

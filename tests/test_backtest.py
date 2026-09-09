from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml_trading.backtest import (
    daily_returns_matrix,
    run_backtest,
    score_matrix,
    signal_to_weights,
)
from ml_trading.config import BacktestConfig
from tests.conftest import make_prices


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(cost_bps=0.0, volatility_target=None)


def _scores(prices: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    return pd.DataFrame(rng.random(wide.shape), index=wide.index, columns=wide.columns)


def test_rank_weights_are_dollar_neutral_and_gross_constrained(config: BacktestConfig) -> None:
    scores = pd.DataFrame(
        [[0.1, 0.4, 0.6, 0.9]],
        index=pd.to_datetime(["2020-01-02"]),
        columns=["A", "B", "C", "D"],
    )
    weights = signal_to_weights(scores, config)
    assert weights.sum(axis=1).abs().max() < 1e-12
    assert weights.abs().sum(axis=1).iloc[0] == pytest.approx(config.max_gross_exposure)
    assert weights.loc[:, "D"].iloc[0] > 0 > weights.loc[:, "A"].iloc[0]


def test_long_only_weights_never_go_short(config: BacktestConfig) -> None:
    scores = pd.DataFrame(
        [[0.1, 0.4, 0.6, 0.9]],
        index=pd.to_datetime(["2020-01-02"]),
        columns=["A", "B", "C", "D"],
    )
    weights = signal_to_weights(scores, replace(config, allow_short=False))
    assert (weights >= 0).all().all()
    assert weights.abs().sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_threshold_weighting_uses_absolute_cutoffs(config: BacktestConfig) -> None:
    scores = pd.DataFrame(
        [[0.10, 0.50, 0.60, 0.90]],
        index=pd.to_datetime(["2020-01-02"]),
        columns=["A", "B", "C", "D"],
    )
    weights = signal_to_weights(scores, replace(config, weighting="threshold"))
    assert weights.loc[:, "B"].iloc[0] == 0.0
    assert weights.loc[:, "A"].iloc[0] < 0
    assert weights.loc[:, ["C", "D"]].iloc[0].gt(0).all()


def test_execution_lag_means_a_signal_cannot_earn_its_own_day(config: BacktestConfig) -> None:
    """A perfectly clairvoyant score applied with lag>=1 must not capture same-day returns."""
    prices = make_prices(days=300)
    returns = daily_returns_matrix(prices)
    # Score = the return of the same day; with correct lagging this is worthless.
    scores = returns.fillna(0.0)

    lagged = run_backtest(scores, prices, replace(config, execution_lag=1), horizon=1)
    assert abs(lagged.stats.sharpe) < 3.0

    positions = signal_to_weights(scores, config).shift(1).fillna(0.0)
    pd.testing.assert_frame_equal(
        lagged.positions,
        positions.loc[lagged.positions.index, lagged.positions.columns],
        atol=1e-12,
    )


def test_holding_period_caps_daily_turnover(config: BacktestConfig) -> None:
    """Holding a book for `horizon` days should turn over about 1/horizon per day."""
    prices = make_prices(days=400)
    scores = _scores(prices)
    fast = run_backtest(scores, prices, config, horizon=1)
    slow = run_backtest(scores, prices, config, horizon=20)
    assert slow.stats.avg_turnover < fast.stats.avg_turnover / 3


def test_costs_reduce_returns_proportionally_to_turnover() -> None:
    prices = make_prices(days=400)
    scores = _scores(prices)
    free = run_backtest(
        scores, prices, BacktestConfig(cost_bps=0.0, volatility_target=None), horizon=5
    )
    charged = run_backtest(
        scores, prices, BacktestConfig(cost_bps=10.0, volatility_target=None), horizon=5
    )
    expected = free.daily["turnover"] * 10.0 / 10_000.0
    pd.testing.assert_series_equal(charged.daily["cost"], expected, check_names=False)
    assert charged.stats.annual_return < free.stats.annual_return


def test_volatility_target_uses_only_trailing_information() -> None:
    prices = make_prices(days=800)
    scores = _scores(prices)
    result = run_backtest(
        scores, prices, BacktestConfig(cost_bps=0.0, volatility_target=0.10), horizon=5
    )
    realised = result.daily["strategy_return"].std() * np.sqrt(252)
    assert 0.02 < realised < 0.30
    assert result.daily["leverage"].max() <= 3.0 + 1e-9


def test_equity_curve_compounds() -> None:
    prices = make_prices(days=300)
    result = run_backtest(
        _scores(prices), prices, BacktestConfig(cost_bps=0.0, volatility_target=None), horizon=5
    )
    expected = (1.0 + result.daily["strategy_return"]).cumprod()
    pd.testing.assert_series_equal(result.daily["strategy_equity"], expected, check_names=False)
    assert result.stats.total_return == pytest.approx(expected.iloc[-1] - 1.0)


def test_score_matrix_pivots_tidy_predictions() -> None:
    tidy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]),
            "symbol": ["A", "B", "A", "B"],
            "score": [0.1, 0.2, 0.3, 0.4],
        }
    )
    wide = score_matrix(tidy)
    assert list(wide.columns) == ["A", "B"]
    assert wide.loc[pd.Timestamp("2020-01-03"), "B"] == 0.4


def test_missing_scores_produce_no_position(config: BacktestConfig) -> None:
    scores = pd.DataFrame(
        [[0.1, np.nan, 0.9]],
        index=pd.to_datetime(["2020-01-02"]),
        columns=["A", "B", "C"],
    )
    weights = signal_to_weights(scores, config)
    assert weights.loc[:, "B"].iloc[0] == 0.0

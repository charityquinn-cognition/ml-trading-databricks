from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_trading.metrics import (
    TRADING_DAYS,
    deflated_sharpe_ratio,
    information_coefficient,
    performance_stats,
)


def test_sharpe_matches_the_closed_form() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 2000))
    stats = performance_stats(returns)
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert stats.sharpe == pytest.approx(expected)
    assert stats.annual_volatility == pytest.approx(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def test_total_return_compounds_rather_than_sums() -> None:
    returns = pd.Series([0.1] * 10)
    stats = performance_stats(returns)
    assert stats.total_return == pytest.approx(1.1**10 - 1.0)
    assert stats.total_return > returns.sum()


def test_max_drawdown_is_peak_to_trough() -> None:
    returns = pd.Series([0.5, -0.5, 0.0])
    stats = performance_stats(returns)
    assert stats.max_drawdown == pytest.approx(-0.5)


def test_constant_returns_have_zero_sharpe_and_neutral_pvalue() -> None:
    stats = performance_stats(pd.Series([0.0] * 100))
    assert stats.sharpe == 0.0
    assert stats.sharpe_pvalue == 1.0


def test_sharpe_pvalue_is_one_sided() -> None:
    rng = np.random.default_rng(1)
    good = performance_stats(pd.Series(rng.normal(0.002, 0.01, 2000)))
    bad = performance_stats(pd.Series(rng.normal(-0.002, 0.01, 2000)))
    assert good.sharpe_pvalue < 0.01
    assert bad.sharpe_pvalue > 0.99


def test_empty_returns_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        performance_stats(pd.Series([], dtype=float))


def test_turnover_is_averaged_when_supplied() -> None:
    stats = performance_stats(pd.Series([0.01, -0.01, 0.02]), pd.Series([0.2, 0.4, 0.6]))
    assert stats.avg_turnover == pytest.approx(0.4)


def test_deflated_sharpe_falls_as_more_configurations_are_tried() -> None:
    one = deflated_sharpe_ratio(1.0, n_trials=1, n_days=2520)
    many = deflated_sharpe_ratio(1.0, n_trials=200, n_days=2520)
    assert 0.0 <= many < one <= 1.0


def test_information_coefficient_is_rank_based() -> None:
    signal = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert information_coefficient(signal, signal**3) == pytest.approx(1.0)
    assert information_coefficient(signal, -signal) == pytest.approx(-1.0)


def test_information_coefficient_handles_degenerate_input() -> None:
    assert np.isnan(information_coefficient(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0])))
    assert np.isnan(information_coefficient(pd.Series([1.0]), pd.Series([1.0])))

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from ml_trading.config import DataConfig
from ml_trading.data import (
    _cache_path,
    coverage_report,
    load_prices,
    tidy_yfinance_frame,
    validate_prices,
)
from ml_trading.universes import UNIVERSES, resolve
from tests.conftest import make_prices


def test_validate_prices_sorts_and_types(prices: pd.DataFrame) -> None:
    shuffled = prices.sample(frac=1.0, random_state=0)
    out = validate_prices(shuffled)
    assert list(out.columns) == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert out.equals(out.sort_values(["symbol", "date"]).reset_index(drop=True))


def test_validate_prices_rejects_missing_columns(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_prices(prices.drop(columns=["volume"]))


def test_validate_prices_rejects_duplicates(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="duplicated"):
        validate_prices(pd.concat([prices, prices.head(1)]))


def test_validate_prices_rejects_non_positive_close(prices: pd.DataFrame) -> None:
    broken = prices.copy()
    broken.loc[0, "close"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(broken)


def test_tidy_yfinance_frame_skips_symbols_with_no_data() -> None:
    dates = pd.bdate_range("2020-01-01", periods=5)
    columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
    raw = pd.DataFrame(1.0, index=dates, columns=columns)
    raw.index.name = "Date"
    tidy = tidy_yfinance_frame(raw, ["AAA", "GONE"])
    assert set(tidy["symbol"]) == {"AAA"}


def test_coverage_report_flags_missing_symbols(prices: pd.DataFrame) -> None:
    report = coverage_report(prices, ["AAA", "BBB", "CCC", "DDD", "GONE"])
    assert bool(report.loc["GONE", "missing"])
    assert not bool(report.loc["AAA", "missing"])
    assert report.loc["AAA", "rows"] == (prices["symbol"] == "AAA").sum()


def test_cache_path_hashes_long_universes() -> None:
    short = _cache_path(DataConfig(symbols=("SPY", "QQQ")))
    assert short is not None and "QQQ-SPY" in short.name

    long = _cache_path(DataConfig(symbols=("@us_large_cap_plus_etfs",)))
    assert long is not None
    expected = len(resolve(("@us_large_cap_plus_etfs",)))
    assert long.name.startswith(f"universe-{expected}_")


def test_cache_path_disabled_without_cache_dir() -> None:
    assert _cache_path(DataConfig(cache_dir="")) is None


def test_load_prices_round_trips_through_the_cache(tmp_path) -> None:
    config = DataConfig(symbols=("AAA", "BBB"), cache_dir=str(tmp_path))
    frame = make_prices(symbols=("AAA", "BBB"), days=20)
    path = _cache_path(config)
    assert path is not None
    frame.to_parquet(path, index=False)
    loaded = load_prices(config)
    pd.testing.assert_frame_equal(loaded, validate_prices(frame))


def test_named_universes_resolve_to_unique_upper_case_tickers() -> None:
    for name in UNIVERSES:
        resolved = resolve((f"@{name}",))
        assert resolved, name
        assert len(set(resolved)) == len(resolved), name
        assert all(symbol == symbol.upper() for symbol in resolved), name


def test_resolve_passes_through_plain_symbols() -> None:
    assert resolve(("SPY", "QQQ")) == ("SPY", "QQQ")


def test_resolve_rejects_unknown_universe() -> None:
    with pytest.raises(ValueError, match="unknown universe"):
        resolve(("@not_a_universe",))


def test_data_config_is_hashable_for_dataset_caching() -> None:
    config = DataConfig()
    assert hash(config) == hash(replace(config))

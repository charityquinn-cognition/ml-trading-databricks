"""Price data loading.

The rest of the package only ever sees a *tidy* price frame: one row per
``(date, symbol)`` with columns ``open, high, low, close, volume`` where ``close``
is already split/dividend adjusted. Keeping this contract in one place is what
lets the same feature and backtest code run on yfinance downloads locally and on
Delta tables inside Databricks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ml_trading.universes import resolve

if TYPE_CHECKING:
    from ml_trading.config import DataConfig

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
TIDY_COLUMNS = ["date", "symbol", *PRICE_COLUMNS]


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Check the tidy price contract and return the frame sorted canonically."""
    missing = [column for column in TIDY_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"price frame is missing columns: {missing}")

    out = prices.loc[:, TIDY_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str)

    duplicated = out.duplicated(subset=["date", "symbol"]).sum()
    if duplicated:
        raise ValueError(f"price frame has {duplicated} duplicated (date, symbol) rows")

    non_positive = (out["close"] <= 0).sum()
    if non_positive:
        raise ValueError(f"price frame has {non_positive} non-positive close prices")

    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_prices(config: DataConfig, *, refresh: bool = False) -> pd.DataFrame:
    """Load adjusted daily bars, using an on-disk parquet cache when available."""
    cache_path = _cache_path(config)
    if cache_path is not None and cache_path.exists() and not refresh:
        logger.info("loading cached prices from %s", cache_path)
        return validate_prices(pd.read_parquet(cache_path))

    prices = download_prices(resolve(config.symbols), config.start, config.end)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path, index=False)
        logger.info("cached %d price rows to %s", len(prices), cache_path)
    return prices


def download_prices(symbols: tuple[str, ...] | list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily bars from Yahoo Finance into the tidy contract."""
    import yfinance as yf

    symbols = resolve(symbols)
    raw = yf.download(
        list(symbols),
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=True,
        threads=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no price data returned for {symbols} between {start} and {end}")
    return tidy_yfinance_frame(raw, symbols)


def tidy_yfinance_frame(raw: pd.DataFrame, symbols: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Reshape yfinance's wide, multi-indexed download into the tidy contract."""
    frames = []
    for symbol in symbols:
        frame: pd.DataFrame
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                logger.warning("no data returned for %s; skipping", symbol)
                continue
            frame = pd.DataFrame(raw[symbol]).copy()
        else:
            frame = raw.copy()
        frame = frame.reset_index().rename(columns=str.lower)
        frame["symbol"] = symbol
        frames.append(frame)

    if not frames:
        raise RuntimeError("yfinance returned no usable symbols")

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.dropna(subset=["close"])
    return validate_prices(tidy)


def coverage_report(prices: pd.DataFrame, symbols: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Per-symbol data coverage, so silently missing or truncated tickers are visible.

    Yahoo Finance quietly drops delisted or renamed tickers, which otherwise shows up
    only as a thinner cross-section on some dates.
    """
    requested = resolve(symbols)
    counts = prices.groupby("symbol")["date"].agg(["count", "min", "max"])
    counts = counts.reindex(list(requested))
    counts.columns = ["rows", "first_date", "last_date"]
    counts["rows"] = counts["rows"].fillna(0).astype(int)
    counts["missing"] = counts["rows"] == 0
    return counts.sort_values("rows")


def _cache_path(config: DataConfig) -> Path | None:
    if not config.cache_dir:
        return None
    symbols = resolve(config.symbols)
    key = f"{'-'.join(sorted(symbols))}_{config.start}_{config.end}"
    if len(key) > 120:
        # Long universes would blow past filename limits; hash them instead.
        import hashlib

        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        key = f"universe-{len(symbols)}_{config.start}_{config.end}_{digest}"
    return Path(config.cache_dir) / f"{key}.parquet"

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_prices(
    *,
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    days: int = 900,
    seed: int = 0,
    start: str = "2015-01-01",
) -> pd.DataFrame:
    """Synthetic but well-formed tidy OHLCV panel (business days, positive prices)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    frames = []
    for index, symbol in enumerate(symbols):
        drift = 0.0002 * (index + 1)
        returns = rng.normal(drift, 0.012, size=days)
        close = 100.0 * np.exp(np.cumsum(returns))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": close * (1 + rng.normal(0, 0.001, days)),
                    "high": close * (1 + abs(rng.normal(0, 0.004, days))),
                    "low": close * (1 - abs(rng.normal(0, 0.004, days))),
                    "close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, days).astype(float),
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()

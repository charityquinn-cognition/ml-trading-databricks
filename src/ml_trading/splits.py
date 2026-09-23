"""Purged, embargoed walk-forward splits for overlapping-label panel data.

A single 2020 train/test cut, as in the original notebooks, gives one noisy
observation of out-of-sample performance. Walk-forward re-fits the model on each
successive block so the reported results are the concatenation of many genuinely
out-of-sample periods.

Because a label at date ``t`` spans ``t .. t + horizon``, the last ``horizon`` days of
a training block overlap the first days of the test block. Those rows are *purged*
from training, following Lopez de Prado's Advances in Financial Machine Learning
(ch. 7).

The embargo in that chapter protects training rows that come *after* a test block from
serial correlation with it. Walk-forward only ever trains on history, so nothing needs
embargoing here and the default is zero: trimming the head of each test block would
punch a hole in the out-of-sample calendar every time a fold rolls, and the concatenated
scores the backtest consumes would then jump silently across the missing days. The
parameter stays available for callers who want the extra separation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ml_trading.config import years_to_days

if TYPE_CHECKING:
    from ml_trading.config import SplitConfig

TRADING_DAYS = 252


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def label(self) -> str:
        return f"fold{self.index:02d}_{self.test_start.date()}_{self.test_end.date()}"

    def __len__(self) -> int:
        return int(self.train_mask.sum())


def walk_forward_folds(
    dates: pd.Series,
    config: SplitConfig,
    *,
    horizon: int,
    embargo: int | None = None,
) -> list[Fold]:
    """Build the walk-forward folds for a panel indexed by ``dates`` (one entry per row)."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if dates.empty:
        return []

    embargo_days = 0 if embargo is None else embargo
    train_span = pd.Timedelta(days=years_to_days(config.train_years))
    test_span = pd.Timedelta(days=years_to_days(config.test_years))
    step = pd.Timedelta(days=years_to_days(config.step_years))
    purge = pd.Timedelta(days=_calendar_days(horizon))
    embargo_span = pd.Timedelta(days=_calendar_days(embargo_days) if embargo_days else 0)

    history_start = dates.min()
    last_date = dates.max()

    folds: list[Fold] = []
    test_start = history_start + train_span
    while test_start < last_date:
        test_end = min(test_start + test_span, last_date)
        train_start = history_start if config.expanding else test_start - train_span
        # Purge: a training label must be fully realised before the test block opens.
        train_end = test_start - purge

        train_mask = ((dates >= train_start) & (dates <= train_end)).to_numpy()
        test_open = test_start + embargo_span
        test_mask = ((dates >= test_open) & (dates <= test_end)).to_numpy()

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            folds.append(
                Fold(
                    index=len(folds),
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
            )
        test_start = test_start + step

    return folds


def _calendar_days(trading_days: int) -> int:
    """Convert a trading-day count to a conservative calendar-day count."""
    return int(np.ceil(trading_days * 365.25 / TRADING_DAYS)) + 1

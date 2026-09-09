from __future__ import annotations

import pandas as pd

from ml_trading.config import SplitConfig
from ml_trading.splits import walk_forward_folds


def _dates(years: int = 12) -> pd.Series:
    return pd.Series(pd.bdate_range("2010-01-01", periods=252 * years))


def test_folds_are_chronological_and_disjoint() -> None:
    dates = _dates()
    folds = walk_forward_folds(dates, SplitConfig(train_years=4), horizon=5)
    assert len(folds) >= 5
    for fold in folds:
        train = dates[fold.train_mask]
        test = dates[fold.test_mask]
        assert train.max() < test.min()
        assert not set(train.index) & set(test.index)
    assert [fold.test_start for fold in folds] == sorted(fold.test_start for fold in folds)


def test_training_labels_are_purged_by_the_label_horizon() -> None:
    """No training label may still be unrealised when the test block opens."""
    dates = _dates()
    horizon = 21
    for fold in walk_forward_folds(dates, SplitConfig(train_years=4), horizon=horizon):
        train_end = dates[fold.train_mask].max()
        gap = (fold.test_start - train_end).days
        assert gap >= horizon * 365.25 / 252


def test_embargo_removes_the_start_of_the_test_block() -> None:
    dates = _dates()
    config = SplitConfig(train_years=4)
    none = walk_forward_folds(dates, config, horizon=5, embargo=0)
    embargoed = walk_forward_folds(dates, config, horizon=5, embargo=40)
    assert embargoed[0].test_mask.sum() < none[0].test_mask.sum()


def test_expanding_grows_and_rolling_does_not() -> None:
    dates = _dates()
    expanding = walk_forward_folds(dates, SplitConfig(train_years=4, expanding=True), horizon=5)
    rolling = walk_forward_folds(dates, SplitConfig(train_years=4, expanding=False), horizon=5)
    assert [len(fold) for fold in expanding] == sorted(len(fold) for fold in expanding)
    assert all(fold.train_start == expanding[0].train_start for fold in expanding)
    assert rolling[-1].train_start > rolling[0].train_start


def test_empty_input_yields_no_folds() -> None:
    assert walk_forward_folds(pd.Series([], dtype="datetime64[ns]"), SplitConfig(), horizon=5) == []


def test_short_history_yields_no_folds() -> None:
    short = pd.Series(pd.bdate_range("2020-01-01", periods=50))
    assert walk_forward_folds(short, SplitConfig(train_years=6), horizon=5) == []

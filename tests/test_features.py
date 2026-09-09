from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml_trading.config import FeatureConfig, LabelConfig
from ml_trading.features import add_labels, build_dataset, build_features, feature_columns
from tests.conftest import make_prices


@pytest.fixture
def config() -> FeatureConfig:
    return FeatureConfig(
        momentum_windows=(5, 21),
        skip_momentum_windows=(63,),
        volatility_windows=(21,),
        ma_ratio_windows=(20,),
        beta_window=63,
        anomaly_signals=True,
        seasonality_years=1,
    )


def test_features_never_use_future_prices(prices: pd.DataFrame, config: FeatureConfig) -> None:
    """Truncating the panel must not change any feature value on the surviving dates.

    This is the single strongest guard against look-ahead: if a feature at t depended on
    a price after t, deleting the tail would move it.
    """
    cut = prices["date"].sort_values().unique()[-40]
    full = build_features(prices, config)
    truncated = build_features(prices[prices["date"] <= cut], config)

    columns = feature_columns(full)
    left = full[full["date"] <= cut].set_index(["symbol", "date"])[columns]
    right = truncated.set_index(["symbol", "date"])[columns]
    pd.testing.assert_frame_equal(left.sort_index(), right.sort_index(), atol=1e-12)


def test_feature_columns_exclude_identifiers_and_labels(
    prices: pd.DataFrame, config: FeatureConfig
) -> None:
    frame = add_labels(build_features(prices, config), LabelConfig(horizon=5))
    columns = feature_columns(frame)
    for reserved in ("date", "symbol", "close", "label", "forward_return"):
        assert reserved not in columns
    assert "momentum_21d" in columns
    assert "xs_rank_momentum_21d" in columns
    assert "xs_z_momentum_21d" in columns


def test_features_are_stationary_scale_free(prices: pd.DataFrame, config: FeatureConfig) -> None:
    """Doubling every price must leave the features unchanged (no raw price levels)."""
    scaled = prices.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * 2.0

    columns = feature_columns(build_features(prices, config))
    base = build_features(prices, config)[columns]
    other = build_features(scaled, config)[columns]
    pd.testing.assert_frame_equal(base, other, atol=1e-10)


def test_forward_return_matches_horizon(prices: pd.DataFrame) -> None:
    frame = add_labels(build_features(prices, FeatureConfig()), LabelConfig(horizon=5))
    one = frame[frame["symbol"] == "AAA"].sort_values("date").reset_index(drop=True)
    expected = np.log(one["close"].shift(-5) / one["close"])
    pd.testing.assert_series_equal(one["forward_return"], expected, check_names=False)
    assert one["forward_return"].tail(5).isna().all()


def test_binary_label_is_sign_of_excess_return(prices: pd.DataFrame) -> None:
    frame = add_labels(
        build_features(prices, FeatureConfig()),
        LabelConfig(horizon=3, kind="binary", excess_of_market=True),
    )
    valid = frame.dropna(subset=["label"])
    assert set(valid["label"].unique()) <= {0.0, 1.0}
    assert ((valid["forward_excess_return"] > 0).astype(float) == valid["label"]).all()
    # Excess returns are cross-sectionally demeaned, so roughly half the panel is long.
    assert 0.3 < valid["label"].mean() < 0.7


def test_continuous_label_is_winsorised_and_vol_normalised(prices: pd.DataFrame) -> None:
    config = LabelConfig(horizon=5, kind="continuous", volatility_normalize=True, winsorize=0.1)
    frame = add_labels(build_features(prices, FeatureConfig()), config)
    valid = frame.dropna(subset=["label"])
    assert not valid.empty
    # Normalising by trailing vol puts the target on a ~unit scale.
    assert valid["label"].abs().median() < 2.0
    daily_max = valid.groupby("date")["label"].max()
    daily_raw_max = valid.groupby("date")["forward_excess_return"].max()
    assert (daily_max.abs() < daily_raw_max.abs().max() * 1e6).all()


def test_build_dataset_drops_incomplete_rows(prices: pd.DataFrame, config: FeatureConfig) -> None:
    dataset = build_dataset(prices, config, LabelConfig(horizon=5))
    assert not dataset[feature_columns(dataset)].isna().any().any()
    assert dataset["date"].min() > prices["date"].min()


def test_anomaly_signals_are_opt_in(prices: pd.DataFrame, config: FeatureConfig) -> None:
    on = feature_columns(build_features(prices, config))
    off = feature_columns(build_features(prices, replace(config, anomaly_signals=False)))
    added = set(on) - set(off)
    assert {"high_52w_ratio", "residual_momentum", "amihud_illiquidity", "macd_norm"} <= added


def test_cross_sectional_zscore_is_within_date(config: FeatureConfig) -> None:
    frame = build_features(make_prices(days=400), config)
    valid = frame.dropna(subset=["xs_z_momentum_21d"])
    means = valid.groupby("date")["xs_z_momentum_21d"].mean()
    assert means.abs().max() < 1e-8

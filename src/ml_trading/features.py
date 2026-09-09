"""Feature engineering and labelling.

Two rules drive everything here:

1. **Stationarity.** The original notebooks fed raw price levels (``sma_20``,
   ``std_20``) to the model, so a model trained on 2015 prices saw completely
   out-of-distribution inputs in 2020. Every feature below is a ratio, a return,
   a rank or a rolling z-score, so its distribution is comparable across time
   and across symbols.
2. **No look-ahead.** Features at date ``t`` use only data up to and including
   ``t``; labels at date ``t`` describe the future and are therefore only ever
   used as a training target, never as an input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ml_trading.config import FeatureConfig, LabelConfig

TRADING_DAYS = 252


def build_features(prices: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Return the tidy price frame with one column per feature appended."""
    frame = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
    close = frame.groupby("symbol")["close"]

    frame["log_return_1d"] = close.transform(lambda s: np.log(s).diff())

    for window in config.momentum_windows:
        frame[f"momentum_{window}d"] = close.transform(lambda s, w=window: np.log(s).diff(w))

    for window in config.volatility_windows:
        frame[f"volatility_{window}d"] = frame.groupby("symbol")["log_return_1d"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).std() * np.sqrt(TRADING_DAYS)
        )

    for window in config.ma_ratio_windows:
        frame[f"ma_ratio_{window}d"] = close.transform(
            lambda s, w=window: s / s.rolling(w, min_periods=w).mean() - 1.0
        )

    for window in config.skip_momentum_windows:
        frame[f"momentum_{window}d_skip{config.skip_days}"] = close.transform(
            lambda s, w=window, k=config.skip_days: np.log(s.shift(k)).diff(w - k)
        )

    frame["rsi"] = close.transform(lambda s, w=config.rsi_window: _rsi(s, w))
    frame["bollinger_position"] = close.transform(_bollinger_position)
    frame["high_low_range"] = _high_low_range(frame)
    frame["gap_open"] = np.log(frame["open"] / frame.groupby("symbol")["close"].shift(1))

    frame["volume_zscore"] = frame.groupby("symbol")["volume"].transform(
        lambda s, w=config.volume_window: _zscore(np.log1p(s), w)
    )

    # Vol-normalised momentum: the same 6-month move means something different in a
    # 10%-vol regime than in a 40%-vol one.
    vol_reference = f"volatility_{max(config.volatility_windows)}d"
    if vol_reference in frame.columns:
        for window in config.momentum_windows:
            frame[f"momentum_{window}d_vol_adj"] = frame[f"momentum_{window}d"] / frame[
                vol_reference
            ].replace(0.0, np.nan)

    frame = _add_market_relative_features(frame, config)

    base_features = feature_columns(frame)
    extra: list[pd.DataFrame] = []

    if config.cross_sectional_rank:
        ranks = frame.groupby("date")[base_features].rank(pct=True) - 0.5
        ranks.columns = [f"xs_rank_{column}" for column in base_features]
        extra.append(ranks)

    if config.cross_sectional_zscore:
        grouped_features = frame.groupby("date")[base_features]
        means = grouped_features.transform("mean")
        stds = grouped_features.transform("std").replace(0.0, np.nan)
        zscores = (frame[base_features] - means) / stds
        zscores.columns = [f"xs_z_{column}" for column in base_features]
        extra.append(zscores.clip(-5.0, 5.0))

    return pd.concat([frame, *extra], axis=1) if extra else frame


def _add_market_relative_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Rolling beta to the equal-weight market, plus the residual (idiosyncratic) return."""
    window = config.beta_window
    market = frame.groupby("date")["log_return_1d"].transform("mean")

    def rolling_mean(series: pd.Series) -> pd.Series:
        return series.groupby(frame["symbol"]).transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )

    product = frame["log_return_1d"] * market
    covariance = rolling_mean(product) - rolling_mean(frame["log_return_1d"]) * rolling_mean(market)
    variance = rolling_mean(market**2) - rolling_mean(market) ** 2

    frame["beta"] = covariance / variance.replace(0.0, np.nan)
    frame["idiosyncratic_return_1d"] = frame["log_return_1d"] - frame["beta"] * market
    frame["idiosyncratic_volatility"] = frame.groupby("symbol")[
        "idiosyncratic_return_1d"
    ].transform(
        lambda s: (
            s.rolling(config.volume_window, min_periods=config.volume_window).std()
            * np.sqrt(TRADING_DAYS)
        )
    )
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns are everything that is neither an identifier, raw bar, nor label."""
    reserved = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "label",
        "forward_return",
        "forward_excess_return",
        "sample_weight",
    }
    return [column for column in frame.columns if column not in reserved]


def add_labels(frame: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Attach the forward return over ``horizon`` days and the model target.

    ``forward_return`` at date ``t`` is the log return from the close of ``t`` to the
    close of ``t + horizon``. It is deliberately *not* shifted: the backtest is what
    decides when the position may be taken, and it applies its own execution lag.
    """
    out = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    horizon = config.horizon

    out["forward_return"] = out.groupby("symbol")["close"].transform(
        lambda s: np.log(s.shift(-horizon) / s)
    )

    target = out["forward_return"]
    if config.excess_of_market:
        market = out.groupby("date")["forward_return"].transform("mean")
        out["forward_excess_return"] = out["forward_return"] - market
        target = out["forward_excess_return"]

    if config.kind == "binary":
        out["label"] = np.where(target.isna(), np.nan, (target > 0).astype(float))
        return out

    if config.volatility_normalize:
        # Trailing (knowable) volatility, scaled to the label horizon.
        trailing = out.groupby("symbol")["log_return_1d"].transform(
            lambda s: s.rolling(63, min_periods=63).std()
        ) * np.sqrt(horizon)
        target = target / trailing.replace(0.0, np.nan)

    if config.winsorize > 0:
        lower = target.groupby(out["date"]).transform(lambda s: s.quantile(config.winsorize))
        upper = target.groupby(out["date"]).transform(lambda s: s.quantile(1 - config.winsorize))
        target = target.clip(lower=lower, upper=upper)

    out["label"] = target
    return out


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def _bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """Where inside the Bollinger band the close sits: -1 at the lower band, +1 at the upper."""
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    return (close - mean) / (2.0 * std.replace(0.0, np.nan))


def _high_low_range(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.log(frame["high"] / frame["low"].replace(0.0, np.nan)), index=frame.index)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def build_dataset(
    prices: pd.DataFrame, features: FeatureConfig, label: LabelConfig
) -> pd.DataFrame:
    """Convenience wrapper: features then labels, with all-NaN feature rows dropped."""
    frame = add_labels(build_features(prices, features), label)
    columns = feature_columns(frame)
    return frame.dropna(subset=columns, how="any").reset_index(drop=True)

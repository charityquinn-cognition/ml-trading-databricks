"""Typed configuration for a walk-forward trading experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "XLE", "XLF", "XLK", "TLT")
    start: str = "2005-01-01"
    end: str = "2024-12-31"
    cache_dir: str = "data/cache"


@dataclass(frozen=True)
class FeatureConfig:
    """All windows are in trading days."""

    momentum_windows: tuple[int, ...] = (5, 21, 63, 126, 252)
    skip_momentum_windows: tuple[int, ...] = (126, 252)
    """Momentum measured over N days but skipping the most recent month.

    The classic cross-sectional momentum anomaly is 12-1: including the latest month
    mixes in short-term reversal, which pushes the other way.
    """

    skip_days: int = 21
    volatility_windows: tuple[int, ...] = (21, 63)
    ma_ratio_windows: tuple[int, ...] = (20, 50, 200)
    rsi_window: int = 14
    volume_window: int = 21
    cross_sectional_rank: bool = True
    cross_sectional_zscore: bool = True
    """Replace each feature with its within-date z-score so the pooled model sees
    comparable inputs across regimes."""

    beta_window: int = 126
    """Window for the rolling beta of each symbol to the equal-weight market."""

    anomaly_signals: bool = False
    """Add the signal families the cross-sectional literature keeps finding: 52-week-high
    proximity, residual momentum, MAX/idiosyncratic-skew lottery proxies, Amihud
    illiquidity and size, overnight-vs-intraday decomposition, annual seasonality,
    volatility-regime and trend (MACD) features. Gu/Kelly/Xiu report that the predictors
    that survive across ML methods are variations on momentum, liquidity and volatility,
    which is exactly what these add."""

    seasonality_years: int = 5
    """How many prior years of same-calendar-time returns the seasonality signal averages."""


@dataclass(frozen=True)
class LabelConfig:
    horizon: int = 5
    """Number of trading days the forward return spans."""

    kind: str = "binary"
    """``binary`` (sign of the forward excess return) or ``continuous``."""

    excess_of_market: bool = True
    """Label the return relative to the cross-sectional mean, not the raw return."""

    volatility_normalize: bool = True
    """For ``continuous`` labels, divide by trailing volatility so high-vol names do not
    dominate the regression loss."""

    winsorize: float = 0.01
    """For ``continuous`` labels, clip each day's target at this quantile from both tails."""

    sample_weight: str = "none"
    """Training-sample weighting: ``none``, ``return_attribution`` (weight each observation
    by the absolute forward target, so the fit is dominated by the moves that actually pay),
    ``time_decay`` (linear decay to ``time_decay_floor`` at the oldest training row), or
    ``both``. Overlapping horizon-h labels are not independent draws; weighting is the cheap
    part of the Lopez de Prado remedy, purging (already applied in splits.py) is the rest."""

    time_decay_floor: float = 0.25
    """Weight given to the oldest row of a training block under ``time_decay``."""


@dataclass(frozen=True)
class SplitConfig:
    train_years: float = 6.0
    test_years: float = 1.0
    step_years: float = 1.0
    expanding: bool = True
    """Grow the training window each fold instead of rolling a fixed-length one."""


@dataclass(frozen=True)
class ModelConfig:
    name: str = "logistic"
    """``logistic``/``ridge`` (linear), ``random_forest``, ``extra_trees``, ``gbm``,
    ``hist_gbm``, ``mlp``, or ``ensemble`` (soft-vote of a linear, a forest and a GBM)."""

    params: dict[str, Any] = field(default_factory=dict)

    tune: tuple[str, ...] = ()
    """Hyper-parameters to tune per fold on an inner purged split of the training block
    (e.g. ``("C",)`` or ``("num_leaves", "learning_rate")``). Empty means use the defaults.
    Tuning inside the fold keeps the outer test block genuinely out of sample."""

    tune_grid: dict[str, list[Any]] = field(default_factory=dict)
    """Candidate values per tuned parameter; falls back to :data:`models.DEFAULT_GRIDS`."""

    meta_label: bool = False
    """Fit a second model that predicts whether the primary model's bet is right, and use
    its confidence to size the bet (Lopez de Prado meta-labelling). The primary is fit on
    the front of the training block and the meta model on a purged tail of it, so the meta
    model never sees the primary's in-sample predictions."""


@dataclass(frozen=True)
class BacktestConfig:
    weighting: str = "rank"
    """``rank`` for a dollar-neutral cross-sectional book, ``threshold`` for +1/0/-1 bets."""

    long_threshold: float = 0.55
    short_threshold: float = 0.45
    allow_short: bool = True
    max_gross_exposure: float = 1.0
    """Sum of absolute position weights is scaled down to at most this value."""

    cost_bps: float = 5.0
    """Round-trip transaction cost in basis points of traded notional."""

    execution_lag: int = 1
    """Trading days between the signal timestamp and the first return earned."""

    volatility_target: float | None = 0.10
    """Annualised volatility target for the portfolio, or ``None`` to disable."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "baseline"
    seed: int = 7
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        if self.label.horizon < 1:
            raise ValueError("label.horizon must be >= 1")
        if min(self.splits.train_years, self.splits.test_years, self.splits.step_years) <= 0:
            # A non-positive step never advances the walk-forward window: the fold loop
            # would run forever rather than fail.
            raise ValueError("splits.train_years, test_years and step_years must all be > 0")
        if self.backtest.execution_lag < 1:
            raise ValueError("backtest.execution_lag must be >= 1 to avoid look-ahead")
        if self.backtest.short_threshold > self.backtest.long_threshold:
            raise ValueError("short_threshold must not exceed long_threshold")
        if self.label.kind not in {"binary", "continuous"}:
            raise ValueError(f"unknown label kind: {self.label.kind}")
        if self.backtest.weighting not in {"rank", "threshold"}:
            raise ValueError(f"unknown weighting scheme: {self.backtest.weighting}")
        if self.label.sample_weight not in {"none", "return_attribution", "time_decay", "both"}:
            raise ValueError(f"unknown sample weighting: {self.label.sample_weight}")
        if self.model.meta_label and self.label.kind != "binary":
            raise ValueError("meta-labelling needs a binary primary label")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_params(self) -> dict[str, Any]:
        """Flatten the config into ``section.key`` pairs suitable for MLflow logging."""
        flat: dict[str, Any] = {"name": self.name, "seed": self.seed}
        for section, value in self.to_dict().items():
            if isinstance(value, dict):
                for key, inner in value.items():
                    flat[f"{section}.{key}"] = inner
        return flat

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        section_types = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in section_types:
                raise ValueError(f"unknown config key: {key}")
            if isinstance(value, dict):
                kwargs[key] = _SECTIONS[key](**_tuplify(value))
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)


_SECTIONS: dict[str, Any] = {
    "data": DataConfig,
    "features": FeatureConfig,
    "label": LabelConfig,
    "splits": SplitConfig,
    "model": ModelConfig,
    "backtest": BacktestConfig,
}


def _tuplify(section: dict[str, Any]) -> dict[str, Any]:
    """YAML gives lists; the dataclasses are frozen and want tuples."""
    return {
        key: tuple(value) if isinstance(value, list) else value for key, value in section.items()
    }

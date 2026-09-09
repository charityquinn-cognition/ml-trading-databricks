"""Fitting machinery for a single walk-forward fold.

Everything here exists because a fold is not an i.i.d. training set:

* horizon-``h`` labels overlap, so consecutive rows carry nearly the same information -
  :func:`sample_weights` down-weights that redundancy and lets the fit concentrate on the
  observations that actually move money;
* hyper-parameters chosen on the test block are not out of sample - :func:`tuned_model`
  therefore selects them on a *purged inner split of the training block only*;
* the direction of a bet and whether that bet is worth taking are different questions -
  :func:`fit_meta_labelled` fits a second model for the second question and uses it to
  size the first (Lopez de Prado's meta-labelling).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_trading.metrics import information_coefficient
from ml_trading.models import build_model, parameter_grid, supports_sample_weight

if TYPE_CHECKING:
    from ml_trading.config import ExperimentConfig, LabelConfig

logger = logging.getLogger(__name__)

INNER_VALIDATION_FRACTION = 0.25
"""Share of the *training* block's dates held out to choose hyper-parameters."""

META_PRIMARY_FRACTION = 0.7
"""Share of the training block used to fit the primary model before meta-labelling."""


def sample_weights(train: pd.DataFrame, config: LabelConfig) -> np.ndarray | None:
    """Per-row training weights, or ``None`` when the config asks for a plain fit."""
    scheme = config.sample_weight
    if scheme == "none":
        return None

    weights = np.ones(len(train), dtype=float)
    if scheme in {"return_attribution", "both"}:
        target = train.get("forward_excess_return", train["forward_return"]).abs()
        weights *= (target / target.mean()).fillna(1.0).to_numpy()
    if scheme in {"time_decay", "both"}:
        age = _date_fraction(train["date"])
        weights *= config.time_decay_floor + (1.0 - config.time_decay_floor) * age
    total = weights.sum()
    return weights * (len(weights) / total) if total > 0 else None


def _date_fraction(dates: pd.Series) -> np.ndarray:
    """0.0 at the oldest date in the block, 1.0 at the newest."""
    ordinal = dates.rank(method="dense")
    span = float(ordinal.max() - ordinal.min())
    if span <= 0:
        return np.ones(len(dates))
    return ((ordinal - ordinal.min()) / span).to_numpy()


def fit_fold_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    config: ExperimentConfig,
) -> tuple[np.ndarray, Pipeline, dict[str, Any]]:
    """Fit the fold's model(s) on ``train`` and score ``test``.

    Returns the test scores, the fitted primary pipeline (for feature importances) and
    the hyper-parameters that were actually used.
    """
    regression = config.label.kind == "continuous"
    overrides = tuned_parameters(train, columns, config)

    if config.model.meta_label:
        scores, model = fit_meta_labelled(train, test, columns, config, overrides)
        return scores, model, overrides

    model = build_model(config.model, regression=regression, seed=config.seed, overrides=overrides)
    _fit(model, train, columns, config)
    return predict_scores(model, test[columns], regression=regression), model, overrides


def tuned_parameters(
    train: pd.DataFrame,
    columns: list[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Choose hyper-parameters on a purged inner split of the training block."""
    candidates = parameter_grid(config.model)
    if len(candidates) <= 1:
        return candidates[0] if candidates else {}

    inner_train, inner_validation = purged_split(
        train, 1.0 - INNER_VALIDATION_FRACTION, horizon=config.label.horizon
    )
    if inner_train.empty or inner_validation.empty:
        logger.warning("inner split is degenerate; falling back to default parameters")
        return {}

    regression = config.label.kind == "continuous"
    target_column = _target_column(inner_validation)
    best: dict[str, Any] = {}
    best_score = -np.inf
    for candidate in candidates:
        model = build_model(
            config.model, regression=regression, seed=config.seed, overrides=candidate
        )
        _fit(model, inner_train, columns, config)
        scores = predict_scores(model, inner_validation[columns], regression=regression)
        # Rank correlation with the forward target, which is what the book actually trades,
        # rather than log-loss on a label that is 51% noise.
        score = information_coefficient(
            pd.Series(scores, index=inner_validation.index), inner_validation[target_column]
        )
        if np.isfinite(score) and score > best_score:
            best_score, best = score, candidate
    logger.info("tuned %s -> %s (inner ic=%.4f)", config.model.name, best, best_score)
    return best


def fit_meta_labelled(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    config: ExperimentConfig,
    overrides: dict[str, Any],
) -> tuple[np.ndarray, Pipeline]:
    """Primary model picks the side; a secondary model sizes it by P(the side is right).

    The secondary model is trained on a purged tail of the training block, on which the
    primary's predictions are genuinely out of sample - training it on the primary's
    in-sample fit would just teach it that the primary is always right.
    """
    primary_block, meta_block = purged_split(
        train, META_PRIMARY_FRACTION, horizon=config.label.horizon
    )
    if primary_block.empty or meta_block.empty:
        logger.warning("training block too short for meta-labelling; using the primary alone")
        model = build_model(config.model, seed=config.seed, overrides=overrides)
        _fit(model, train, columns, config)
        return predict_scores(model, test[columns], regression=False), model

    staged = build_model(config.model, seed=config.seed, overrides=overrides)
    _fit(staged, primary_block, columns, config)
    meta_scores = predict_scores(staged, meta_block[columns], regression=False)

    meta_features = _with_primary(meta_block[columns], meta_scores)
    was_right = ((meta_scores > 0.5).astype(float) == meta_block["label"].to_numpy()).astype(int)
    meta_model = build_model(config.model, seed=config.seed + 1)
    if was_right.min() == was_right.max():
        logger.warning("meta labels are constant; using the primary alone")
        return predict_scores(staged, test[columns], regression=False), staged
    meta_model.fit(meta_features, was_right)

    # Refit the primary on the whole training block now that the meta model is calibrated.
    primary = build_model(config.model, seed=config.seed, overrides=overrides)
    _fit(primary, train, columns, config)
    side = predict_scores(primary, test[columns], regression=False)
    confidence = predict_scores(meta_model, _with_primary(test[columns], side), regression=False)
    return 0.5 + (side - 0.5) * confidence, primary


def purged_split(
    frame: pd.DataFrame, front_fraction: float, *, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a block in two by date, dropping ``horizon`` days of overlapping labels.

    The front part's last labels reach ``horizon`` days into the back part, so those rows
    are dropped rather than allowed to leak across the boundary.
    """
    dates = np.sort(frame["date"].unique())
    if len(dates) < 3:
        return frame.iloc[:0], frame.iloc[:0]
    cut = dates[min(int(len(dates) * front_fraction), len(dates) - 1)]
    front = frame[frame["date"] <= cut]
    if front.empty:
        return frame.iloc[:0], frame.iloc[:0]
    purge_until = front["date"].max() + pd.Timedelta(days=_calendar_days(horizon))
    back = frame[frame["date"] > purge_until]
    return front, back


def _calendar_days(trading_days: int) -> int:
    """Trading days padded to calendar days, so the purge cannot be defeated by weekends."""
    return int(np.ceil(trading_days * 7 / 5)) + 1


def predict_scores(model: Pipeline, features: pd.DataFrame, *, regression: bool) -> np.ndarray:
    scores = model.predict(features) if regression else model.predict_proba(features)[:, 1]
    return np.asarray(scores, dtype=float)


def _with_primary(features: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = features.copy()
    out["primary_score"] = scores
    return out


def _target_column(frame: pd.DataFrame) -> str:
    return "forward_excess_return" if "forward_excess_return" in frame.columns else "forward_return"


def _fit(
    model: Pipeline, train: pd.DataFrame, columns: list[str], config: ExperimentConfig
) -> None:
    target = (
        train["label"].astype(float)
        if config.label.kind == "continuous"
        else train["label"].astype(int)
    )
    weights = sample_weights(train, config.label) if supports_sample_weight(config.model) else None
    if weights is None:
        model.fit(train[columns], target)
    else:
        model.fit(train[columns], target, model__sample_weight=weights)

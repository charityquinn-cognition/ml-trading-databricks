# ML Trading on Databricks

A reproducible research harness for cross-sectional equity/ETF trading signals: a tested
Python package (`src/ml_trading`) with a CLI and YAML configs, plus thin Databricks
notebooks that wire it to Delta and MLflow.

The point of the harness is to make it *hard to fool yourself*. Every result below is
out-of-sample across many walk-forward folds, earned at least one day after the signal,
and net of transaction costs.

## Headline result: no reliable edge yet

`ml-trading sweep --config-dir configs` on 158 current US large caps and liquid ETFs,
2005-2024, 13 annual walk-forward folds (out-of-sample 2012-2024):

| config | net Sharpe | ann. return | max DD | p-value | deflated Sharpe | mean fold IC | benchmark Sharpe |
|---|---|---|---|---|---|---|---|
| `logistic_h21` | 0.28 | 2.2% | -24.8% | 0.17 | 0.46 | 0.024 | 1.05 |
| `gbm_h21` | 0.21 | 1.4% | -15.5% | 0.24 | 0.37 | 0.017 | 1.05 |
| `baseline_logistic` (5d) | 0.02 | -0.3% | -37.3% | 0.47 | 0.17 | 0.026 | 1.01 |
| `ridge_h21_continuous` | -0.11 | -1.4% | -36.6% | 0.64 | 0.08 | 0.008 | 1.05 |

Read that as a **negative result**, not a strategy:

- The models do carry a small amount of real cross-sectional information: for the best
  config the daily cross-sectional IC averages 0.024 with a t-statistic of 7.2. That is a
  genuine signal - it is simply too weak to pay for its own trading.
- Net Sharpe is indistinguishable from zero (one-sided p ≈ 0.17 for the best config), and
  the *deflated* Sharpe - which accounts for having tried several configurations - is 0.46,
  i.e. better than even odds that the best number here is selection noise.
- Every config is beaten by simply holding the equal-weight universe (Sharpe ≈ 1.05),
  which itself is flattered by survivorship bias (see caveats).
- Costs decide the outcome. Sweeping `backtest.cost_bps` for `logistic_h21` at ~0.11 daily
  turnover: 0bps → 0.43, 2bps → 0.36, 5bps → 0.27, 10bps → 0.12. Any conclusion drawn from
  a costless backtest is meaningless.

The honest summary: **daily price/volume features on a liquid universe do not produce a
tradeable edge here.** Getting one plausibly needs a different information set
(fundamentals, estimate revisions, flows, intraday microstructure) rather than more
hyperparameter search on these features.

## What changed from the original notebooks

The original three notebooks trained a Spark ML logistic regression on `sma_20`,
`std_20` and `daily_return` for SPY/AAPL/MSFT and reported a Sharpe from a single
2020 train/test cut. Four problems made that number meaningless:

| problem | fix |
|---|---|
| Positions multiplied the *same day's* return, so the signal traded on information it could not have had, against a label that described the *next 5 days* | positions are held for the full label horizon and shifted by `backtest.execution_lag` (≥1 day, enforced in config validation) |
| Raw price levels as features: a model fitted on 2015 prices extrapolates in 2020 | every feature is a return, ratio, rank or z-score - `build_features` output is invariant to rescaling prices (tested) |
| One arbitrary train/test date, with training labels overlapping the test window | purged, embargoed walk-forward folds (`splits.py`), 13 refits |
| No costs, no turnover, no benchmark, PnL summed instead of compounded | turnover-based costs, volatility targeting with a leverage cap, compounded equity curves, equal-weight benchmark, Sharpe p-values and deflated Sharpe |

## Layout

```
src/ml_trading/
  config.py      typed, validated experiment config (YAML-backed)
  data.py        tidy price contract: one row per (date, symbol), adjusted OHLCV + parquet cache
  universes.py   named universes (@etfs, @us_large_cap, @us_large_cap_plus_etfs)
  features.py    stationary features and horizon-aware labels
  splits.py      purged/embargoed walk-forward folds
  models.py      sklearn pipelines: logistic / ridge / random forest / LightGBM
  backtest.py    scores -> weights -> costed, vol-targeted portfolio
  metrics.py     performance stats, Sharpe p-value, deflated Sharpe, IC
  pipeline.py    walk-forward experiment + sweep
  cli.py         ml-trading data|run|sweep
  spark_io.py    Delta <-> tidy pandas (Databricks only)
  tracking.py    MLflow logging (optional)
configs/         one YAML per experiment
scripts/         diagnose.py: IC t-stats, cost sensitivity, per-year returns
tests/           unit tests, including explicit look-ahead guards
```

## Quickstart

```bash
pip install -e ".[gbm,plots,dev]"

ml-trading data  --config configs/logistic_h21.yaml    # download + cache prices
ml-trading run   --config configs/logistic_h21.yaml    # one walk-forward experiment
ml-trading sweep --config-dir configs                  # all configs, ranked, deflated
```

Artifacts land in `artifacts/<name>/`: `report.txt`, `metrics.json`, `daily.csv`
(equity curves, turnover, leverage), `predictions.csv`, `config.json`.

Add `--mlflow` to either `run` or `sweep` to log params, metrics, per-fold diagnostics
and artifacts to MLflow.

To look past the headline Sharpe at whether a config has any signal at all:

```bash
python scripts/diagnose.py configs/logistic_h21.yaml
```

## Databricks

The notebooks are thin wrappers; the logic they call is the same code the tests cover.

1. `ml_trading_01_data_and_features.ipynb` - load prices, build the dataset, write
   `market.prices` and `market.features_labeled`.
2. `ml_trading_02_model_and_backtest.ipynb` - run the walk-forward experiment, plot the
   net equity curve and turnover, write `market.predictions`, log to MLflow.
3. `experiments/factor_models_mlflow.py.ipynb` - sweep every config and log each run.

Spark is used only for storage (`spark_io.py`). A daily panel of a few hundred symbols is
tens of megabytes, so the modelling itself is plain pandas/sklearn - which is also what
makes it unit-testable off-cluster.

## Caveats you must not ignore

- **Survivorship bias.** `universes.py` holds *today's* index members. Names that were
  delisted or dropped never appear, which flatters both the strategy and (especially) the
  equal-weight benchmark. Point-in-time constituents are required before any of these
  numbers can be trusted as an absolute level; the relative comparisons are more robust.
- **Cost model is a single spread parameter.** No market impact, no borrow cost for the
  short leg, no financing. The short book is assumed freely available.
- **Multiple testing.** Every configuration tried against the same 2012-2024 out-of-sample
  window erodes what "out-of-sample" means. `deflated_sharpe_ratio` in `metrics.py` puts a
  number on that; use it, and keep a genuinely untouched holdout before trading anything.
- **Yahoo Finance data.** Free daily bars with occasional gaps and silently missing
  tickers; `coverage_report` in `data.py` surfaces the latter.

## Development

```bash
pytest -q
ruff check src tests scripts && ruff format --check src tests scripts
mypy
```

CI runs the same three commands on Python 3.10 and 3.12.

# ML Trading on Databricks

A reproducible research harness for cross-sectional equity/ETF trading signals: a tested
Python package (`src/ml_trading`) with a CLI and YAML configs, plus thin Databricks
notebooks that wire it to Delta and MLflow.

The point of the harness is to make it *hard to fool yourself*. Every result below is
out-of-sample across many walk-forward folds, earned at least one day after the signal,
and net of transaction costs.

## Headline result: a modest edge, and it comes from the features

`ml-trading sweep --config-dir configs` on 158 current US large caps and liquid ETFs,
2005-2024, 13 annual walk-forward folds (out-of-sample 2012-2024), 5bps costs, 1-day
execution lag. `_anomaly` configs add the literature signal block
(`features.anomaly_signals`); the rest are the original momentum/volatility features:

| config | net Sharpe | ann. return | max DD | p-value | deflated Sharpe | mean fold IC |
|---|---|---|---|---|---|---|
| `extra_trees_anomaly` | 0.81 | 7.2% | -18.2% | 0.003 | 0.86 | 0.043 |
| `gbm_anomaly_tuned` | 0.71 | 5.9% | -18.7% | 0.008 | 0.78 | 0.038 |
| `ensemble_anomaly` | 0.63 | 5.3% | -10.8% | 0.016 | 0.69 | 0.039 |
| `gbm_anomaly` | 0.60 | 4.8% | -11.5% | 0.020 | 0.65 | 0.034 |
| `gbm_anomaly_meta` | 0.59 | 4.7% | -11.7% | 0.022 | 0.63 | 0.034 |
| `logistic_anomaly` | 0.57 | 4.7% | -14.0% | 0.026 | 0.61 | 0.035 |
| `gbm_anomaly_weighted` | 0.32 | 2.3% | -18.3% | 0.138 | 0.28 | 0.025 |
| `logistic_h21` | 0.27 | 2.1% | -24.8% | 0.174 | 0.23 | 0.024 |
| `gbm_h21` | 0.21 | 1.4% | -15.5% | 0.239 | 0.17 | 0.017 |
| `baseline_logistic` (5d) | 0.03 | -0.2% | -37.3% | 0.464 | 0.06 | 0.026 |
| `mlp_anomaly` | -0.06 | -0.8% | -25.8% | 0.588 | 0.03 | 0.005 |
| `ridge_h21_continuous` | -0.11 | -1.4% | -36.6% | 0.647 | 0.02 | 0.008 |

Benchmark (equal-weight in the same universe, rebalanced daily and costlessly) is
Sharpe 1.00, +15.9% a year, -30.4% max drawdown.

What the table supports, and what it does not:

- **The features moved the needle, the estimators mostly did not.** Every family gains
  ~0.3 Sharpe from the anomaly block; swapping logistic for boosted trees is worth a
  further ~0.1-0.25. That ordering matches the asset-pricing ML literature (Gu, Kelly &
  Xiu): nonlinearity helps, but only once the predictors are there.
- **The best config survives its own multiple testing.** Twelve configurations were run;
  the deflated Sharpe for `extra_trees_anomaly` is 0.86, i.e. ~86% probability the Sharpe
  is genuinely positive after adjusting for the number of trials.
- **It survives costs.** At 0.071 daily turnover: 0bps → 0.91, 5bps → 0.81, 10bps → 0.71,
  20bps → 0.51. Daily cross-sectional IC is 0.043 (t = 14.3).
- **It is a diversifier, not a replacement for the index.** The long/short book earns 7.2%
  a year at 9% volatility against 15.9% at 16% for holding the universe. Its worst year
  is -2.6% (2020) and it made +12.5% in 2022 while the universe lost 4.9%, so the case for
  it is low correlation, not standalone return.
- **The edge is mostly slow characteristic tilts.** The top features are cross-sectional
  ranks of Amihud illiquidity, dollar volume, beta and volatility - i.e. the book is
  substantially harvesting the illiquidity, low-beta and low-volatility premia rather than
  predicting anything new. Turnover of 0.071 (≈14-day holding) says the same thing. A fair
  next test is whether it survives *neutralising* those exposures; until then, treat it as
  a factor portfolio, not alpha.
- **Two of the more elaborate ideas did not work here.** Sample weighting by
  |forward return| and recency halved the Sharpe (0.60 → 0.32) - the "redundant"
  overlapping rows were carrying signal, not noise. Meta-labelling was a wash (0.60 →
  0.59), and the MLP failed outright. They are kept in the codebase because the negative
  result is the useful part.

Caveat that outranks all of the above: the universe is *current* large caps, so the
survivorship bias flatters both the strategy and its benchmark, and nothing here has been
traded.

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
  features.py    stationary features, the anomaly signal block, horizon-aware labels
  splits.py      purged/embargoed walk-forward folds
  models.py      sklearn pipelines: logistic/ridge, forests, LightGBM, HistGB, MLP, ensemble
  training.py    per-fold fitting: sample weights, purged inner-fold tuning, meta-labelling
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
- **Factor exposure is not alpha.** The winning config's importances are dominated by
  liquidity, beta and volatility ranks, which are compensated risk exposures with decades
  of published history. Nothing here separates "predicts returns" from "is short liquidity
  and beta"; a factor-neutralised rerun is the missing test.
- **Yahoo Finance data.** Free daily bars with occasional gaps and silently missing
  tickers; `coverage_report` in `data.py` surfaces the latter.

## Development

```bash
pytest -q
ruff check src tests scripts && ruff format --check src tests scripts
mypy
```

CI runs the same three commands on Python 3.10 and 3.12.

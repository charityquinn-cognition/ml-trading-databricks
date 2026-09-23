"""Databricks/Delta adapters.

The research code is deliberately plain pandas: a daily panel of a few hundred
symbols is tens of megabytes, so Spark buys nothing for the modelling itself and
costs testability. Spark is still the right place for *storage*, so this module is
the only boundary between Delta tables and the tidy price contract in
:mod:`ml_trading.data`.

Nothing here imports pyspark at module scope, so the package stays importable off-cluster.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from ml_trading.data import TIDY_COLUMNS, validate_prices

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

PRICES_TABLE = "market.prices"
FEATURES_TABLE = "market.features_labeled"
PREDICTIONS_TABLE = "market.predictions"


def read_prices(spark: SparkSession, table: str = PRICES_TABLE) -> pd.DataFrame:
    """Read a Delta price table into the tidy pandas contract.

    Column names are lower-cased first, so the legacy ``Date``/``Close`` schema written
    by the original notebooks is accepted unchanged.
    """
    sdf = spark.table(table)
    renamed = sdf.toDF(*[column.lower() for column in sdf.columns])
    frame = renamed.select(*TIDY_COLUMNS).toPandas()
    return validate_prices(frame)


def write_table(
    spark: SparkSession,
    frame: pd.DataFrame,
    table: str,
    *,
    mode: str = "overwrite",
    partition_by: str | None = None,
) -> None:
    """Write a pandas frame back to Delta, creating the database if needed."""
    database = table.split(".")[0] if "." in table else None
    if database:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

    writer: Any = (
        spark.createDataFrame(frame.reset_index(drop=True)).write.format("delta").mode(mode)
    )
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.option("overwriteSchema", "true").saveAsTable(table)
    logger.info("wrote %d rows to %s", len(frame), table)


def to_spark(spark: SparkSession, frame: pd.DataFrame) -> SparkDataFrame:
    return spark.createDataFrame(frame.reset_index(drop=True))

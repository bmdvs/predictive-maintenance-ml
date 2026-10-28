"""Spark session for local runs and tests: single machine, Delta enabled, UTC session time zone.

On Databricks the platform session is used instead; this module is only imported by the local
parity runner and the tests.
"""

from __future__ import annotations

from pyspark.sql import SparkSession


def local_session(app: str = "pdm-local", cores: int = 4, shuffle_partitions: int = 8) -> SparkSession:
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.legacy.parquet.nanosAsLong", "true")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

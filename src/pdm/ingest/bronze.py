"""Bronze table catalogue: one entry per source export table, with its folder and file filter.

The pipeline loads each entry with Auto Loader from the landing volume; the local parity runner
loads the same entries as batch Parquet from the local exports. Both add _source_file and
_ingested_at. Three CMMS and platform tables carry nanosecond timestamps; with NANOS_AS_LONG set
they are read as integers and converted to timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

NANOS_AS_LONG = {"spark.sql.legacy.parquet.nanosAsLong": "true"}


@dataclass(frozen=True)
class BronzeTable:
    name: str
    folder: str
    glob: str = "*.parquet"
    nanos_columns: tuple[str, ...] = ()


BRONZE_TABLES: tuple[BronzeTable, ...] = (
    BronzeTable("cm_scalars", "cm_platform/scalars"),
    BronzeTable("cm_alerts", "cm_platform/alerts"),
    BronzeTable("cm_sensor_events", "cm_platform/sensor_events", nanos_columns=("ts",)),
    BronzeTable("cm_burst_index", "cm_platform", "burst_index.parquet"),
    BronzeTable("ctrl_minute", "controller/minute_agg"),
    BronzeTable("ctrl_hz1", "controller/hz1"),
    BronzeTable("ctrl_events", "controller/events"),
    BronzeTable("ctrl_khz_index", "controller/khz_index"),
    BronzeTable("cmp_minute", "compressor_ctrl/minute"),
    BronzeTable("cmp_room", "compressor_ctrl/room"),
    BronzeTable("cmp_events", "compressor_ctrl/events"),
    BronzeTable("cmms_work_orders", "cmms", "work_orders.parquet", ("opened", "started", "closed")),
    BronzeTable("cmms_pm_schedule", "cmms", "pm_schedule.parquet", ("due", "completed")),
    BronzeTable("cmms_drawbar_force_log", "cmms", "drawbar_force_log.parquet"),
    BronzeTable("cmms_oil_samples", "cmms", "oil_samples.parquet"),
    BronzeTable("erp_jobs", "erp", "jobs.parquet"),
    BronzeTable("erp_contribution_margin", "erp", "contribution_margin.parquet"),
    BronzeTable("reg_bearings", "register", "bearings.parquet"),
    BronzeTable("reg_machines", "register", "machines.parquet"),
    BronzeTable("reg_hmc_spindles", "register", "hmc_spindles.parquet"),
    BronzeTable("reg_pumps", "register", "pumps.parquet"),
    BronzeTable("reg_compressors", "register", "compressors.parquet"),
    BronzeTable("reg_sensors", "register", "sensors.parquet"),
)
TABLES = {t.name: t for t in BRONZE_TABLES}


def from_nanos(df: DataFrame, columns: tuple[str, ...]) -> DataFrame:
    for c in columns:
        if c in df.columns and dict(df.dtypes)[c] in ("bigint", "long"):
            df = df.withColumn(c, F.timestamp_micros(F.floor(F.col(c) / 1000).cast("long")))
    return df


def with_ingest_columns(df: DataFrame) -> DataFrame:
    return df.withColumn("_source_file", F.col("_metadata.file_path")).withColumn(
        "_ingested_at", F.current_timestamp()
    )


def read_batch(spark: SparkSession, root: str, name: str) -> DataFrame:
    """Batch read of one bronze table from a local or mounted export root."""
    t = TABLES[name]
    df = (
        spark.read.option("pathGlobFilter", t.glob)
        .option("recursiveFileLookup", "true")
        .parquet(f"{root.rstrip('/')}/{t.folder}")
    )
    return with_ingest_columns(from_nanos(df, t.nanos_columns))


def read_stream(spark: SparkSession, root: str, name: str, schema_location: str) -> DataFrame:
    """Auto Loader read of one bronze table from the landing volume."""
    t = TABLES[name]
    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{schema_location}/{name}")
        .option("pathGlobFilter", t.glob)
        .option("recursiveFileLookup", "true")
        .load(f"{root.rstrip('/')}/{t.folder}")
    )
    return with_ingest_columns(from_nanos(df, t.nanos_columns))

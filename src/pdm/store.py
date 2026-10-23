"""Table access for job tasks and the local parity runner.

On the platform tables are Unity Catalog tables (catalog.schema.table); locally they are Delta folders
under exports/local_gold/<schema>/<table>. Every task reads and writes through a Store, so the same
task code runs in both places.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession

TABLE_SCHEMA = {
    "baseline_sanity": "gold",
    "baseline_stats": "gold",
    "cm_scalars": "bronze",
    "cm_sensor_events": "bronze",
    "cmms_work_orders": "bronze",
    "confirmed_conditions": "silver",
    "controller_event_daily": "silver",
    "controller_hourly": "silver",
    "ctrl_events": "bronze",
    "ctrl_hz1": "bronze",
    "ctrl_minute": "bronze",
    "data_quality_monthly": "monitoring",
    "gap_table": "silver",
    "healthy_windows": "silver",
    "hourly_indicators": "silver",
    "hz1_reconciliation": "silver",
    "measurement_state_validation": "silver",
    "points": "silver",
    "reg_sensors": "bronze",
    "rolling_features": "silver",
    "rule_floor_alerts": "gold",
    "rule_floor_exceedances": "gold",
}


def table_name(catalog: str, table: str) -> str:
    """Fully qualified Unity Catalog name of a pipeline table; tables not listed are bronze."""
    return f"{catalog}.{TABLE_SCHEMA.get(table, 'bronze')}.{table}"


@dataclass
class Store:
    spark: SparkSession
    catalog: str | None = None
    local_root: Path | None = None

    def name(self, schema: str, table: str) -> str:
        if self.catalog:
            return f"{self.catalog}.{schema}.{table}"
        assert self.local_root is not None
        return (self.local_root / schema / table).as_posix()

    def version(self, schema: str, table: str) -> int | str | None:
        """Current Delta version of a table; for a pipeline materialized view or streaming table, which
        has no Delta history of its own, its last refresh time and pipeline update id. None when the
        table does not exist."""
        if not self.exists(schema, table):
            return None
        ref = self.name(schema, table) if self.catalog else f"delta.`{self.name(schema, table)}`"
        try:
            return int(self.spark.sql(f"DESCRIBE HISTORY {ref} LIMIT 1").collect()[0]["version"])
        except Exception:
            info = {
                r["col_name"].strip(): (r["data_type"] or "").strip()
                for r in self.spark.sql(f"DESCRIBE TABLE EXTENDED {ref}").collect()
                if r["col_name"]
            }
            update = info.get("Latest Refresh", "").rstrip("/").split("/")[-1]
            kind, refreshed = info.get("Type", "view"), info.get("Last Refreshed", "unknown")
            return f"{kind} refreshed {refreshed}, update {update}"

    def read(self, schema: str, table: str) -> DataFrame:
        if self.catalog:
            return self.spark.table(self.name(schema, table))
        return self.spark.read.format("delta").load(self.name(schema, table))

    def read_pd(self, schema: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.read(schema, table)
        return (df.select(*columns) if columns else df).toPandas()

    def write(self, df: DataFrame, schema: str, table: str) -> int:
        w = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        if self.catalog:
            w.saveAsTable(self.name(schema, table))
        else:
            w.save(self.name(schema, table))
        return self.read(schema, table).count()

    def write_pd(self, pdf: pd.DataFrame, schema: str, table: str) -> int:
        pdf = pdf.copy()
        pdf.columns = [re.sub(r"[ ,;{}()\n\t=]", "_", str(c)).strip("_") for c in pdf.columns]
        for c in pdf.columns:
            if pd.api.types.is_datetime64tz_dtype(pdf[c]):
                pdf[c] = pdf[c].dt.tz_convert("UTC").dt.tz_localize(None)
            elif pdf[c].dtype == object and pdf[c].map(lambda v: isinstance(v, (list, dict))).any():
                pdf[c] = pdf[c].astype(str)
        if pdf.empty:
            pdf = pdf.astype({c: "string" for c in pdf.columns if pdf[c].dtype == object})
        return self.write(
            self.spark.createDataFrame(pdf)
            if len(pdf.columns)
            else self.spark.createDataFrame([], "empty string"),
            schema,
            table,
        )

    def exists(self, schema: str, table: str) -> bool:
        if self.catalog:
            return self.spark.catalog.tableExists(self.name(schema, table))
        return (Path(self.name(schema, table)) / "_delta_log").exists()

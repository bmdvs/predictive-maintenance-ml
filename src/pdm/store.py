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

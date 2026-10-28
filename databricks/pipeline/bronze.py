"""Bronze streaming tables: one per source export table, loaded by Auto Loader from the landing volume."""

from pyspark import pipelines as dp

from pdm.config import load_config
from pdm.ingest.bronze import BRONZE_TABLES, read_stream
from pdm.quality.expectations import expectations

CFG = load_config(spark.conf.get("pdm.config_path"))  # noqa: F821
LANDING = spark.conf.get("pdm.landing_root")  # noqa: F821
SCHEMAS = spark.conf.get("pdm.schema_root")  # noqa: F821
CATALOG = spark.conf.get("pdm.catalog")  # noqa: F821
RULES = expectations(CFG)


def _define(name: str) -> None:
    rules = RULES.get(name, [])
    warn = {n: e for n, e, a in rules if a == "warn"}
    drop = {n: e for n, e, a in rules if a == "drop"}
    fail = {n: e for n, e, a in rules if a == "fail"}

    @dp.table(name=f"{CATALOG}.bronze.{name}", comment=f"Bronze export table {name}")
    @dp.expect_all(warn)
    @dp.expect_all_or_drop(drop)
    @dp.expect_all_or_fail(fail)
    def _table():
        return read_stream(spark, LANDING, name, SCHEMAS)  # noqa: F821


for _t in BRONZE_TABLES:
    _define(_t.name)

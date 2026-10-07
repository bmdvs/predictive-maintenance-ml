"""Labels from bronze CMMS records only: confirmed conditions by type and the unplanned failures used
for event-based evaluation.

Input: bronze CMMS work orders and the split dates. Output: one row per confirmed condition with its
type, asset, machine, the time the condition was confirmed (work-order open), the evaluation period,
and whether it is a PM finding (credited to a method only when the method alerted before the PM).
Findings are read from failure codes and the source work order link, never from note text.
Label noise in the CMMS (wrong or missing codes) is carried through, as it is in the shop's records.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

LOCAL_TZ = "America/Chicago"
BEARING_CODES = ("BRG-WEAR", "BRG-SPALL", "BRG-LUBE", "PMP-BRG", "CMP-BRG-MOTOR", "CMP-BRG-AIREND")
TYPE_BY_CODE = {
    "VIB-IMBAL": "imbalance",
    "VIB-MISALIGN": "misalignment",
    "VIB-LOOSE": "looseness",
    "RU-LEAK": "rotary union",
    "SP-COOLANT-INGRESS": "rotary union",
    "CMP-OILCOOL-FOUL": "cooler (compressor)",
    "SP-COOLER": "spindle cooling",
    "CRASH-DMG": "crash",
    "PMP-SEAL": "pump seal or impeller",
    "PMP-IMP": "pump seal or impeller",
    "CMP-TRIP": "compressor trip",
    "CMP-VALVE": "compressor trip",
}
LUBE_CODES = ("BRG-LUBE", "CMP-LUBE")


def condition_type(followed: F.Column) -> F.Column:
    """Condition type of a work order. An inspection or PM coded with a lubrication code is a
    lubrication finding unless a repair follows from it (`followed`), which then carries the condition.

    Args:
        followed: true when another work order names this one in `source_wo_id`.
    """
    code, wt = F.col("failure_code"), F.col("wo_type")
    mapped = F.create_map(*[F.lit(x) for kv in TYPE_BY_CODE.items() for x in kv])[code]
    repair = wt.isin("unplanned_repair", "condition_based", "corrective")
    lube = wt.isin("inspection", "pm") & code.isin(*LUBE_CODES) & ~followed
    return (
        F.when(lube, F.lit("lubrication"))
        .when(wt == "inspection", F.lit(None))
        .when((wt == "pm") & code.isNotNull(), mapped)
        .when(repair & code.isNull() & (wt == "unplanned_repair"), F.lit("unplanned, no code"))
        .when(repair & code.isin(*BEARING_CODES) & (wt == "unplanned_repair"), F.lit("bearing (unplanned)"))
        .when(repair & code.isin(*BEARING_CODES), F.lit("bearing (condition-based)"))
        .when(repair & mapped.isNotNull(), mapped)
        .when(wt == "unplanned_repair", F.lit("other machine stoppage"))
    )


def period_expr(cfg: dict[str, Any], ts_col: str) -> F.Column:
    s = cfg["splits"]
    d = F.to_date(F.from_utc_timestamp(F.col(ts_col), LOCAL_TZ))
    return (
        F.when(d.between(F.lit(str(s["train"][0])), F.lit(str(s["train"][1]))), "training")
        .when(d.between(F.lit(str(s["validation"][0])), F.lit(str(s["validation"][1]))), "validation")
        .when(d.between(F.lit(str(s["shadow"][0])), F.lit(str(s["shadow"][1]))), "shadow")
        .otherwise("outside")
    )


def confirmed_conditions(work_orders: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    sources = (
        work_orders.where(F.col("source_wo_id").isNotNull())
        .select(F.col("source_wo_id").alias("_source"))
        .distinct()
        .withColumn("_followed", F.lit(True))
    )
    wo = (
        work_orders.join(sources, work_orders["wo_id"] == sources["_source"], "left")
        .withColumn("condition_type", condition_type(F.coalesce(F.col("_followed"), F.lit(False))))
        .where(F.col("condition_type").isNotNull())
    )
    return wo.select(
        "wo_id",
        "asset_id",
        F.regexp_extract("asset_id", r"^([A-Z]+-\d+)", 1).alias("machine_id"),
        "wo_type",
        "failure_code",
        "condition_type",
        F.col("opened").alias("confirmed_ts"),
        "closed",
        "downtime_hours",
        "labor_hours",
        (F.col("wo_type") == "unplanned_repair").alias("unplanned"),
        (
            (F.col("wo_type") == "pm")
            | (
                F.coalesce(F.col("source_wo_id"), F.lit("")).startswith("PM-")
                & (F.col("failure_code") == "SP-COOLER")
            )
        ).alias("pm_finding"),
        period_expr(cfg, "opened").alias("period"),
    )

"""Data-quality expectations per pipeline table, shared by the Lakeflow pipeline (as expectation
decorators) and the local parity runner (evaluated as counts).

Each expectation is (name, SQL boolean expression, action) with action one of warn, drop or fail.
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

RECORD_START = "2023-07-01"
RECORD_END = "2026-07-01T06:00:00"
STATES = ("measured", "not_met")
EXECUTION = ("active", "ready", "stopped", "interrupted")
WO_TYPES = ("unplanned_repair", "pm", "inspection", "condition_based", "corrective", "project")


def failure_codes(cfg: dict[str, Any]) -> list[str]:
    return sorted({c for v in cfg["cmms"]["failure_codes"].values() for c in v})


def expectations(cfg: dict[str, Any]) -> dict[str, list[tuple[str, str, str]]]:
    codes = ",".join(f"'{c}'" for c in failure_codes(cfg))
    return {
        "cm_scalars": [
            ("hour_in_record", f"hour_utc >= '{RECORD_START}' AND hour_utc < '{RECORD_END}'", "fail"),
            ("state_accepted", "measurement_state IN ('measured','not_met')", "warn"),
            ("indicators_present_when_measured", "no_measurement OR vel_rms_mm_s IS NOT NULL", "drop"),
        ],
        "ctrl_minute": [
            ("minute_in_record", f"minute_utc >= '{RECORD_START}' AND minute_utc < '{RECORD_END}'", "fail"),
            ("execution_accepted", "execution IN ('active','ready','stopped','interrupted')", "warn"),
        ],
        "cmms_work_orders": [
            (
                "type_accepted",
                "wo_type IN ('unplanned_repair','pm','inspection','condition_based','corrective','project')",
                "warn",
            ),
            ("code_accepted", f"failure_code IS NULL OR failure_code IN ({codes})", "warn"),
            ("opened_in_record", f"opened >= '2023-06-01' AND opened < '{RECORD_END}'", "fail"),
        ],
        "hourly_indicators": [
            ("indicators_not_null", "vel_rms_mm_s IS NOT NULL AND hf_rms_g IS NOT NULL", "drop"),
        ],
    }


def evaluate(df: DataFrame, rules: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    total = df.count()
    out = []
    for name, expr, action in rules:
        failed = df.where(~F.expr(f"coalesce(({expr}), false)")).count()
        out.append(
            {
                "expectation": name,
                "action": action,
                "rows": total,
                "failed": failed,
                "passed": failed == 0 or action != "fail",
            }
        )
    return out


def duplicate_keys(df: DataFrame, keys: list[str]) -> int:
    return df.groupBy(*keys).count().where("count > 1").count()

"""Reads the maintenance queue's inputs from the prod workspace into the page's build cache.

Inputs: prod Unity Catalog tables through the SQL statement API (Databricks CLI profile and SQL
warehouse from the environment: DATABRICKS_CONFIG_PROFILE, PDM_WAREHOUSE_ID) and the retained bursts
from the landing volume. Output: Parquet files and .npz bursts under build/queue_cache (gitignored).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from pdm import log, paths

LOG = log.get_logger("pdm.reports.queue_data")
CACHE = paths.REPO_ROOT / "build" / "queue_cache"
SCORE_RUN = "477653893505463"
FROM = "2025-11-01"
NUM = ("INT", "LONG", "BIGINT", "SHORT", "BYTE", "DOUBLE", "FLOAT", "DECIMAL")
SILVER_COLS = [
    "point_id",
    "asset_id",
    "hour_utc",
    "speed_rpm_at_measurement",
    "vel_rms_mm_s",
    "hf_rms_g",
    "temp_c",
    "env_bpfo_1_g2",
    "env_bpfi_1_g2",
    "env_bsf_1_g2",
    "env_ftf_1_g2",
    "implausible",
]
SCORE_COLS = ["point_id", "asset_id", "hour_utc", "segment", "state", "score", "threshold", "temp_resid_c"]


def _api(*args: str) -> dict:
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "pdm")
    for _ in range(4):
        p = subprocess.run(["databricks", "--profile", profile, "api", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return json.loads(p.stdout)
        time.sleep(10)
    raise RuntimeError(p.stderr[-500:])


def fetch(stmt: str) -> pd.DataFrame:
    """One SQL statement's result as a DataFrame (inline JSON chunks, typed from the result schema)."""
    body = json.dumps(
        {
            "warehouse_id": os.environ["PDM_WAREHOUSE_ID"],
            "statement": stmt,
            "wait_timeout": "50s",
            "on_wait_timeout": "CONTINUE",
            "disposition": "INLINE",
            "format": "JSON_ARRAY",
        }
    )
    out = _api("post", "/api/2.0/sql/statements", "--json", body)
    while out["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(3)
        out = _api("get", f"/api/2.0/sql/statements/{out['statement_id']}")
    if out["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(out["status"])
    cols = out["manifest"]["schema"]["columns"]
    rows = list(out["result"].get("data_array") or [])
    nxt = out["result"].get("next_chunk_internal_link")
    while nxt:
        ch = _api("get", nxt)
        rows += ch.get("data_array") or []
        nxt = ch.get("next_chunk_internal_link")
    d = pd.DataFrame(rows, columns=[c["name"] for c in cols])
    for c in cols:
        t, n = str(c["type_name"]).upper(), c["name"]
        if t.startswith(NUM):
            d[n] = pd.to_numeric(d[n], errors="coerce")
        elif t == "BOOLEAN":
            d[n] = d[n].map({"true": True, "false": False})
        elif t.startswith("TIMESTAMP"):
            d[n] = pd.to_datetime(d[n], utc=True, format="ISO8601")
    return d


def _monthly(table: str, cols: list[str], start: str, end: str = "2026-07-02") -> pd.DataFrame:
    sel = ", ".join(f"`{c}`" for c in cols)
    parts = []
    for m in pd.date_range(start, end, freq="MS"):
        z = m + pd.offsets.MonthBegin(1)
        q = f"SELECT {sel} FROM {table} WHERE hour_utc >= TIMESTAMP'{m:%Y-%m-%d}' AND hour_utc < TIMESTAMP'{z:%Y-%m-%d}'"
        parts.append(fetch(q))
    return pd.concat(parts, ignore_index=True)


def _work_requests_version() -> int:
    """The work_requests Delta version the D2 score run wrote."""
    h = fetch("DESCRIBE HISTORY pdm.gold.work_requests")
    hit = h[h["job"].astype(str).str.contains(f'"jobRunId":"{SCORE_RUN}"')]
    if len(hit) != 1:
        raise RuntimeError(f"work_requests: expected one version from job run {SCORE_RUN}, found {len(hit)}")
    return int(hit["version"].iloc[0])


def fetch_all(cache: Path = CACHE) -> dict[str, int]:
    """Writes every input table to the cache and returns the row counts."""
    cache.mkdir(parents=True, exist_ok=True)
    v = _work_requests_version()
    q = {
        "level_episodes": "SELECT * FROM pdm.gold.level_episodes WHERE period = 'shadow'",
        "cases": "SELECT * FROM pdm.gold.cases",
        "work_requests": f"SELECT * FROM pdm.gold.work_requests VERSION AS OF {v}",
        "watch_hours": f"SELECT * FROM pdm.gold.watch_hours WHERE period = 'shadow' AND hour_utc >= TIMESTAMP'{FROM}'",
        "ml_segments": "SELECT * FROM pdm.gold.ml_segments",
        "vendor_alerts": "SELECT * FROM pdm.gold.rule_floor_alerts WHERE method = 'vendor'",
        "work_orders": "SELECT wo_id, asset_id, component, wo_type, failure_code, opened, started, closed, downtime_hours, "
        "labor_hours, notes, no_fault_found, platform_alert_id, source_wo_id FROM pdm.bronze.cmms_work_orders",
        "tsc_pressure": "SELECT * FROM pdm.gold.tsc_pressure_daily",
        "discharge": "SELECT * FROM pdm.gold.compressor_discharge_daily",
        "controller_daily": "SELECT machine_id, local_date, thermistor_residual_c FROM pdm.gold.controller_health_daily "
        "WHERE machine_id LIKE 'HMC-%'",
        "conditions": "SELECT wo_id, asset_id, condition_type, failure_code, wo_type, confirmed_ts, credit_ts "
        "FROM pdm.gold.eval_conditions",
        "sensors": "SELECT point_id, asset_id, location, bearing, install_date, remount_dates FROM pdm.bronze.reg_sensors",
        "bearings": "SELECT * FROM pdm.bronze.reg_bearings",
        "sensor_events": "SELECT point_id, ts, event, detail FROM pdm.bronze.cm_sensor_events",
        "burst_index": "SELECT * FROM pdm.bronze.cm_burst_index",
    }
    counts = {}
    for name, stmt in q.items():
        d = fetch(stmt)
        d.to_parquet(cache / f"{name}.parquet", index=False)
        counts[name] = len(d)
        LOG.info("fetched %s rows=%d", name, len(d))
    sc = _monthly("pdm.gold.nbm_scores", SCORE_COLS, FROM)
    sc.to_parquet(cache / "scores.parquet", index=False)
    counts["scores"] = len(sc)
    ind = _monthly("pdm.silver.hourly_indicators", SILVER_COLS, FROM)
    ind.to_parquet(cache / "indicators.parquet", index=False)
    counts["indicators"] = len(ind)
    counts["level_rows"] = len(fetch_level_rows(cache))
    base = _vendor_window_rows(cache)
    base.to_parquet(cache / "baseline_rows.parquet", index=False)
    counts["baseline_rows"] = len(base)
    (cache / "fetch.json").write_text(
        json.dumps({"work_requests_version": v, "score_run": SCORE_RUN, "rows": counts}, indent=1),
        encoding="utf-8",
    )
    return counts


def _vendor_window_rows(cache: Path) -> pd.DataFrame:
    """The hourly rows of each point's Vendor Rules baselining windows (21 days from each install, remount or
    baselining start), for the healthy levels and limits."""
    from pdm.config import load_config

    days = int(load_config()["measurement"]["baselining_days"])
    ev = pd.read_parquet(cache / "sensor_events.parquet")
    ev = ev[ev["event"].isin(["install", "baselining_start", "remount"])]
    conds = [
        f"(point_id = '{r.point_id}' AND hour_utc >= TIMESTAMP'{pd.Timestamp(r.ts):%Y-%m-%d %H:%M:%S}' "
        f"AND hour_utc < TIMESTAMP'{pd.Timestamp(r.ts) + pd.Timedelta(days=days + 1):%Y-%m-%d %H:%M:%S}')"
        for r in ev.itertuples()
    ]
    sel = ", ".join(f"`{c}`" for c in SILVER_COLS)
    parts = [
        fetch(f"SELECT {sel} FROM pdm.silver.hourly_indicators WHERE " + " OR ".join(conds[i : i + 40]))
        for i in range(0, len(conds), 40)
    ]
    return pd.concat(parts, ignore_index=True).drop_duplicates(["point_id", "hour_utc"])


def fetch_bursts(paths_: list[str], cache: Path = CACHE) -> list[Path]:
    """Copies retained bursts (paths relative to the bursts folder of the landing volume) into the cache."""
    out = []
    for rel in paths_:
        dst = cache / "bursts" / "retained" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "pdm")
            src = f"dbfs:/Volumes/pdm/landing/exports/cm_platform/{rel}"
            p = subprocess.run(
                ["databricks", "--profile", profile, "fs", "cp", src, str(dst)],
                capture_output=True,
                text=True,
            )
            if p.returncode != 0:
                raise RuntimeError(p.stderr[-300:])
        out.append(dst)
    return out


def fetch_level_rows(cache: Path = CACHE) -> pd.DataFrame:
    """The ML Model's temperature residual over each segment's level window (from segment start to level
    ready), from the training and evaluation-year scores, for the residual chart's level bands."""
    seg = pd.read_parquet(cache / "ml_segments.parquet")
    parts = []
    for table in ("pdm.gold.ml_train_scores", "pdm.gold.nbm_scores"):
        conds = [
            f"(point_id = '{r.point_id}' AND hour_utc >= TIMESTAMP'{pd.Timestamp(r.segment_start):%Y-%m-%d %H:%M:%S}' "
            f"AND hour_utc < TIMESTAMP'{pd.Timestamp(r.level_ready):%Y-%m-%d %H:%M:%S}')"
            for r in seg.itertuples()
            if pd.notna(r.level_ready)
        ]
        for i in range(0, len(conds), 40):
            q = f"SELECT point_id, hour_utc, temp_resid_c FROM {table} WHERE " + " OR ".join(
                conds[i : i + 40]
            )
            parts.append(fetch(q))
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["point_id", "hour_utc"])
    d.to_parquet(cache / "level_rows.parquet", index=False)
    return d


def fetch_burst_rows(plan: list[dict], cache: Path = CACHE) -> pd.DataFrame:
    """The stored hourly indicator rows (every reported axis) of the planned burst hours, from bronze."""
    conds = " OR ".join(
        f"(point_id = '{r['point_id']}' AND hour_utc = TIMESTAMP'{pd.Timestamp(r['hour_utc']):%Y-%m-%d %H:%M:%S}')"
        for r in plan
    )
    d = fetch(f"SELECT * FROM pdm.bronze.cm_scalars WHERE {conds}")
    d.to_parquet(cache / "burst_rows.parquet", index=False)
    return d


if __name__ == "__main__":
    log.configure()
    print(json.dumps(fetch_all(), indent=1))

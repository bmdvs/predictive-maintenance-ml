"""Rolling features over 24 h, 7 d and 30 d per point: median, slope per day and ratio to baseline for
the core indicators.

Inputs: the hourly indicator table and the per-window baseline medians. Output: one row per measured
point-hour. The baseline for an hour is the point's most recent usable window that ended before the
hour, in the hour's regime when that regime has a baseline, otherwise pooled across regimes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

CORE_MAP = {
    "accel_rms_g": "accel_rms_g",
    "vel_rms_mm_s": "vel_rms_mm_s",
    "hf_rms_g": "hf_rms_g",
    "crest_factor": "crest_factor",
    "kurtosis": "kurtosis",
    "temp_c": "temp_c",
    "order_1x": "ord_01_g2",
    "order_2x": "ord_02_g2",
    "env_bpfo_1": "env_bpfo_1_g2",
    "env_bpfi_1": "env_bpfi_1_g2",
    "env_bsf_1": "env_bsf_1_g2",
    "env_ftf_1": "env_ftf_1_g2",
}
SECONDS_PER_DAY = 86400.0


def rolling_schema(cfg: dict[str, Any]) -> str:
    cols = ["point_id string", "hour_utc timestamp", "regime string", "baseline_window string"]
    for name in cfg["rolling"]["core_indicators"]:
        for w in cfg["rolling"]["windows"]:
            cols += [f"{name}_med_{w} double", f"{name}_slope_{w} double", f"{name}_ratio_{w} double"]
    return ", ".join(cols)


def _point_rolling(pdf: pd.DataFrame, base: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    pdf = pdf.sort_values("hour_utc").reset_index(drop=True)
    pdf = pdf[~pdf["implausible"].fillna(False).astype(bool)].reset_index(drop=True)
    out = pdf[["point_id", "hour_utc", "regime"]].copy()
    b = base[base["point_id"] == pdf["point_id"].iat[0]].sort_values("end") if len(pdf) else base.iloc[0:0]
    out["baseline_window"] = None
    if len(b) and len(pdf):
        ends = b[["window_id", "end"]].drop_duplicates().sort_values("end")
        idx = np.searchsorted(ends["end"].to_numpy(), pdf["hour_utc"].to_numpy(), side="right") - 1
        out["baseline_window"] = np.where(idx >= 0, ends["window_id"].to_numpy()[np.clip(idx, 0, None)], None)
    t = (
        (pdf["hour_utc"] - pdf["hour_utc"].iat[0]).dt.total_seconds().to_numpy() / SECONDS_PER_DAY
        if len(pdf)
        else np.array([])
    )
    s = pdf.set_index("hour_utc")
    for name in cfg["rolling"]["core_indicators"]:
        col = CORE_MAP[name]
        x = s[col].astype(float)
        bl = pd.Series(np.nan, index=range(len(pdf)))
        if len(b):
            sub = b[b["indicator"] == col]
            pooled = sub.groupby("window_id")["median"].median()
            byreg = sub.set_index(["window_id", "regime"])["median"]
            keys = list(zip(out["baseline_window"], out["regime"], strict=True))
            bl = pd.Series(
                [byreg.get(k, pooled.get(k[0], np.nan)) if k[0] is not None else np.nan for k in keys]
            )
        tt = pd.Series(t, index=s.index)
        for w in cfg["rolling"]["windows"]:
            r = x.rolling(w, min_periods=1)
            med = r.median().to_numpy()
            mt, mx = tt.rolling(w, min_periods=2).mean(), r.mean()
            cov = (tt * x).rolling(w, min_periods=2).mean() - mt * mx
            var = (tt * tt).rolling(w, min_periods=2).mean() - mt * mt
            slope = np.where(
                var.to_numpy() > 0, cov.to_numpy() / np.where(var.to_numpy() > 0, var.to_numpy(), 1.0), np.nan
            )
            out[f"{name}_med_{w}"] = med
            out[f"{name}_slope_{w}"] = slope
            out[f"{name}_ratio_{w}"] = med / bl.to_numpy()
    return out


def rolling_features(indicators: DataFrame, baselines: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    """baselines: window stats with window end times (baseline_with_ends), paired with each point's
    indicator rows by point."""
    cols = ["point_id", "hour_utc", "regime", "implausible", *CORE_MAP.values()]

    def fn(pdf: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
        return _point_rolling(pdf, base, cfg) if len(pdf) else pd.DataFrame(columns=_rolling_columns(cfg))

    return (
        indicators.select(*cols)
        .groupBy("point_id")
        .cogroup(baselines.groupBy("point_id"))
        .applyInPandas(fn, rolling_schema(cfg))
    )


def _rolling_columns(cfg: dict[str, Any]) -> list[str]:
    return [c.split(" ")[0] for c in rolling_schema(cfg).split(", ")]


def baseline_with_ends(stats: DataFrame, windows: DataFrame) -> DataFrame:
    return stats.join(windows.where(F.col("usable")).select("window_id", "end"), "window_id")

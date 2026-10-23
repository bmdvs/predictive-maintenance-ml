"""Baseline statistics per healthy window and regime, and the fleet baseline sanity check.

Inputs: the hourly indicator table restricted to healthy windows. Outputs: per window and regime the
median, MAD, 5th and 95th percentiles of the baseline indicators; and per window the sanity flags:
the ratio of its key indicator medians to its sister points' (groups of three or more, high side only)
and the rise across the window implied by the Theil-Sen trend of HF and envelope fault-band energy.
Bronze only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from scipy.stats import theilslopes

BASELINE_INDICATORS = [
    "accel_rms_g",
    "vel_rms_mm_s",
    "hf_rms_g",
    "crest_factor",
    "kurtosis",
    "temp_c",
    "ord_01_g2",
    "ord_02_g2",
    "env_bpfo_1_g2",
    "env_bpfi_1_g2",
    "env_bsf_1_g2",
    "env_ftf_1_g2",
]
SISTER_INDICATORS = ["hf_rms_g", "vel_rms_mm_s", "env_fault_g2"]
TREND_INDICATORS = ["hf_rms_g", "env_fault_g2"]


def with_env_fault(df: DataFrame) -> DataFrame:
    """Envelope fault-band energy: BPFO, BPFI and BSF first harmonics; missing bands count as zero."""
    return df.withColumn(
        "env_fault_g2",
        F.coalesce(F.col("env_bpfo_1_g2"), F.lit(0.0))
        + F.coalesce(F.col("env_bpfi_1_g2"), F.lit(0.0))
        + F.coalesce(F.col("env_bsf_1_g2"), F.lit(0.0)),
    )


def window_stats(rows_in_windows: DataFrame) -> DataFrame:
    """Long table: window, regime, indicator, median, MAD, p05, p95, n (clean rows only)."""
    clean = rows_in_windows.where(~F.coalesce(F.col("implausible"), F.lit(False)))
    cols = ["window_id", "point_id", "regime", *BASELINE_INDICATORS]

    def stats(pdf: pd.DataFrame) -> pd.DataFrame:
        out = []
        for c in BASELINE_INDICATORS:
            v = pdf[c].dropna().to_numpy(dtype=float)
            if v.size == 0:
                continue
            med = float(np.median(v))
            out.append(
                {
                    "window_id": pdf["window_id"].iat[0],
                    "point_id": pdf["point_id"].iat[0],
                    "regime": pdf["regime"].iat[0],
                    "indicator": c,
                    "median": med,
                    "mad": float(np.median(np.abs(v - med))),
                    "p05": float(np.quantile(v, 0.05)),
                    "p95": float(np.quantile(v, 0.95)),
                    "n": int(v.size),
                }
            )
        return pd.DataFrame(
            out, columns=["window_id", "point_id", "regime", "indicator", "median", "mad", "p05", "p95", "n"]
        )

    schema = (
        "window_id string, point_id string, regime string, indicator string, "
        "median double, mad double, p05 double, p95 double, n long"
    )
    return clean.select(*cols).groupBy("window_id", "regime").applyInPandas(stats, schema)


def window_medians(rows_in_windows: DataFrame) -> DataFrame:
    """Per window, the pooled median of each sanity indicator over clean rows."""
    clean = with_env_fault(rows_in_windows).where(~F.coalesce(F.col("implausible"), F.lit(False)))
    aggs = [F.percentile_approx(c, 0.5, 10000).alias(c) for c in SISTER_INDICATORS]
    return clean.groupBy("window_id", "point_id").agg(*aggs, F.count("*").alias("n"))


def sister_ratios(medians: pd.DataFrame, pts: pd.DataFrame, min_group: int) -> pd.DataFrame:
    """Ratio of each window's median to the median of its sister points' windows, per indicator, for
    sister groups of at least min_group points."""
    m = medians.merge(pts[["point_id", "sister_group"]], on="point_id")
    sizes = pts.groupby("sister_group")["point_id"].nunique()
    rows = []
    for _, w in m.iterrows():
        if sizes.get(w["sister_group"], 0) < min_group:
            continue
        sis = m[(m["sister_group"] == w["sister_group"]) & (m["point_id"] != w["point_id"])]
        for ind in SISTER_INDICATORS:
            ref = float(np.median(sis[ind].dropna())) if sis[ind].notna().any() else np.nan
            val = float(w[ind]) if pd.notna(w[ind]) else np.nan
            rows.append(
                {
                    "window_id": w["window_id"],
                    "point_id": w["point_id"],
                    "indicator": ind,
                    "value": val,
                    "sister_median": ref,
                    "ratio": val / ref if ref and ref > 0 and pd.notna(val) else np.nan,
                }
            )
    return pd.DataFrame(
        rows, columns=["window_id", "point_id", "indicator", "value", "sister_median", "ratio"]
    )


def window_trends(rows_in_windows: DataFrame) -> DataFrame:
    """Per window and trend indicator, the rise across the window implied by the Theil-Sen slope of the
    log indicator against time."""
    clean = with_env_fault(rows_in_windows).where(~F.coalesce(F.col("implausible"), F.lit(False)))
    cols = ["window_id", "point_id", "hour_utc", *TREND_INDICATORS]

    def fn(pdf: pd.DataFrame) -> pd.DataFrame:
        t = (
            pd.to_datetime(pdf["hour_utc"]) - pd.to_datetime(pdf["hour_utc"]).min()
        ).dt.total_seconds() / 86400.0
        span = float(t.max()) if len(t) else 0.0
        out = []
        for ind in TREND_INDICATORS:
            v = pdf[ind].to_numpy(dtype=float)
            ok = np.isfinite(v) & (v > 0)
            if ok.sum() < 10 or span <= 0:
                continue
            slope = theilslopes(np.log(v[ok]), t.to_numpy()[ok])[0]
            out.append(
                {
                    "window_id": pdf["window_id"].iat[0],
                    "point_id": pdf["point_id"].iat[0],
                    "indicator": ind,
                    "slope_log_per_day": float(slope),
                    "span_days": span,
                    "implied_rise": float(np.exp(slope * span)),
                }
            )
        return pd.DataFrame(
            out,
            columns=["window_id", "point_id", "indicator", "slope_log_per_day", "span_days", "implied_rise"],
        )

    schema = (
        "window_id string, point_id string, indicator string, slope_log_per_day double, "
        "span_days double, implied_rise double"
    )
    return clean.select(*cols).groupBy("window_id").applyInPandas(fn, schema)


def proposed_factors(
    ratios: pd.DataFrame, trends: pd.DataFrame, install_windows: set[str], sc: dict[str, Any]
) -> dict[str, float]:
    """Sister factor: the configured quantile of install-window sister ratios above 1 (high side only).
    Trend factor: the configured quantile of the implied rises across all baseline windows."""
    r = ratios[ratios["window_id"].isin(install_windows)]["ratio"].dropna()
    r = r[r > 0]
    hi = np.exp(np.abs(np.log(r)))
    rises = trends["implied_rise"].dropna()
    return {
        "sister_factor": float(np.quantile(hi, sc["sister_quantile"])) if len(hi) else np.nan,
        "trend_factor": float(np.quantile(np.maximum(rises, 1.0 / rises), sc["trend_quantile"]))
        if len(rises)
        else np.nan,
    }


def sanity_flags(
    ratios: pd.DataFrame, trends: pd.DataFrame, windows: pd.DataFrame, factors: dict[str, float]
) -> pd.DataFrame:
    """One flag row per window: sister flag (high side only), trend flag, and the indicators behind them."""
    rows = []
    for w in windows.itertuples():
        r = ratios[ratios["window_id"] == w.window_id]
        tr = trends[trends["window_id"] == w.window_id]
        sis_hit = r[r["ratio"] > factors["sister_factor"]]
        tr_hit = tr[tr["implied_rise"] > factors["trend_factor"]]
        rows.append(
            {
                "window_id": w.window_id,
                "point_id": w.point_id,
                "reason": w.reason,
                "sister_tested": bool(len(r)),
                "sister_ratio_max": float(r["ratio"].max()) if len(r) else np.nan,
                "trend_rise_max": float(tr["implied_rise"].max()) if len(tr) else np.nan,
                "flag_sister": bool(len(sis_hit)),
                "flag_trend": bool(len(tr_hit)),
                "flagged": bool(len(sis_hit) or len(tr_hit)),
                "indicators": ",".join(
                    [f"sister:{i}" for i in sis_hit["indicator"]]
                    + [f"trend:{i}" for i in tr_hit["indicator"]]
                ),
                "sister_factor": factors["sister_factor"],
                "trend_factor": factors["trend_factor"],
            }
        )
    return pd.DataFrame(rows)


SANITY_SCHEMA = (
    "window_id string, point_id string, reason string, sister_tested boolean, sister_ratio_max double, "
    "trend_rise_max double, flag_sister boolean, flag_trend boolean, flagged boolean, indicators string, "
    "sister_factor double, trend_factor double, proposed_sister double, proposed_trend double"
)


def sanity_table(
    rows_in_windows: DataFrame, pts: DataFrame, windows: DataFrame, cfg: dict[str, Any]
) -> DataFrame:
    """Fleet baseline sanity flags for every window, with the factors used (configured, or proposed from
    healthy spread while the configured values are null) and the proposed factors. Lazy: the inputs are
    combined into one Spark group and the flag logic runs when the table is computed."""
    sc = cfg["baselines"]["sanity"]
    parts = [
        window_medians(rows_in_windows).withColumn("kind", F.lit("median")),
        pts.select("point_id", "sister_group").withColumn("kind", F.lit("point")),
        window_trends(rows_in_windows).withColumn("kind", F.lit("trend")),
        windows.select("window_id", "point_id", "reason").withColumn("kind", F.lit("window")),
    ]
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.unionByName(p, allowMissingColumns=True)

    def fn(pdf: pd.DataFrame) -> pd.DataFrame:
        med = pdf[pdf["kind"] == "median"][["window_id", "point_id", *SISTER_INDICATORS, "n"]]
        pts_pd = pdf[pdf["kind"] == "point"][["point_id", "sister_group"]]
        trends = pdf[pdf["kind"] == "trend"][
            ["window_id", "point_id", "indicator", "slope_log_per_day", "span_days", "implied_rise"]
        ]
        w = pdf[pdf["kind"] == "window"][["window_id", "point_id", "reason"]]
        ratios = sister_ratios(med.reset_index(drop=True), pts_pd, sc["sister_min_group"])
        install = set(w.loc[w["reason"] == "install", "window_id"])
        proposed = proposed_factors(ratios, trends.reset_index(drop=True), install, sc)
        factors = {
            "sister_factor": sc.get("sister_factor") or proposed["sister_factor"],
            "trend_factor": sc.get("trend_factor") or proposed["trend_factor"],
        }
        flags = sanity_flags(ratios, trends, w, factors)
        return flags.assign(
            proposed_sister=proposed["sister_factor"], proposed_trend=proposed["trend_factor"]
        )

    return combined.groupBy(F.lit(1).alias("_all")).applyInPandas(fn, SANITY_SCHEMA)

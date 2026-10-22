"""Baseline statistics per healthy window and regime, and the fleet baseline sanity check.

Inputs: the hourly indicator table restricted to healthy windows. Outputs: per window and regime the
median, MAD, 5th and 95th percentiles of the baseline indicators; and per window the ratio of its key
indicator medians to the median of its sister points' windows, flagged when outside the configured
factor. Bronze only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

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
SANITY_INDICATORS = {"hf_rms_g": "hf_rms_g", "vel_rms_mm_s": "vel_rms_mm_s", "env_fault_g2": "env_fault_g2"}


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
    aggs = [F.percentile_approx(c, 0.5, 10000).alias(c) for c in SANITY_INDICATORS]
    return clean.groupBy("window_id", "point_id").agg(*aggs, F.count("*").alias("n"))


def sister_ratios(medians: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:
    """Ratio of each window's median to the median of the sister points' windows, per indicator."""
    m = medians.merge(pts[["point_id", "sister_group"]], on="point_id")
    rows = []
    for _, w in m.iterrows():
        sis = m[(m["sister_group"] == w["sister_group"]) & (m["point_id"] != w["point_id"])]
        for ind in SANITY_INDICATORS:
            ref = float(np.median(sis[ind].dropna())) if sis[ind].notna().any() else np.nan
            val = float(w[ind]) if pd.notna(w[ind]) else np.nan
            rows.append(
                {
                    "window_id": w["window_id"],
                    "point_id": w["point_id"],
                    "sister_group": w["sister_group"],
                    "indicator": ind,
                    "value": val,
                    "sister_median": ref,
                    "ratio": val / ref if ref and ref > 0 and pd.notna(val) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def proposed_factor(ratios: pd.DataFrame, install_windows: set[str], quantile: float) -> float:
    """Sanity factor from the healthy sister-to-sister spread of install-window ratios: the given
    quantile of |log ratio| across all sister comparisons, as a multiplicative factor."""
    r = ratios[ratios["window_id"].isin(install_windows)]["ratio"].dropna()
    r = r[r > 0]
    return float(np.exp(np.quantile(np.abs(np.log(r)), quantile))) if len(r) else np.nan


def sanity_flags(ratios: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = ratios.copy()
    out["outside"] = (out["ratio"] > factor) | (out["ratio"] < 1.0 / factor)
    per = (
        out.groupby(["window_id", "point_id"])
        .apply(
            lambda g: pd.Series(
                {
                    "flagged": bool(g["outside"].any()),
                    "indicators": ",".join(g.loc[g["outside"], "indicator"]),
                    "max_ratio": float(g["ratio"].max()),
                    "min_ratio": float(g["ratio"].min()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per["factor"] = factor
    return per


def sanity_table(
    spark: Any, rows_in_windows: DataFrame, pts: DataFrame, windows: DataFrame, cfg: dict[str, Any]
) -> tuple[DataFrame, float]:
    """Fleet baseline sanity flags for every window, and the factor used."""
    med = window_medians(rows_in_windows).toPandas()
    pts_pd = pts.select("point_id", "sister_group").toPandas()
    ratios = sister_ratios(med, pts_pd)
    w = windows.select("window_id", "reason").toPandas()
    install = set(w.loc[w["reason"] == "install", "window_id"])
    sc = cfg["baselines"]["sanity"]
    factor = sc.get("factor") or proposed_factor(ratios, install, sc["spread_quantile"])
    flags = sanity_flags(ratios, factor)
    return spark.createDataFrame(flags), factor

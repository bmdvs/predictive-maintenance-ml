"""MLflow pyfunc wrapper for one candidate's per-point normal-behavior models, for the Unity Catalog
registry (pdm.ml.nbm_point_models).

Holds each segment's version history (initial fit, then each monthly causal update, with the hours each
version applied to) and the per-segment healthy stats as an artifact. Input: hourly indicator rows.
Output: per input row, in input order, the segment and update in effect, the score, the threshold at every
sweep percentile, the top contributors, and the threshold and exceedance at the operating percentile;
rows the models do not score (implausible, unmeasured, re-baseline windows, rule-floor-only segments,
after the training horizon) have no score. Persistence is applied downstream.
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from pdm.models import nbm

HEALTHY_STATS = "healthy_stats"


class NBMPointModels(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        candidate: str,
        percentile: float,
        segments: dict[str, list[dict[str, Any]]],
        top_k: int,
        sweep_percentiles: list[float],
        horizon: pd.Timestamp,
        healthy_stats: pd.DataFrame | None = None,
    ) -> None:
        self.candidate = candidate
        self.percentile = float(percentile)
        self.segments = segments
        self.top_k = int(top_k)
        self.sweep_percentiles = [float(p) for p in sweep_percentiles]
        self.horizon = pd.Timestamp(horizon)
        self.healthy_stats = healthy_stats

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["healthy_stats"] = None
        return state

    def load_context(self, context: Any) -> None:
        path = (getattr(context, "artifacts", None) or {}).get(HEALTHY_STATS)
        if path:
            self.healthy_stats = pd.read_parquet(path)

    def score_frame(self, rows: pd.DataFrame, carry: tuple[str, ...] = ()) -> pd.DataFrame:
        """Scored hours in the training task's format, point by point."""
        parts = []
        for pid, g in rows.groupby("point_id", sort=True):
            s = nbm.score_versions(
                g, self.segments.get(pid, []), self.candidate, self.top_k, self.horizon, carry
            )
            if len(s):
                parts.append(s)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        x = model_input.reset_index(drop=True).copy()
        x["hour_utc"] = pd.to_datetime(x["hour_utc"], utc=True)
        if "implausible" not in x:
            x["implausible"] = False
        x["implausible"] = x["implausible"].fillna(False).astype(bool)
        x["_row"] = np.arange(len(x))
        s = self.score_frame(x, carry=("_row",))
        thr_cols = [f"thr_{p:g}" for p in self.sweep_percentiles]
        cols = ["model_segment", "model_regime", "model_update", "score", "top_contributors", *thr_cols]
        out = pd.DataFrame({"point_id": x["point_id"], "hour_utc": x["hour_utc"].dt.tz_localize(None)})
        for c in cols:
            out[c] = np.nan if c == "score" or c.startswith("thr_") else None
        if len(s):
            idx = s["_row"].to_numpy()
            for c in cols:
                if c in s:
                    vals = s[c].to_numpy()
                    if c == "score" or c.startswith("thr_"):
                        out.loc[idx, c] = vals.astype(float)
                    else:
                        out[c] = out[c].astype(object)
                        out.loc[idx, c] = vals
        op = f"thr_{self.percentile:g}"
        out["threshold"] = out.get(op, np.nan)
        out["exceeds"] = out["score"] > out["threshold"]
        return out

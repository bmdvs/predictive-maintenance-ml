"""MLflow pyfunc wrapper for one candidate's per-point normal-behavior models, for the Unity Catalog
registry (pdm.ml.nbm_point_models).

Input: hourly indicator rows (point_id, hour_utc, regime and the candidate's indicator columns). Output:
per input row, in input order, the segment in effect, the anomaly score from that segment's current
model, the threshold at the operating percentile and whether the score exceeds it (persistence is
applied downstream).
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from pdm.models import nbm


class NBMPointModels(mlflow.pyfunc.PythonModel):
    def __init__(self, candidate: str, percentile: float, segments: dict[str, list[dict[str, Any]]]) -> None:
        self.candidate = candidate
        self.percentile = float(percentile)
        self.segments = segments

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        parts = []
        for pid, g in model_input.groupby("point_id", sort=False):
            segs = sorted(self.segments.get(pid, []), key=lambda s: s["start"])
            out = pd.DataFrame(
                {
                    "point_id": pid,
                    "hour_utc": g["hour_utc"].to_numpy(),
                    "segment": -1,
                    "score": np.nan,
                    "threshold": np.nan,
                },
                index=g.index,
            )
            if segs:
                starts = [s["start"] for s in segs]
                idx = nbm.segment_ids(g["hour_utc"], starts)
                x, _ = nbm.features(g, self.candidate)
                reg = g["regime"].to_numpy()
                for i, s in enumerate(segs):
                    rows = idx == i
                    for key, fm in s["models"].items():
                        m = rows & np.isfinite(x).all(axis=1) & ((reg == key) if key != "all" else True)
                        if not m.any():
                            continue
                        sc, _ = fm.raw_scores(x[m], reg[m])
                        out.loc[g.index[m], "score"] = sc
                        out.loc[g.index[m], "threshold"] = fm.thresholds[self.percentile]
                        out.loc[g.index[m], "segment"] = int(s["segment"])
            parts.append(out)
        res = (
            pd.concat(parts).reindex(model_input.index)
            if parts
            else pd.DataFrame(columns=["point_id", "hour_utc", "segment", "score", "threshold"])
        )
        res["exceeds"] = res["score"] > res["threshold"]
        return res.reset_index(drop=True)

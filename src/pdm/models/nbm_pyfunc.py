"""MLflow pyfunc wrapper for the frozen ML Model, for the Unity Catalog registry (pdm.ml.nbm_point_models).

Holds one fitted model per sensor point (segment-relative features, per-point temperature residual fit,
thresholds by speed band or for the point), the plausibility factor set from training, and the training
healthy stats as an artifact. Input: hourly indicator rows with each point's segment starts in the
column segment_starts (ISO times separated by ';', from the healthy-window table) and, for compressor
points, the room ambient in room_ambient_c. Output: per input row, in input order, the state (scored,
recentering, plausibility, no level), the score, the threshold at the operating percentile and at every
sweep percentile, the temperature residual, the top contributors and the inputs above their own limits
(limit_inputs). Persistence and levels are applied
downstream. (Revision 2; originally each segment's monthly version history.)
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from pdm.models import nbm, temperature

HEALTHY_STATS = "healthy_stats"


class NBMPointModels(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        settings_key: str,
        models: dict[str, nbm.PointModel],
        plausibility_factor: float,
        cfg: dict[str, Any],
        healthy_stats: pd.DataFrame | None = None,
    ) -> None:
        self.settings_key = settings_key
        self.models = models
        self.plausibility_factor = float(plausibility_factor)
        nb = cfg["models"]["nbm"]
        self.cfg = {
            "models": {
                "nbm": {
                    **{
                        k: nb[k]
                        for k in (
                            "recenter_days",
                            "recenter_min_hours",
                            "recenter_floor_hours",
                            "sweep_percentiles",
                            "sweep_extension_percentiles",
                            "threshold_percentile",
                            "top_contributors",
                        )
                    },
                    # the model fixes (2026-11-01) travel with the registered model, so scoring applies them
                    **({"fixes": dict(nb["fixes"])} if nb.get("fixes") else {}),
                },
                "seed": cfg["models"]["seed"],
            },
            "regime": {"speed_bands_frac_rated": cfg["regime"]["speed_bands_frac_rated"]},
        }
        self.healthy_stats = healthy_stats

    @property
    def candidate(self) -> str:
        return next(iter(self.models.values())).settings.candidate if self.models else ""

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["healthy_stats"] = None
        return state

    def load_context(self, context: Any) -> None:
        path = (getattr(context, "artifacts", None) or {}).get(HEALTHY_STATS)
        if path:
            self.healthy_stats = pd.read_parquet(path)

    def score_span(
        self,
        rows: pd.DataFrame,
        windows: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        ambient: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Scored hours from start to end and the segment table, point by point (the task format)."""
        parts, segs = [], []
        for pid, g in rows.groupby("point_id", sort=True):
            m = self.models.get(pid)
            if m is None:
                continue
            w = windows[windows["point_id"] == pid]
            amb = ambient if str(m.point_class).startswith("compressor") else None
            s, sg = nbm.score_point(
                m,
                g,
                w,
                self.cfg,
                start,
                end,
                self.plausibility_factor,
                self.cfg["models"]["nbm"]["top_contributors"],
                amb,
            )
            if len(s):
                parts.append(s)
            if len(sg):
                segs.append(sg)
        return (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(),
            pd.concat(segs, ignore_index=True) if segs else pd.DataFrame(),
        )

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        x = model_input.reset_index(drop=True).copy()
        x["hour_utc"] = pd.to_datetime(x["hour_utc"], utc=True)
        if "implausible" not in x:
            x["implausible"] = False
        x["implausible"] = x["implausible"].fillna(False).astype(bool)
        win_rows = []
        for pid, g in x.groupby("point_id"):
            starts = (
                str(g["segment_starts"].dropna().iat[0])
                if "segment_starts" in g and g["segment_starts"].notna().any()
                else ""
            )
            for s in [v for v in starts.split(";") if v]:
                win_rows.append({"point_id": pid, "start": pd.Timestamp(s), "reason": ""})
        windows = pd.DataFrame(win_rows, columns=["point_id", "start", "reason"])
        if len(windows):
            windows["start"] = pd.to_datetime(windows["start"], utc=True)
        amb = None
        if temperature.AMBIENT in x:
            amb = x.dropna(subset=[temperature.AMBIENT]).groupby("hour_utc")[temperature.AMBIENT].mean()
        start, end = x["hour_utc"].min(), x["hour_utc"].max() + pd.Timedelta(hours=1)
        s, _ = self.score_span(x, windows, start, end, amb)
        out = pd.DataFrame({"point_id": x["point_id"], "hour_utc": x["hour_utc"].dt.tz_localize(None)})
        if s.empty:
            out["state"], out["score"], out["threshold"] = None, np.nan, np.nan
            return out
        s = s.assign(hour_utc=pd.to_datetime(s["hour_utc"], utc=True).dt.tz_localize(None))
        keep = [
            "point_id",
            "hour_utc",
            "state",
            "score",
            "threshold",
            "temp_resid_c",
            "top_contributors",
            "limit_inputs",
            *[c for c in s.columns if c.startswith("thr_")],
        ]
        out = out.merge(
            s[keep].drop_duplicates(["point_id", "hour_utc"]), on=["point_id", "hour_utc"], how="left"
        )
        out["exceeds"] = out["score"] > out["threshold"]
        return out

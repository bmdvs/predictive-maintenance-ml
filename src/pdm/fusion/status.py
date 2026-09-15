"""Asset status from all alerting layers and the case log.

Inputs: a unified signal table (method, asset, point, time, level, rules) from the rule floor, the
per-point models and controller health, and CMMS work orders. Outputs: per-asset status changes
(OK, WATCH, ALARM) and cases. ALARM needs a persisted model or controller alert, an alarm-level rule,
or two layers within the rate-of-change window; any other signal in the window is WATCH. One case per
asset at a time: it opens at ALARM and closes when a work order on the asset closes after it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

LAYER = {"vendor": "rule_floor", "iso": "rule_floor", "nbm": "nbm", "controller": "controller"}
STRONG = {"nbm", "controller"}


def status_changes(signals: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Status at each signal time and when the window expires, per asset."""
    win = pd.Timedelta(hours=cfg["models"]["fusion"]["rate_of_change_window_h"])
    rows = []
    for asset, g in signals.sort_values("ts").groupby("asset_id"):
        times = sorted(set(g["ts"]) | set(g["ts"] + win))
        last = None
        for t in times:
            recent = g[(g["ts"] <= t) & (g["ts"] > t - win)]
            if recent.empty:
                st = "OK"
            else:
                layers = set(recent["method"].map(LAYER))
                alarm = (
                    bool(set(recent["method"]) & STRONG)
                    or (recent["level"] == "alarm").any()
                    or len(layers) >= 2
                )
                st = "ALARM" if alarm else "WATCH"
            if st != last:
                rows.append(
                    {
                        "asset_id": asset,
                        "ts": t,
                        "status": st,
                        "layers": ",".join(sorted(set(recent["method"]))),
                    }
                )
                last = st
    return pd.DataFrame(rows, columns=["asset_id", "ts", "status", "layers"])


def cases(changes: pd.DataFrame, signals: pd.DataFrame, work_orders: pd.DataFrame) -> pd.DataFrame:
    wo = work_orders[~work_orders["wo_type"].isin(["project"])][["asset_id", "wo_id", "wo_type", "closed"]]
    rows = []
    for asset, g in changes.sort_values("ts").groupby("asset_id"):
        open_until = None
        for r in g[g["status"] == "ALARM"].itertuples():
            if open_until is not None and r.ts < open_until:
                continue
            w = (
                wo[(wo["asset_id"].str.startswith(asset)) & (wo["closed"] > r.ts)]
                .sort_values("closed")
                .head(1)
            )
            trig = signals[(signals["asset_id"] == asset) & (signals["ts"] <= r.ts)].sort_values("ts").tail(5)
            closed = w["closed"].iat[0] if len(w) else pd.NaT
            rows.append(
                {
                    "case_id": f"{asset}|{pd.Timestamp(r.ts):%Y%m%d%H}",
                    "asset_id": asset,
                    "opened": r.ts,
                    "first_alarm": r.ts,
                    "layers": r.layers,
                    "trigger_points": ",".join(sorted(set(trig["point_id"].dropna()))),
                    "trigger_rules": ";".join(
                        f"{m}:{x}" for m, x in zip(trig["method"], trig["rules"], strict=True)
                    ),
                    "closed": closed,
                    "closing_wo": w["wo_id"].iat[0] if len(w) else None,
                    "closing_wo_type": w["wo_type"].iat[0] if len(w) else None,
                }
            )
            open_until = closed if pd.notna(closed) else pd.Timestamp.max.tz_localize("UTC")
    return pd.DataFrame(rows)

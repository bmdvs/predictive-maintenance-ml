"""Replay of the platform's shipped alert logic from the bronze export alone.

Inputs: one point's resultant scalar rows (including no_measurement rows) and its sensor events
(install, remount, baselining starts). Output: the alert records the platform logic produces,
for comparison with the exported alert log.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pdm.features.burst import RESULTANT
from pdm.rules.vendor_baseline import VendorRuleSettings, VendorRuleState, compute_baseline

LOCAL_TZ = "America/Chicago"
_BASELINE_EVENTS = {"install", "baselining_start", "remount"}


def replay_point(
    point_id: str,
    rows: pd.DataFrame,
    sensor_events: pd.DataFrame,
    settings: VendorRuleSettings,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    meas = cfg["measurement"]
    days = pd.Timedelta(days=meas["baselining_days"])
    state = VendorRuleState(point_id, settings)
    res = rows[rows["axis"] == RESULTANT].sort_values("hour_utc")
    ev = sensor_events[
        (sensor_events["point_id"] == point_id) & sensor_events["event"].isin(_BASELINE_EVENTS)
    ].sort_values("ts")
    ev_times = list(ev["ts"])
    k = 0
    baseline_start = None
    baseline_rows: list[dict[str, float]] = []
    alerts = []
    local_day = res["hour_utc"].dt.tz_convert(LOCAL_TZ).dt.normalize()
    for day, g in res.groupby(local_day, sort=True):
        day_end = (day + pd.Timedelta(days=1)).tz_convert("UTC")
        while k < len(ev_times) and ev_times[k] < day_end:
            baseline_start, baseline_rows, state.baseline = ev_times[k], [], None
            state.reset_counters()
            k += 1
        for rec in g.to_dict("records"):
            ts = rec["hour_utc"]
            if rec["no_measurement"]:
                state.update(ts, None)
                continue
            in_baselining = baseline_start is not None and ts < baseline_start + days
            if in_baselining:
                baseline_rows.append(rec)
                state.update(ts, None)
            elif state.baseline is None and baseline_start is not None:
                if baseline_rows:
                    state.baseline = compute_baseline(
                        pd.DataFrame(baseline_rows), settings, meas["baseline_statistic"]
                    )
                baseline_start = None
            alert = state.update(ts, rec) if state.baseline is not None and not in_baselining else None
            if alert is not None:
                alerts.append(alert)
    return alerts

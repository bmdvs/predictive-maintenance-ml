"""The maintenance queue's state at the snapshot hour: each method's level per asset, the active list in
queue order, the open work orders and work requests, and the inputs of each asset's detail section.

Inputs: the build cache written by pdm.reports.queue_data. Outputs: plain frames and dicts for
pdm.reports.queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdm.advisory import requests as adv
from pdm.config import load_config
from pdm.features import kinematics
from pdm.reports.queue_data import CACHE
from pdm.rules.vendor_baseline import VendorRuleSettings, compute_baseline

TZ = "America/Chicago"
SPAN = (
    pd.Timestamp("2026-04-01 00:00", tz=TZ).tz_convert("UTC"),
    pd.Timestamp("2026-07-01 00:00", tz=TZ).tz_convert("UTC"),
)
MIN_ALERT, MIN_ALARM = 4, 2
CONFIRM_DAYS = 45
TREND_DAYS, HISTORY_DAYS, RECENT_DAYS = 90, 365, 7
METHODS = {"vendor_rules": "Vendor Rules", "ml_model": "ML Model"}
POINT_CLASS = {
    "front_bearing_housing": "spindle_front",
    "rear_bearing_housing": "spindle_rear",
    "motor_de": "compressor_motor",
    "airend_inlet": "compressor_airend",
}
KIND = {"SP": "spindle", "TSC": "tsc_pump", "CP": "flood_pump"}
KIND_LABEL = {
    "spindle": "HMC spindle",
    "tsc_pump": "TSC pump",
    "flood_pump": "Flood coolant pump",
    "compressor": "Air compressor",
}
COMPONENT = {
    "spindle": "SPINDLE",
    "tsc_pump": "TSC_PUMP",
    "flood_pump": "COOLANT_PUMP",
    "compressor": "COMPRESSOR",
}
POINT_LABEL = {
    "front_bearing_housing": "Front",
    "rear_bearing_housing": "Rear",
    "pump_motor_de": "Motor DE",
    "motor_de": "Motor DE",
    "airend_inlet": "Airend",
}
PM_TASKS = {
    "spindle": ("PM: lube", "PM: spindle lube check", "PM: drawbar force"),
    "tsc_pump": ("PM: tsc filter",),
    "flood_pump": ("PM: coolant check",),
}
BANDS = [
    ("bpfo", "BPFO", "Outer race"),
    ("bpfi", "BPFI", "Inner race"),
    ("bsf", "BSF", "Ball"),
    ("ftf", "FTF", "Cage"),
]
RANK = {
    ("vendor_rules", "ALARM"): 0,
    ("vendor_rules", "ALERT"): 1,
    ("ml_model", "ALARM"): 2,
    ("ml_model", "ALERT"): 3,
}


def utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def local(ts: Any, fmt: str = "%Y-%m-%d") -> str:
    if ts is None or pd.isna(ts):
        return ""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(TZ).strftime(fmt)


LOCATIONS = ("outer race", "inner race", "ball", "cage")


def located_band(detail: object) -> str | None:
    """The bearing location a D2 detail names from its dominant fault band, or None (no dominant band)."""
    d = str(detail or "")
    return next((x for x in LOCATIONS if d.startswith(x)), None)


def asset_kind(asset: str) -> str:
    return "compressor" if asset.startswith("CMP") else KIND[asset.split("-")[-1]]


def machine_of(asset: str) -> str:
    return "-".join(asset.split("-")[:2])


@dataclass
class Level:
    method: str
    level: str
    opened: pd.Timestamp
    alarm_ts: pd.Timestamp | None
    alarm_cause: str | None
    trigger_points: list[str]
    trigger_rules: str


class QueueState:
    """Every frame the page needs, read from the cache, with the snapshot hour chosen by the stated rule."""

    def __init__(self, cache: Path = CACHE) -> None:
        self.cache = cache
        self.cfg = load_config()

    def t(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.cache / f"{name}.parquet")

    # ------------------------------------------------------------------ levels and snapshot
    @cached_property
    def episodes(self) -> pd.DataFrame:
        e = self.t("level_episodes")
        for c in ("opened", "closed", "alarm_ts", "last"):
            e[c] = utc(e[c])
        return e

    def counts_at(self, ts: pd.Timestamp) -> dict[str, int]:
        e = self.episodes
        o = e[(e["opened"] <= ts) & (e["closed"] > ts)]
        alarm = o["alarm_ts"].notna() & (o["alarm_ts"] <= ts)
        out = {
            "ALERT": int((~alarm).sum()),
            "ALARM": int(alarm.sum()),
            "assets": int(o["asset_id"].nunique()),
        }
        for m in METHODS:
            out[f"{m} ALERT"] = int(((~alarm) & (o["method"] == m)).sum())
            out[f"{m} ALARM"] = int((alarm & (o["method"] == m)).sum())
        return out

    @cached_property
    def conditions(self) -> pd.DataFrame:
        c = self.t("conditions")
        c["confirmed_ts"] = utc(c["confirmed_ts"])
        c["family"] = [
            adv.family_of(t, a, False) for t, a in zip(c["condition_type"], c["asset_id"], strict=True)
        ]
        return c

    def developing(self, ts: pd.Timestamp, located: bool) -> list[dict[str, Any]]:
        """Assets at the hour with a condition confirmed within the following 45 days, both methods at ALERT or
        above, and a D2 work request naming the confirmed family (with a located fault band when asked)."""
        c = self.conditions
        c = c[(c["confirmed_ts"] > ts) & (c["confirmed_ts"] <= ts + pd.Timedelta(days=CONFIRM_DAYS))]
        out = []
        lv = self.levels(ts)
        for r in c.sort_values("confirmed_ts").itertuples():
            m = lv.get(r.asset_id, {})
            if "vendor_rules" not in m or "ml_model" not in m:
                continue
            wr = self.work_request(r.asset_id, m["ml_model"])
            if wr is None or wr["suspected_condition"] != r.family:
                continue
            loc = located_band(wr["suspected_detail"])
            if located and not loc:
                continue
            out.append(
                {
                    "asset_id": r.asset_id,
                    "wo_id": r.wo_id,
                    "family": r.family,
                    "confirmed_ts": r.confirmed_ts,
                    "location": loc,
                }
            )
        return out

    @cached_property
    def snapshot_choice(self) -> dict[str, Any]:
        """The latest hour from April 1 to June 30, 2026 (local) with at least 4 ALERTs and 2 ALARMs open across
        both methods and at least one asset where a condition confirmed within the following 45 days is
        developing, both methods are at ALERT or above and the D2 work request names the confirmed family with a
        located fault band; without a qualifying hour, the located-band requirement is dropped; without one
        still, the latest hour meeting the counts. (Was: the counts alone, then the most ALARMs and ALERTs.)"""
        hours = pd.date_range(SPAN[0], SPAN[1] - pd.Timedelta(hours=1), freq="h")
        d = pd.DataFrame([{"ts": h, **self.counts_at(h)} for h in hours])
        q = d[(d["ALERT"] >= MIN_ALERT) & (d["ALARM"] >= MIN_ALARM)]
        for located, rule in (
            (True, "counts and a developing condition named with a located band"),
            (False, "counts and a developing condition named by family (located band dropped)"),
        ):
            for r in q.sort_values("ts", ascending=False).itertuples():
                dev = self.developing(r.ts, located)
                if dev:
                    n = sum(1 for h in q["ts"] if self.developing(h, located)) if located else None
                    return {
                        "ts": r.ts,
                        "rule": rule,
                        "showcase": dev[0],
                        "developing": dev,
                        "qualifying_hours": n,
                        **{k: getattr(r, k) for k in d.columns if k != "ts" and k.isidentifier()},
                        **{k: int(q.loc[q["ts"] == r.ts, k].iloc[0]) for k in d.columns if k != "ts"},
                    }
        r = q.iloc[-1]
        return {
            "ts": r["ts"],
            "rule": "counts only",
            "showcase": None,
            **{k: r[k] for k in d.columns if k != "ts"},
        }

    @property
    def T(self) -> pd.Timestamp:
        return self.snapshot_choice["ts"]

    def levels(self, ts: pd.Timestamp | None = None) -> dict[str, dict[str, Level]]:
        """asset -> method -> its open level at the hour."""
        ts = self.T if ts is None else ts
        e = self.episodes
        o = e[(e["opened"] <= ts) & (e["closed"] > ts)]
        out: dict[str, dict[str, Level]] = {}
        for r in o.itertuples():
            alarm = pd.notna(r.alarm_ts) and r.alarm_ts <= ts
            out.setdefault(r.asset_id, {})[r.method] = Level(
                r.method,
                "ALARM" if alarm else "ALERT",
                r.opened,
                r.alarm_ts if alarm else None,
                r.alarm_cause if alarm else None,
                [p for p in str(r.trigger_points).split(",") if p],
                str(r.trigger_rules),
            )
        return out

    # ------------------------------------------------------------------ registers
    @cached_property
    def sensors(self) -> pd.DataFrame:
        return self.t("sensors")

    def points(self, asset: str) -> list[str]:
        s = self.sensors
        order = {
            "front_bearing_housing": 0,
            "rear_bearing_housing": 1,
            "motor_de": 0,
            "airend_inlet": 1,
            "pump_motor_de": 0,
        }
        g = s[s["asset_id"] == asset]
        return list(g.assign(o=g["location"].map(order)).sort_values("o")["point_id"])

    def point_label(self, pid: str) -> str:
        loc = self.sensors.set_index("point_id").at[pid, "location"]
        return POINT_LABEL[loc]

    def point_class(self, pid: str) -> str:
        loc = self.sensors.set_index("point_id").at[pid, "location"]
        if loc == "pump_motor_de":
            return asset_kind(self.sensors.set_index("point_id").at[pid, "asset_id"])
        return POINT_CLASS[loc]

    @cached_property
    def assets(self) -> list[str]:
        return sorted(self.sensors["asset_id"].unique(), key=lambda a: (a.startswith("CMP"), a))

    def location(self, asset: str) -> str:
        if asset.startswith("CMP"):
            return "Compressor room"
        return f"HMC cell, {machine_of(asset)}"

    def asset_name(self, asset: str) -> str:
        return KIND_LABEL[asset_kind(asset)]

    # ------------------------------------------------------------------ CMMS
    @cached_property
    def work_orders(self) -> pd.DataFrame:
        w = self.t("work_orders")
        for c in ("opened", "started", "closed"):
            w[c] = utc(w[c])
        return w

    def asset_work_orders(self, asset: str) -> pd.DataFrame:
        """Work orders on the asset, machine-level work orders on its component, and the machine's PM tasks that
        service it (logged on the machine with no component)."""
        w = self.work_orders
        kind = asset_kind(asset)
        own = w["asset_id"] == asset
        if not asset.startswith("CMP"):
            on_machine = w["asset_id"] == machine_of(asset)
            task = w["notes"].fillna("").str.rstrip(".").isin(PM_TASKS[kind])
            own |= on_machine & ((w["component"] == COMPONENT[kind]) | ((w["wo_type"] == "pm") & task))
        return w[own & (w["opened"] <= self.T)].sort_values("opened")

    def open_work_orders(self, asset: str) -> pd.DataFrame:
        w = self.asset_work_orders(asset)
        return w[(w["wo_type"] != "pm") & (w["closed"].isna() | (w["closed"] > self.T))]

    @cached_property
    def work_requests(self) -> pd.DataFrame:
        w = self.t("work_requests")
        w["created"] = utc(w["created"])
        return w.set_index("case_id")

    def work_request(self, asset: str, lv: Level) -> pd.Series | None:
        cid = f"{asset}|{lv.opened:%Y%m%d%H}"
        return self.work_requests.loc[cid] if cid in self.work_requests.index else None

    # ------------------------------------------------------------------ measurements
    @cached_property
    def indicators(self) -> pd.DataFrame:
        d = self.t("indicators")
        d["hour_utc"] = utc(d["hour_utc"])
        return d.sort_values(["point_id", "hour_utc"])

    @cached_property
    def scores(self) -> pd.DataFrame:
        d = self.t("scores")
        d["hour_utc"] = utc(d["hour_utc"])
        return d.sort_values(["point_id", "hour_utc"])

    @cached_property
    def segments(self) -> pd.DataFrame:
        d = self.t("ml_segments")
        for c in ("segment_start", "level_ready"):
            d[c] = utc(d[c])
        return d

    @cached_property
    def watch(self) -> pd.DataFrame:
        d = self.t("watch_hours")
        d["hour_utc"] = utc(d["hour_utc"])
        return d

    @cached_property
    def sensor_events(self) -> pd.DataFrame:
        d = self.t("sensor_events")
        d["ts"] = utc(d["ts"])
        return d

    @cached_property
    def baseline_rows(self) -> pd.DataFrame:
        d = self.t("baseline_rows")
        d["hour_utc"] = utc(d["hour_utc"])
        return d

    def vendor_settings(self, pid: str) -> VendorRuleSettings:
        return VendorRuleSettings.for_point(self.cfg, self.point_class(pid))

    @cached_property
    def vendor_windows(self) -> pd.DataFrame:
        """Each point's Vendor Rules baselining windows with their healthy levels (median of the window's
        hourly rows, the platform's baseline) and spread (5th to 95th percentile)."""
        days = pd.Timedelta(days=int(self.cfg["measurement"]["baselining_days"]))
        ev = self.sensor_events[self.sensor_events["event"].isin(["install", "baselining_start", "remount"])]
        rows = []
        br = self.baseline_rows
        for r in ev.sort_values(["point_id", "ts"]).itertuples():
            g = br[(br["point_id"] == r.point_id) & (br["hour_utc"] >= r.ts) & (br["hour_utc"] < r.ts + days)]
            if g.empty:
                continue
            s = self.vendor_settings(r.point_id)
            b = compute_baseline(g, s.__class__(**{**s.__dict__, "harmonics": 1}), "median")
            row = {"point_id": r.point_id, "start": r.ts, "ready": r.ts + days, "event": r.event, **b}
            for c in (
                "vel_rms_mm_s",
                "hf_rms_g",
                "temp_c",
                "env_bpfo_1_g2",
                "env_bpfi_1_g2",
                "env_bsf_1_g2",
                "env_ftf_1_g2",
            ):
                v = g[c].dropna()
                row[f"{c}_p05"] = float(v.quantile(0.05)) if len(v) else np.nan
                row[f"{c}_p95"] = float(v.quantile(0.95)) if len(v) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def window_at(self, pid: str, ts: pd.Timestamp) -> pd.Series | None:
        w = self.vendor_windows
        w = w[(w["point_id"] == pid) & (w["ready"] <= ts)]
        return w.iloc[-1] if len(w) else None

    def vendor_limits(self, pid: str, ts: pd.Timestamp) -> dict[str, float]:
        """Vendor Rules limits at the hour (from the window in force): velocity and HF alert and alarm, envelope
        fault-band alert (6 dB over the healthy level), housing temperature warning and alarm."""
        s = self.vendor_settings(pid)
        w = self.window_at(pid, ts)
        from pdm.rules.iso_zones import zone_b_width

        out = {"temp_alert": s.temp_warning_c, "temp_alarm": s.temp_alarm_c}
        if w is None:
            return out
        width = zone_b_width(s.group, s.foundation)
        out.update(
            {
                "vel_alert": w["vel_rms_mm_s"] + s.alert_frac * width,
                "vel_alarm": w["vel_rms_mm_s"] + s.alarm_frac * width,
                "hf_alert": w["hf_rms_g"] + s.alert_frac * s.hf_reference_g,
                "hf_alarm": w["hf_rms_g"] + s.alarm_frac * s.hf_reference_g,
            }
        )
        ratio = 10 ** (s.env_db / 10.0)
        for k, _, _ in BANDS:
            out[f"env_{k}_alert"] = w[f"env_{k}_1_g2"] * ratio if pd.notna(w[f"env_{k}_1_g2"]) else np.nan
        return out

    # ------------------------------------------------------------------ bearings
    def fault_orders(self, pid: str) -> kinematics.FaultOrders:
        b = self.t("bearings").set_index("part")
        r = b.loc[self.sensors.set_index("point_id").at[pid, "bearing"]]
        return kinematics.fault_orders(
            int(r["n"]), float(r["d_mm"]), float(r["pitch_mm"]), float(r["contact_deg"])
        )

    # ------------------------------------------------------------------ bursts for the spectrum panels
    @cached_property
    def vendor_alerts(self) -> pd.DataFrame:
        a = self.t("vendor_alerts")
        a["alert_ts"] = utc(a["alert_ts"])
        a["asset_id"] = a["point_id"].map(self.sensors.set_index("point_id")["asset_id"])
        return a

    @cached_property
    def burst_index(self) -> pd.DataFrame:
        b = self.t("burst_index")
        b["hour_utc"] = utc(b["hour_utc"])
        return b

    def alert_opening(self, asset: str) -> dict[str, Any] | None:
        """The opening the spectrum's first panel shows: the Vendor Rules episode when one is open (its first
        alert's point and hour), otherwise the ML Model's (its work request's point, the next measured hour)."""
        lv = self.levels().get(asset, {})
        if "vendor_rules" in lv:
            v = lv["vendor_rules"]
            a = self.vendor_alerts
            a = a[(a["asset_id"] == asset) & (a["alert_ts"] >= v.opened) & (a["alert_ts"] <= self.T)]
            a = a.assign(o=a["point_id"].map({p: i for i, p in enumerate(self.points(asset))})).sort_values(
                ["alert_ts", "o"]
            )
            r = a.iloc[0]
            return {
                "method": "vendor_rules",
                "point_id": r["point_id"],
                "opened": v.opened,
                "hour": r["alert_ts"],
            }
        if "ml_model" in lv:
            m = lv["ml_model"]
            wr = self.work_request(asset, m)
            pid = wr["point_id"] if wr is not None else m.trigger_points[0]
            ind = self.indicators
            nxt = ind[(ind["point_id"] == pid) & (ind["hour_utc"] >= m.opened) & ind["vel_rms_mm_s"].notna()]
            return {
                "method": "ml_model",
                "point_id": pid,
                "opened": m.opened,
                "hour": nxt["hour_utc"].iloc[0],
            }
        return None

    def burst_plan(self) -> list[dict[str, Any]]:
        """Per active asset: the burst at the alert, the current burst (last measured hour at the snapshot) and
        the healthy reference (the weekly retained burst in the Vendor Rules window in force at the alert)."""
        out = []
        ind = self.indicators
        bi = self.burst_index
        for asset in self.levels():
            op = self.alert_opening(asset)
            if op is None:
                continue
            pid = op["point_id"]
            last = ind[(ind["point_id"] == pid) & (ind["hour_utc"] <= self.T) & ind["vel_rms_mm_s"].notna()][
                "hour_utc"
            ].iloc[-1]
            w = self.window_at(pid, op["hour"])
            ref = bi[(bi["point_id"] == pid) & (bi["reason"] == "weekly_reference")]
            if w is not None:
                inside = ref[(ref["hour_utc"] >= w["start"]) & (ref["hour_utc"] < w["ready"])]
                ref = inside if len(inside) else ref[ref["hour_utc"] < op["hour"]]
            ref_hour = ref["hour_utc"].iloc[-1]
            for role, hour in (("alert", op["hour"]), ("current", last), ("reference", ref_hour)):
                hit = bi[(bi["point_id"] == pid) & (bi["hour_utc"] == hour)]
                out.append(
                    {
                        "asset_id": asset,
                        "point_id": pid,
                        "role": role,
                        "hour_utc": hour,
                        "retained": hit["path"].iloc[0] if len(hit) else None,
                    }
                )
        return out

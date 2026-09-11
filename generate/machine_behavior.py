"""Machine-level behavior shared by every CNC: operating state per hour, collisions, other
stoppages, axis and ball-screw wear, and the per-hour machine state the controller export renders.

Mixed into asset-group engines that provide new_wo, log, schedule, take_down, open_wo and failed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from generate import crash
from generate.degradation import BearingInstance
from pdm.constants import DAYS_PER_YEAR, SECONDS_PER_MINUTE

AXES_BY_CLASS = {
    "hmc": ["x", "y", "z", "b"],
    "vmc": ["x", "y", "z"],
    "lathe": ["x", "z"],
    "mill_turn": ["x", "y", "z"],
    "grinder": ["x", "y", "z"],
}


def draw_axis_events(
    cfg: dict[str, Any], reg: Any, start: pd.Timestamp, end: pd.Timestamp, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Axis and ball-screw wear events: count drawn in the configured range, placed by machine weight."""
    aw = cfg["axis_wear"]
    lo, hi = aw["events_in_record"]
    count = int(rng.integers(lo, hi + 1))
    weights = aw["machine_weight"]
    ids, w = [], []
    for mid, m in reg.machines.items():
        key = (
            m["class"] if m["class"] in ("hmc", "mill_turn", "grinder") else f"{m['age_class']}_{m['class']}"
        )
        ids.append(mid)
        w.append(float(weights[key]))
    w = np.asarray(w) / np.sum(w)
    chosen = rng.choice(ids, size=count, replace=False, p=w)
    events = []
    for mid in chosen:
        months = float(rng.uniform(*aw["duration_months"]))
        dur = pd.Timedelta(days=months * DAYS_PER_YEAR / 12)
        span = (end - start - dur).total_seconds()
        t0 = start + pd.Timedelta(seconds=float(rng.uniform(0, max(span, 0.0))))
        axes = AXES_BY_CLASS[reg.machines[mid]["class"]]
        axes = [a for a in axes if a != "b"]
        events.append(
            {
                "machine_id": str(mid),
                "axis": str(rng.choice(axes)),
                "start": t0,
                "end": t0 + dur,
                "friction_end": float(rng.uniform(*aw["friction_multiplier_at_end"])),
                "preload_end": float(rng.uniform(*aw["preload_fraction_at_end"])),
                "backlash_increase_um": float(rng.uniform(*aw["backlash_entry_increase_um_at_end"])),
            }
        )
    return events


class MachineMixin:
    """Machine-level state for an asset-group engine."""

    def init_machine(self, machine_id: str, sched: pd.DataFrame, axis_events: list[dict[str, Any]]) -> None:
        cfg, sh = self.cfg, self.sh
        self.mid = machine_id
        self.machine = self.reg.machines[machine_id]
        self.sched = sched
        self.down_until = sh.window_start
        years = (sh.window_end - sh.window_start) / pd.Timedelta(days=DAYS_PER_YEAR)
        sched_h_year = max(float(sched["scheduled_min"].sum()) / SECONDS_PER_MINUTE / years, 1.0)
        cut_h_year = max(float(sched["cutting_min"].sum()) / SECONDS_PER_MINUTE / years, 1.0)
        self.crash_rate_h = crash.rate_per_cutting_hour(cfg, cut_h_year)
        key = "hmc" if self.machine["class"] == "hmc" else "other"
        self.stoppage_rate_h = {}
        for mode, rate in cfg["other_stoppages"]["rates_per_machine_year"].items():
            r = rate[key] if isinstance(rate, dict) else rate
            if r > 0:
                self.stoppage_rate_h[mode] = r / sched_h_year
        self.axis_events = [
            dict(e, done=False, replaced=False) for e in axis_events if e["machine_id"] == machine_id
        ]
        self.axes = AXES_BY_CLASS[self.machine["class"]]
        self.machine_hours: list[dict[str, Any]] = []
        self.machine_asset = machine_id

    def operating(self, i: int, ts: pd.Timestamp) -> dict[str, Any]:
        row = self.sched.iloc[i]
        down = ts < self.down_until
        cut = 0.0 if down else float(row["cutting_min"])
        return {
            "cut": cut,
            "down": down,
            "speed": float(row["speed_rpm"]),
            "load": float(row["load_pct"]),
            "flutes": int(row["flutes"]),
            "tsc": float(row["tsc_frac"]),
            "flood": float(row["flood_frac"]),
            "shift": row["shift"],
            "program_created": row["program_created"],
        }

    def machine_crash(
        self, ts: pd.Timestamp, op: dict[str, Any], front: BearingInstance, sp_asset: str
    ) -> None:
        if op["cut"] <= 0 or sp_asset in self.failed:
            return
        age = ts - op["program_created"] if pd.notna(op["program_created"]) else None
        if self.rng.random() >= crash.hour_probability(
            self.cfg, self.crash_rate_h, op["cut"], op["shift"], age
        ):
            return
        outcome, above = crash.draw_outcome(self.cfg, self.rng)
        self.log(ts, "collision", asset_id=self.mid, outcome=outcome, above_threshold=above)
        if above:
            factor = self.cfg["bearing_degradation"]["hazard_multipliers"]["crash_remaining_life_factor"]
            front.scale_remaining(float(self.rng.uniform(*factor)))
        if outcome == "brinelling" and front.onset_ts is None:
            front.begin_onset(
                ts, "brinelling", theta=self.cfg["crash"]["brinelling_initial_hi"], defect="outer_race"
            )
            self.log(
                ts,
                "bearing_onset",
                component_id=front.component_id,
                instance_id=front.instance_id,
                cause="brinelling",
                defect="outer_race",
            )
        elif outcome == "immediate_failure":
            self.unplanned_failure(
                ts, sp_asset, list(self.asset_components[sp_asset]), "crash", "spindle_front"
            )

    def machine_stoppage(
        self,
        ts: pd.Timestamp,
        asset_id: str,
        mode: str,
        hours: tuple[float, float],
        axis_event: int | None = None,
    ) -> None:
        dur = pd.Timedelta(hours=float(self.rng.uniform(*hours)))
        wo = self.new_wo(
            asset_id,
            "unplanned_repair",
            ts,
            started=ts,
            closed=ts + dur,
            failure_mode=mode,
            downtime_h=dur / pd.Timedelta(hours=1),
            components="",
        )
        self.log(ts, "unplanned_failure", asset_id=asset_id, mode=mode, wo_id=wo["wo_id"], components="")
        self.open_wo[asset_id] = wo["wo_id"]
        self.failed.add(asset_id)
        self.take_down(asset_id, ts, ts + dur)
        self.schedule(
            ts + dur, "stoppage_cleared", asset_id=asset_id, wo_id=wo["wo_id"], axis_event=axis_event
        )

    def on_stoppage_cleared(
        self, ts: pd.Timestamp, asset_id: str, wo_id: str, axis_event: int | None
    ) -> None:
        self.failed.discard(asset_id)
        if self.open_wo.get(asset_id) == wo_id:
            self.open_wo.pop(asset_id)
        if axis_event is not None:
            self.axis_events[axis_event]["replaced"] = True
            self.log(ts, "axis_replacement", asset_id=asset_id)

    def axis_multipliers(self, ts: pd.Timestamp) -> dict[str, float]:
        mult = {a: 1.0 for a in self.axes}
        for e in self.axis_events:
            if e["replaced"] or ts < e["start"]:
                continue
            frac = min(1.0, (ts - e["start"]) / (e["end"] - e["start"]))
            mult[e["axis"]] *= 1 + frac * (e["friction_end"] - 1)
        return mult

    def machine_hour(self, i: int, ts: pd.Timestamp, op: dict[str, Any], extra: dict[str, Any]) -> None:
        repair = self.cfg["other_stoppages"]["repair_hours"]
        if op["cut"] > 0 and self.machine_asset not in self.failed:
            for mode, p in self.stoppage_rate_h.items():
                if self.rng.random() < p * op["cut"] / SECONDS_PER_MINUTE:
                    self.machine_stoppage(ts, self.machine_asset, mode, tuple(repair[mode]))
                    break
        for k, e in enumerate(self.axis_events):
            if not e["done"] and ts >= e["end"] and op["cut"] > 0:
                e["done"] = True
                self.machine_stoppage(
                    ts,
                    f"{self.mid}-AX-{e['axis'].upper()}",
                    "ball_screw_or_guide",
                    tuple(repair["ball_screw_or_guide"]),
                    axis_event=k,
                )
        mult = self.axis_multipliers(ts)
        self.machine_hours.append(
            {
                "machine_id": self.mid,
                "hour_utc": ts,
                "down": op["down"],
                "cut_min": op["cut"],
                **{f"axis_mult_{a}": mult.get(a, np.nan) for a in ("x", "y", "z", "b")},
                **extra,
            }
        )

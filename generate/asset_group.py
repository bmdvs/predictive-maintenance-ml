"""Hour-by-hour plant state for one asset group, stepped one local day at a time.

An asset group is one HMC (spindle, TSC pump, flood pump) or the compressor pair. Each hour
advances operating state, episodes, collisions, bearing degradation, failures and maintenance
events; at the end of each local day the sensor-platform export renders and reduces the day's
bursts and applies the platform alert logic, whose alerts schedule inspections on later days.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from generate import maintenance, pump_wear
from generate.degradation import BearingInstance
from generate.lubrication import LubeEpisode
from generate.machine_behavior import MachineMixin
from generate.provenance import RunContext, derive_seed, rng_for
from generate.shop_schedule import LOCAL_TZ, _hm
from generate.vibration_synth import BurstInputs, render_burst
from pdm.config import Register
from pdm.constants import DAYS_PER_YEAR, HOURS_PER_DAY, SECONDS_PER_MINUTE
from pdm.features.burst import RESULTANT, ExtractionSettings, extract_burst, select_point_band
from pdm.features.kinematics import bearing_fault_orders, fault_frequencies_hz
from pdm.rules.vendor_baseline import VendorRuleSettings, VendorRuleState, compute_baseline

HOURS_PER_YEAR = DAYS_PER_YEAR * HOURS_PER_DAY
_ISO_KEY = {
    "spindle_front": "spindle",
    "spindle_rear": "spindle",
    "tsc_pump": "tsc_pump",
    "flood_pump": "flood_pump",
    "compressor_motor": "compressor",
    "compressor_airend": "compressor",
}
_TEMP_KEY = {
    "spindle_front": "spindle",
    "spindle_rear": "spindle",
    "tsc_pump": "pump",
    "flood_pump": "pump",
    "compressor_motor": "compressor_motor",
    "compressor_airend": "compressor_airend",
}


@dataclass
class PointRuntime:
    point_id: str
    point_class: str
    asset_id: str
    component: BearingInstance
    install: pd.Timestamp
    gateway: str
    rendered: bool
    rules: VendorRuleState
    resonance_hz: float
    mount_gain: float = 1.0
    fn_shift: float = 0.0
    baseline_start: pd.Timestamp | None = None
    baseline_rows: list[dict[str, float]] = field(default_factory=list)
    last_row: dict[str, dict[str, float]] | None = None
    last_temp: float | None = None
    last_weekly_ref: Any = None
    env_band: tuple[float, float] | None = None
    housing_gain_db: float = 0.0
    pending: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GroupOutput:
    scalars: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    sensor_events: list[dict[str, Any]] = field(default_factory=list)
    truth_hours: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    work_orders: list[dict[str, Any]] = field(default_factory=list)
    pm_rows: list[dict[str, Any]] = field(default_factory=list)
    bursts: list[tuple[dict[str, Any], np.ndarray]] = field(default_factory=list)
    bursts_rendered: int = 0
    render_seconds: float = 0.0


@dataclass
class Shared:
    """Inputs shared across groups: calendar, gateway outages, ambient."""

    frame: pd.DataFrame
    holidays: set[date]
    plant_minutes: np.ndarray
    gateway_outages: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]
    shop_ambient: np.ndarray
    room_ambient: np.ndarray
    rendered_points: set[str]
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    heat_episodes: list = field(default_factory=list)


class AssetGroup:
    """Base engine; subclasses define assets, operating state and PM tasks."""

    def __init__(
        self,
        ctx: RunContext,
        cfg: dict[str, Any],
        reg: Register,
        shared: Shared,
        group_id: str,
        overrides: dict[str, Any] | None = None,
    ):
        self.ctx, self.cfg, self.reg, self.sh, self.group_id = ctx, cfg, reg, shared, group_id
        self.overrides = overrides or {}
        self.rng = rng_for(ctx.run_seed, "group", group_id)
        self.settings = ExtractionSettings.from_config(cfg)
        self.out = GroupOutput()
        self.queue: list[tuple[pd.Timestamp, int, str, dict[str, Any]]] = []
        self._seq = itertools.count()
        self.components: dict[str, BearingInstance] = {}
        self.points: dict[str, PointRuntime] = {}
        self.asset_components: dict[str, list[str]] = {}
        self.asset_points: dict[str, list[str]] = {}
        self.open_wo: dict[str, str] = {}
        self.last_nff: dict[str, pd.Timestamp] = {}
        self.lube: dict[str, LubeEpisode | None] = {}
        self.lube_missed: dict[str, bool] = {}
        self.wo_count = 0
        self.failed: set[str] = set()
        self.scalar_sink = None
        self.sig = dict(cfg["bearing_signature"])
        self.shift_start_h = int(_hm(cfg["shop"]["calendar"]["shifts_local"]["first"]["start"]))

    # ---------------- setup helpers ----------------
    def add_component(
        self, comp_id: str, comp_class: str, asset_id: str, point_id: str | None, age_class: str, age_h: float
    ) -> BearingInstance:
        b = BearingInstance(
            component_id=comp_id,
            component_class=comp_class,
            asset_id=asset_id,
            point_id=point_id,
            age_class=age_class,
            params=self.cfg["bearing_degradation"],
            sig=self.sig,
            rng=rng_for(self.ctx.run_seed, "bearing", comp_id),
        )
        b.start(age_h)
        self.components[comp_id] = b
        self.asset_components.setdefault(asset_id, []).append(comp_id)
        return b

    def add_point(self, point_id: str, comp: BearingInstance) -> None:
        s = self.reg.sensors[point_id]
        pclass = self.reg.point_class(point_id)
        iso_key = _ISO_KEY[pclass]
        hf_key = "compressor" if pclass.startswith("compressor") else iso_key
        rules = VendorRuleState(point_id, VendorRuleSettings.for_point_class(self.cfg, iso_key, hf_key))
        fn = self.sig["resonance_fn_hz"][pclass]
        jit = self.sig["resonance_fn_jitter_frac_by_point"]
        prng = rng_for(self.ctx.run_seed, "point", point_id)
        install = pd.Timestamp(s.install_date, tz=LOCAL_TZ).tz_convert("UTC")
        pr = PointRuntime(
            point_id,
            pclass,
            s.asset_id,
            comp,
            install,
            s.gateway,
            point_id in self.sh.rendered_points,
            rules,
            fn * (1 + prng.uniform(-jit, jit)),
            baseline_start=install,
        )
        pr.housing_gain_db = float(prng.uniform(*self.sig["housing_resonance_gain_db"]))
        self.points[point_id] = pr
        comp.bearing_kind = self.reg.bearings[s.bearing].kind
        self.asset_points.setdefault(s.asset_id, []).append(point_id)
        self.out.sensor_events.append(
            {"point_id": point_id, "ts": install, "event": "install", "detail": s.mounting}
        )

    def schedule(self, ts: pd.Timestamp, kind: str, **payload: Any) -> None:
        heapq.heappush(self.queue, (ts, next(self._seq), kind, payload))

    def apply_overrides(self) -> None:
        for fi in self.overrides.get("forced_instances", []):
            comp = next((c for c in self.components.values() if c.point_id == fi["point_id"]), None)
            if comp is None:
                continue
            comp.forced = True
            comp.detectable = fi.get("inspection_detection", "normal") != "none"
            comp.beta_quantile = fi.get("beta_quantile")
            onset = pd.Timestamp(fi["onset_date"], tz=LOCAL_TZ).tz_convert("UTC")
            comp.remaining_h = np.inf
            self.schedule(onset, "forced_onset", comp_id=comp.component_id, defect=fi.get("defect"))

    def schedule_remounts(self) -> None:
        for rm in self.cfg["benign"]["sensor_remounts"]:
            if rm["point"] in self.points:
                ts = pd.Timestamp(rm["date"], tz=LOCAL_TZ).tz_convert("UTC")
                if self.sh.window_start <= ts < self.sh.window_end:
                    self.schedule(ts, "remount", point_id=rm["point"])

    # ---------------- events ----------------
    def log(self, ts: pd.Timestamp, event: str, **kw: Any) -> None:
        self.out.events.append({"ts": ts, "event": event, "group": self.group_id, **kw})

    def new_wo(self, asset_id: str, wo_type: str, opened: pd.Timestamp, **kw: Any) -> dict[str, Any]:
        self.wo_count += 1
        wo = {
            "wo_id": f"{self.group_id}-WO{self.wo_count:05d}",
            "asset_id": asset_id,
            "type": wo_type,
            "opened": opened,
            **kw,
        }
        self.out.work_orders.append(wo)
        return wo

    def rebaseline(self, asset_id: str, ts: pd.Timestamp, reason: str) -> None:
        for pid in self.asset_points.get(asset_id, []):
            p = self.points[pid]
            if p.pending:
                self.flush_pending(p)
            p.env_band = None
            p.baseline_start = ts
            p.baseline_rows = []
            p.rules.baseline = None
            p.rules.reset_counters()
            self.out.sensor_events.append(
                {"point_id": pid, "ts": ts, "event": "baselining_start", "detail": reason}
            )

    def replace(self, comp_ids: list[str], ts: pd.Timestamp, reason: str) -> None:
        for cid in comp_ids:
            c = self.components[cid]
            self.log(
                ts,
                "replacement",
                component_id=cid,
                instance_id=c.instance_id,
                stage=c.stage,
                reason=reason,
                onset_ts=c.onset_ts,
                defect=c.defect,
            )
            c.renew()

    def handle(self, ts: pd.Timestamp, kind: str, p: dict[str, Any]) -> None:
        getattr(self, f"on_{kind}")(ts, **p)

    def on_forced_onset(self, ts: pd.Timestamp, comp_id: str, defect: str | None) -> None:
        c = self.components[comp_id]
        if c.onset_ts is None:
            c.begin_onset(ts, "forced", defect=defect)
            self.log(
                ts,
                "bearing_onset",
                component_id=comp_id,
                instance_id=c.instance_id,
                cause="forced",
                defect=c.defect,
            )

    def on_remount(self, ts: pd.Timestamp, point_id: str) -> None:
        tf = self.sig["transfer_function_by_mounting"]
        p = self.points[point_id]
        p.mount_gain = float(self.rng.uniform(*tf["remount_gain_range"]))
        p.fn_shift = float(self.rng.uniform(*tf["remount_fn_shift_frac_range"]))
        self.out.sensor_events.append(
            {"point_id": point_id, "ts": ts, "event": "remount", "detail": f"gain={p.mount_gain:.3f}"}
        )
        self.log(ts, "benign", kind="sensor_remount", asset_id=p.asset_id, point_id=point_id)
        if p.pending:
            self.flush_pending(p)
        p.env_band = None
        p.baseline_start, p.baseline_rows, p.rules.baseline = ts, [], None
        p.rules.reset_counters()

    def on_inspection(self, ts: pd.Timestamp, asset_id: str, wo: dict[str, Any]) -> None:
        if self.open_wo.get(asset_id) != wo["wo_id"]:
            wo.update(started=pd.NaT, closed=ts, findings="superseded", no_fault_found=False)
            return
        findings = self.inspect(ts, asset_id)
        wo.update(
            started=ts,
            closed=ts + pd.Timedelta(hours=1),
            findings=",".join(findings) or None,
            no_fault_found=not findings,
        )
        self.log(ts, "inspection", asset_id=asset_id, wo_id=wo["wo_id"], findings=wo["findings"])
        if not findings:
            self.last_nff[asset_id] = ts
            self.open_wo.pop(asset_id, None)
            return
        repair = [f for f in findings if f.startswith(("bearing:", "wear"))]
        if not repair:
            self.open_wo.pop(asset_id, None)
            return
        lead = pd.Timedelta(
            days=float(self.rng.uniform(*self.cfg["maintenance"]["planned_repair_lead_days"]))
        )
        when = self.planned_window(ts + lead, asset_id)
        self.schedule(when, "planned_repair", asset_id=asset_id, findings=repair, inspection_wo=wo["wo_id"])

    def planned_window(self, ts: pd.Timestamp, asset_id: str) -> pd.Timestamp:
        return maintenance.next_saturday_on_or_after(ts, self.shift_start_h)

    def inspect(self, ts: pd.Timestamp, asset_id: str) -> list[str]:
        found: list[str] = []
        for cid in self.asset_components.get(asset_id, []):
            c = self.components[cid]
            if (
                c.stage > 0
                and c.detectable
                and self.rng.random() < maintenance.detection_probability(self.cfg, c.stage)
            ):
                found.append(f"bearing:{cid}")
        ep = self.lube.get(asset_id)
        if (
            ep is not None
            and ep.active(ts)
            and self.rng.random() < self.cfg["lubrication"]["inspection_detection_prob"]
        ):
            ep.clear(ts, "inspection")
            self.log(ts, "lube_cleared", asset_id=ep.asset_id, by="inspection")
            found.append("lubrication")
        found += self.inspect_extra(ts, asset_id)
        return found

    def inspect_extra(self, ts: pd.Timestamp, asset_id: str) -> list[str]:
        return []

    def on_planned_repair(
        self, ts: pd.Timestamp, asset_id: str, findings: list[str], inspection_wo: str
    ) -> None:
        if self.open_wo.get(asset_id) != inspection_wo:
            return
        comps = self.repair_scope(asset_id, findings)
        dur = maintenance.repair_duration(self.cfg, self.components[comps[0]].component_class, self.rng)
        wo = self.new_wo(
            asset_id,
            "condition_based",
            ts,
            started=ts,
            closed=ts + dur,
            findings=",".join(findings),
            downtime_h=dur / pd.Timedelta(hours=1),
            source_wo=inspection_wo,
        )
        self.log(ts, "planned_repair", asset_id=asset_id, wo_id=wo["wo_id"], components=",".join(comps))
        self.take_down(asset_id, ts, ts + dur)
        self.schedule(
            ts + dur,
            "return_to_service",
            asset_id=asset_id,
            comps=comps,
            reason="planned_repair",
            wo_id=wo["wo_id"],
        )

    def repair_scope(self, asset_id: str, findings: list[str]) -> list[str]:
        return [f.split(":", 1)[1] for f in findings if f.startswith("bearing:")] or list(
            self.asset_components[asset_id]
        )

    def on_return_to_service(
        self, ts: pd.Timestamp, asset_id: str, comps: list[str], reason: str, wo_id: str
    ) -> None:
        self.replace(comps, ts, reason)
        self.after_replacement(asset_id, ts)
        self.failed.discard(asset_id)
        if self.open_wo.get(asset_id) == wo_id:
            self.open_wo.pop(asset_id)
        self.rebaseline(asset_id, ts, reason)

    def after_replacement(self, asset_id: str, ts: pd.Timestamp) -> None:
        return None

    def unplanned_failure(
        self, ts: pd.Timestamp, asset_id: str, comps: list[str], mode: str, comp_class: str
    ) -> None:
        dur = maintenance.repair_duration(self.cfg, comp_class, self.rng)
        wo = self.new_wo(
            asset_id,
            "unplanned_repair",
            ts,
            started=ts,
            closed=ts + dur,
            failure_mode=mode,
            downtime_h=dur / pd.Timedelta(hours=1),
            components=",".join(comps),
        )
        instances = {c: self.components[c].instance_id for c in comps}
        self.log(
            ts,
            "unplanned_failure",
            asset_id=asset_id,
            mode=mode,
            wo_id=wo["wo_id"],
            components=",".join(comps),
            instances=str(instances),
            stages=",".join(str(self.components[c].stage) for c in comps),
        )
        self.open_wo[asset_id] = wo["wo_id"]
        self.failed.add(asset_id)
        self.cancel_planned(asset_id)
        self.take_down(asset_id, ts, ts + dur)
        self.schedule(
            ts + dur,
            "return_to_service",
            asset_id=asset_id,
            comps=comps,
            reason=f"unplanned_{mode}",
            wo_id=wo["wo_id"],
        )

    def cancel_planned(self, asset_id: str) -> None:
        self.queue = [
            q for q in self.queue if not (q[2] == "planned_repair" and q[3]["asset_id"] == asset_id)
        ]
        heapq.heapify(self.queue)

    def take_down(self, asset_id: str, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
        raise NotImplementedError

    # ---------------- alerts to inspections ----------------
    def respond_to_alert(self, alert: dict[str, Any]) -> None:
        asset_id = self.points[alert["point_id"]].asset_id
        ts = alert["alert_ts"]
        m = self.cfg["maintenance"]
        if asset_id in self.open_wo:
            alert["response"] = "open_work_order"
            return
        nff = self.last_nff.get(asset_id)
        recent_nff = nff is not None and ts - nff < pd.Timedelta(days=m["repeat_alert_window_days"])
        if recent_nff and self.rng.random() >= m["repeat_alert_inspection_prob"]:
            alert["response"] = "not_inspected_after_nff"
            return
        lo, hi = m["alert_to_inspection_working_days"]
        when = maintenance.add_working_days(
            ts, int(self.rng.integers(lo, hi + 1)), self.sh.holidays, self.shift_start_h + 1
        )
        wo = self.new_wo(asset_id, "inspection", ts, alert_id=alert["alert_id"])
        self.open_wo[asset_id] = wo["wo_id"]
        alert["response"] = wo["wo_id"]
        self.schedule(when, "inspection", asset_id=asset_id, wo=wo)

    # ---------------- PM ----------------
    def schedule_calendar_pm(
        self, asset_id: str, task: str, interval_days: float, first_due: pd.Timestamp | None = None
    ) -> None:
        due = first_due or self.sh.window_start + pd.Timedelta(days=float(self.rng.uniform(0, interval_days)))
        self.schedule(due, "pm_due", asset_id=asset_id, task=task, interval_days=interval_days, due=due)

    def on_pm_due(
        self, ts: pd.Timestamp, asset_id: str, task: str, interval_days: float, due: pd.Timestamp
    ) -> None:
        outcome, late = maintenance.pm_outcome(self.cfg, self.rng)
        row = {
            "asset_id": asset_id,
            "task": task,
            "basis": "calendar_days",
            "due": due,
            "outcome": outcome,
            "days_late": late if outcome != "skipped" else np.nan,
            "completed": due + pd.Timedelta(days=late) if outcome != "skipped" else pd.NaT,
        }
        self.out.pm_rows.append(row)
        if outcome != "skipped":
            self.schedule(row["completed"], "pm_done", asset_id=asset_id, task=task)
        self.on_pm_skipped(asset_id, task) if outcome == "skipped" else None
        nxt = due + pd.Timedelta(days=interval_days)
        self.schedule(nxt, "pm_due", asset_id=asset_id, task=task, interval_days=interval_days, due=nxt)

    def on_pm_skipped(self, asset_id: str, task: str) -> None:
        return None

    def on_pm_done(self, ts: pd.Timestamp, asset_id: str, task: str) -> None:
        self.log(ts, "pm_done", asset_id=asset_id, task=task)
        self.pm_effect(ts, asset_id, task)

    def pm_effect(self, ts: pd.Timestamp, asset_id: str, task: str) -> None:
        return None

    # ---------------- sensor platform ----------------
    def point_scheduled(self, pid: str, i: int) -> bool:
        raise NotImplementedError

    def point_condition(self, pid: str, i: int, ts: pd.Timestamp) -> dict[str, Any] | None:
        """Operating condition for a measurement, or None when the measurement state is not met."""
        raise NotImplementedError

    def in_outage(self, p: PointRuntime, ts: pd.Timestamp) -> bool:
        return any(a <= ts < b for a, b in self.sh.gateway_outages.get(p.gateway, [])) or any(
            a <= ts < b for a, b in self.battery_gaps.get(p.point_id, [])
        )

    def render_point_hour(
        self, p: PointRuntime, ts: pd.Timestamp, cond: dict[str, Any]
    ) -> tuple[dict, dict, np.ndarray]:
        comp = p.component
        s = self.settings
        orders = bearing_fault_orders(self.reg.bearings[self.reg.sensors[p.point_id].bearing])
        fault_hz = fault_frequencies_hz(orders, cond["shaft_hz"])
        sig = (
            self.sig
            if not p.point_class.endswith("pump")
            else {**self.sig, "_wear": self.cfg["pump"]["wear_signature_at_end"]}
        )
        inp = BurstInputs(
            point_class=p.point_class,
            fs=s.fs,
            n=int(round(s.fs * self.cfg["measurement"]["burst_s"])),
            band=s.band,
            hf_band=s.hf_band,
            shaft_hz=cond["shaft_hz"],
            rated_hz=cond["rated_hz"],
            load_pct=cond.get("load_pct", 0.0),
            flutes=cond.get("flutes", 0),
            fault_hz=fault_hz,
            resonance_hz=p.resonance_hz * (1 + p.fn_shift),
            mount_gain=p.mount_gain,
            hi=comp.signature_hi(),
            defect=comp.defect,
            lube_rise=cond.get("lube_rise", 0.0),
            cavitation_rise=cond.get("cavitation_rise", 0.0),
            wear_frac=cond.get("wear_frac", 0.0),
            lobe_hz=cond.get("lobe_hz", 0.0),
            am_inner=comp.am_inner,
            housing_gain_db=p.housing_gain_db,
        )
        th = self.cfg["bearing_degradation"]
        burst_rng = np.random.default_rng(derive_seed(self.ctx.run_seed, "burst", p.point_id, ts.isoformat()))
        burst = render_burst(
            inp,
            sig,
            self.cfg["regime"],
            th["stage_thresholds_hi"],
            th["failure"]["hi_threshold"],
            burst_rng,
        )
        return burst, fault_hz

    def reduce_burst(
        self, p: PointRuntime, burst: np.ndarray, cond: dict[str, Any], fault_hz: dict[str, float]
    ):
        return extract_burst(burst, self.settings, cond["shaft_hz"], fault_hz, band=p.env_band)

    def select_band(self, p: PointRuntime) -> None:
        if not p.pending:
            p.env_band = self.settings.sk_band
            return
        e0 = p.pending[0]
        rated = e0["cond"]["rated_hz"]
        orders = bearing_fault_orders(self.reg.bearings[self.reg.sensors[p.point_id].bearing])
        p.env_band = select_point_band(
            [e["burst"] for e in p.pending], self.settings, fault_frequencies_hz(orders, rated), rated
        )
        self.log(
            e0["ts"],
            "envelope_band_selected",
            point_id=p.point_id,
            band_lo_hz=p.env_band[0],
            band_hi_hz=p.env_band[1],
            resonance_hz=p.resonance_hz * (1 + p.fn_shift),
            housing_gain_db=p.housing_gain_db,
            bursts=len(p.pending),
        )
        self.out.sensor_events.append(
            {
                "point_id": p.point_id,
                "ts": e0["ts"],
                "event": "envelope_band",
                "detail": f"{p.env_band[0]:.0f}-{p.env_band[1]:.0f}",
            }
        )

    def flush_pending(self, p: PointRuntime) -> None:
        """Reduce the held baselining bursts with the point's selected band and emit their rows."""
        if p.env_band is None:
            self.select_band(p)
        for e in p.pending:
            rows, diags = self.reduce_burst(p, e["burst"], e["cond"], e["fault_hz"])
            self.emit_measured(p, e, rows, diags, baseline=True)
        p.pending = []

    def emit_measured(
        self, p: PointRuntime, e: dict[str, Any], rows: dict, diags: dict, baseline: bool
    ) -> dict:
        temp, kind = e["temp"], None
        if e["radio"]:
            rows, temp, kind = self.corrupt(p, rows, temp)
        p.last_row = rows
        p.last_temp = temp
        cond = e["cond"]
        for axis in self.reported_axes(p):
            self.out.scalars.append(
                {
                    **e["base"],
                    "axis": axis,
                    "measurement_state": "measured",
                    "no_measurement": False,
                    "speed_rpm_at_measurement": cond["shaft_hz"] * SECONDS_PER_MINUTE,
                    "load_pct_at_measurement": cond.get("load_pct_report", np.nan),
                    "temp_c": temp,
                    **rows[axis],
                }
            )
        truth = e["truth"]
        if truth is not None:
            truth.update(
                measured=True,
                radio_noise=kind,
                temp_c=temp,
                **{f"res_{k}": v for k, v in diags[RESULTANT].items()},
                **{
                    f"y_{k}": v
                    for k, v in diags["y"].items()
                    if k.startswith(("envr_", "envp_", "env_floor", "sk_"))
                },
            )
        if baseline:
            p.baseline_rows.append(rows[RESULTANT])
        return rows

    def temperature(self, p: PointRuntime, i: int, ts: pd.Timestamp, cond: dict[str, Any]) -> float:
        rise = self.cfg["sensor_temperature_rise_c"][_TEMP_KEY[p.point_class]]
        if p.point_class.startswith("spindle"):
            rise *= cond["shaft_hz"] / cond["rated_hz"]
        amb = self.sh.room_ambient[i] if p.point_class.startswith("compressor") else self.sh.shop_ambient[i]
        th = self.cfg["bearing_degradation"]
        s4lo, hf = th["stage_thresholds_hi"]["stage4"], th["failure"]["hi_threshold"]
        s4 = (
            float(np.clip((p.component.signature_hi() - s4lo) / (hf - s4lo), 0, 1))
            if p.component.defect
            else 0.0
        )
        t = amb + rise + s4 * self.sig["stage4"]["temp_rise_c_at_failure"]
        t += (
            cond.get("lube_temp", 0.0)
            + cond.get("wear_frac", 0.0) * self.cfg["pump"]["wear_signature_at_end"]["temp_rise_c"]
        )
        return float(t)

    def reported_axes(self, p: PointRuntime) -> list[str]:
        ax = self.cfg["indicators"]["axes_reported"]
        if p.point_class.startswith("spindle"):
            return [RESULTANT, self.reg.sensors[p.point_id].radial_axis]
        names = ax["pump" if p.point_class.endswith("pump") else "compressor"]
        return [RESULTANT if a == "resultant" else a for a in names]

    def corrupt(
        self, p: PointRuntime, rows: dict[str, dict[str, float]], temp: float
    ) -> tuple[dict, float, str]:
        modes = self.cfg["sensor_behavior"]["radio_noise_modes"]
        kind = str(self.rng.choice(["temp_jump", "spike", "stuck"]))
        if kind == "stuck" and p.last_row is not None:
            return (
                {a: dict(v) for a, v in p.last_row.items()},
                p.last_temp if p.last_temp is not None else temp,
                kind,
            )
        if kind == "temp_jump":
            return (
                rows,
                temp + float(self.rng.choice([-1, 1]) * self.rng.uniform(*modes["temp_jump_c"])),
                kind,
            )
        g = float(self.rng.uniform(*modes["spike_gain"]))
        out = {}
        for a, r in rows.items():
            out[a] = dict(r)
            for k in ("accel_rms_g", "peak_g", "vel_rms_mm_s", "hf_rms_g"):
                out[a][k] = r[k] * g
        return out, temp, "spike"

    def end_of_day(self, idx: list[int]) -> None:
        stamp = self.ctx.stamp()
        meas = self.cfg["measurement"]
        ref_day = meas["retention"]["weekly_reference_weekday"]
        for pid, p in self.points.items():
            if not p.rendered:
                continue
            for i in idx:
                ts = self.sh.frame["hour_utc"].iat[i]
                if ts < p.install or not self.point_scheduled(pid, i) or self.in_outage(p, ts):
                    continue
                cond = self.point_condition(pid, i, ts)
                base = {"point_id": pid, "hour_utc": ts, **stamp}
                truth = self.truth_index.get((pid, i))
                if cond is None:
                    self.out.scalars.append(
                        {**base, "axis": RESULTANT, "measurement_state": "not_met", "no_measurement": True}
                    )
                    p.rules.update(ts, None)
                    if truth is not None:
                        truth["measured"] = False
                    continue
                t0 = time.perf_counter()
                burst, fault_hz = self.render_point_hour(p, ts, cond)
                temp = self.temperature(p, i, ts, cond)
                radio = self.rng.random() < self.cfg["sensor_behavior"]["radio_noise_row_frac"]
                entry = {
                    "ts": ts,
                    "base": base,
                    "cond": cond,
                    "burst": burst,
                    "fault_hz": fault_hz,
                    "temp": temp,
                    "radio": radio,
                    "truth": truth,
                }
                in_baselining = p.baseline_start is not None and ts < p.baseline_start + pd.Timedelta(
                    days=meas["baselining_days"]
                )
                alert = None
                if in_baselining:
                    p.pending.append(entry)
                    p.rules.update(ts, None)
                else:
                    if p.pending or p.env_band is None:
                        self.flush_pending(p)
                    if p.rules.baseline is None and p.baseline_start is not None:
                        if p.baseline_rows:
                            p.rules.baseline = compute_baseline(
                                pd.DataFrame(p.baseline_rows), p.rules.settings, meas["baseline_statistic"]
                            )
                            self.out.sensor_events.append(
                                {"point_id": pid, "ts": ts, "event": "baselining_end", "detail": ""}
                            )
                        p.baseline_start = None
                    rows, diags = self.reduce_burst(p, burst, cond, fault_hz)
                    rows = self.emit_measured(p, entry, rows, diags, baseline=False)
                    alert = p.rules.update(ts, rows[RESULTANT]) if p.rules.baseline is not None else None
                self.out.render_seconds += time.perf_counter() - t0
                self.out.bursts_rendered += 1
                local = ts.tz_convert(LOCAL_TZ)
                week = local.isocalendar()[:2]
                weekly = local.strftime("%a").lower() == ref_day and p.last_weekly_ref != week
                if weekly:
                    p.last_weekly_ref = week
                if alert is not None:
                    alert.update(stamp, alert_id=f"{pid}-{ts:%Y%m%d%H}")
                    self.out.alerts.append(alert)
                    self.respond_to_alert(alert)
                if alert is not None or weekly:
                    meta = {
                        "point_id": pid,
                        "hour_utc": ts,
                        "reason": "alert" if alert is not None else "weekly_reference",
                        "fs_hz": self.settings.fs,
                        "speed_rpm": cond["shaft_hz"] * SECONDS_PER_MINUTE,
                        **stamp,
                    }
                    self.out.bursts.append((meta, burst.astype(np.float32)))

    # ---------------- main loop ----------------
    def hour_step(self, i: int, ts: pd.Timestamp) -> None:
        raise NotImplementedError

    def run(self) -> GroupOutput:
        frame = self.sh.frame
        self.truth_index: dict[tuple[str, int], dict[str, Any]] = {}
        local_dates = frame["local_date"].to_numpy()
        months = pd.to_datetime(frame["local_date"]).dt.strftime("%Y-%m").to_numpy()
        day_idx: list[int] = []
        for i in range(len(frame)):
            ts = frame["hour_utc"].iat[i]
            while self.queue and self.queue[0][0] < ts + pd.Timedelta(hours=1):
                qts, _, kind, payload = heapq.heappop(self.queue)
                self.handle(max(qts, ts), kind, payload)
            self.hour_step(i, ts)
            day_idx.append(i)
            if i == len(frame) - 1 or local_dates[i + 1] != local_dates[i]:
                self.end_of_day(day_idx)
                day_idx = []
                if (
                    self.scalar_sink is not None
                    and self.out.scalars
                    and (i == len(frame) - 1 or months[i + 1] != months[i])
                ):
                    self.scalar_sink(months[i], self.out.scalars)
                    self.out.scalars = []
        for p in self.points.values():
            if p.pending:
                self.flush_pending(p)
        self.emit_gap_events()
        return self.out

    def emit_gap_events(self) -> None:
        for pid, p in self.points.items():
            for a, b in getattr(self, "battery_gaps", {}).get(pid, []):
                self.out.sensor_events.append(
                    {"point_id": pid, "ts": a, "event": "battery_gap_start", "detail": ""}
                )
                self.out.sensor_events.append(
                    {"point_id": pid, "ts": b, "event": "battery_change", "detail": ""}
                )
            for a, b in self.sh.gateway_outages.get(p.gateway, []):
                if a >= p.install:
                    self.out.sensor_events.append(
                        {"point_id": pid, "ts": a, "event": "dropout", "detail": f"{b.isoformat()}"}
                    )


class HmcGroup(MachineMixin, AssetGroup):
    def __init__(
        self,
        ctx,
        cfg,
        reg,
        shared,
        machine_id: str,
        schedule_frame: pd.DataFrame,
        overrides=None,
        axis_events=None,
    ):
        super().__init__(ctx, cfg, reg, shared, machine_id, overrides)
        self.init_machine(machine_id, schedule_frame, axis_events or [])
        sp_asset, tsc_asset, cp_asset = f"{machine_id}-SP", f"{machine_id}-TSC", f"{machine_id}-CP"
        age_class = self.machine["age_class"]
        spin = reg.hmc_spindles[machine_id]
        front = self.add_component(
            f"{machine_id}-SP-F",
            "spindle_front",
            sp_asset,
            f"{machine_id}-SP-F",
            age_class,
            spin["running_h_at_record_start"],
        )
        rear = self.add_component(
            f"{machine_id}-SP-R",
            "spindle_rear",
            sp_asset,
            f"{machine_id}-SP-R",
            age_class,
            spin["running_h_at_record_start"],
        )
        tsc = self.add_component(
            f"{machine_id}-TSC",
            "tsc_pump",
            tsc_asset,
            f"{machine_id}-TSC",
            age_class,
            reg.pumps[tsc_asset]["running_h_at_record_start"],
        )
        cp = self.add_component(
            f"{machine_id}-CP",
            "flood_pump",
            cp_asset,
            f"{machine_id}-CP",
            age_class,
            reg.pumps[cp_asset]["running_h_at_record_start"],
        )
        for pid, comp in (
            (f"{machine_id}-SP-F", front),
            (f"{machine_id}-SP-R", rear),
            (f"{machine_id}-TSC", tsc),
            (f"{machine_id}-CP", cp),
        ):
            self.add_point(pid, comp)
        self.pumps = {
            tsc_asset: pump_wear.PumpState(tsc_asset, "tsc", []),
            cp_asset: pump_wear.PumpState(cp_asset, "flood", []),
        }
        self.lube[sp_asset] = None
        self.lube_missed[sp_asset] = False
        self.sp_asset = sp_asset
        self.spindle_run_h = float(spin["running_h_at_record_start"])
        self.battery_gaps = battery_gaps(cfg, self.points, shared, ctx)
        pm = cfg["maintenance"]["pm_program"]["hmc"]
        self.schedule_calendar_pm(sp_asset, "spindle_lube_check", pm["spindle_lube_check"]["interval"])
        self.schedule_calendar_pm(sp_asset, "drawbar_force", pm["drawbar_force"]["interval"])
        self.schedule_calendar_pm(tsc_asset, "tsc_filter", pm["tsc_filter"]["interval"])
        self.schedule_calendar_pm(machine_id, "coolant_check", pm["coolant_check"]["interval"])
        self.schedule_calendar_pm(machine_id, "way_lube_hydraulic", pm["way_lube_hydraulic"]["interval"])
        self.schedule_calendar_pm(machine_id, "backlash_check", pm["backlash_check"]["interval"])
        last = reg.pumps[tsc_asset]["last_replacement"]
        due = pd.Timestamp(str(last), tz=LOCAL_TZ).tz_convert("UTC") + pd.Timedelta(
            days=pm["tsc_pump_unit_replacement"]["interval"]
        )
        self.schedule_calendar_pm(
            tsc_asset,
            "tsc_pump_unit_replacement",
            pm["tsc_pump_unit_replacement"]["interval"],
            max(due, shared.window_start),
        )
        self.rebuild_due_h = pm["spindle_rebuild_older"]["interval"] if age_class == "older" else np.inf
        self.rebuild_pending = False
        self.schedule_remounts()
        self.apply_overrides()

    def take_down(self, asset_id: str, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
        self.down_until = max(self.down_until, t1)

    def planned_window(self, ts: pd.Timestamp, asset_id: str) -> pd.Timestamp:
        return maintenance.next_saturday_on_or_after(ts, self.shift_start_h)

    def repair_scope(self, asset_id: str, findings: list[str]) -> list[str]:
        return list(self.asset_components[asset_id])

    def after_replacement(self, asset_id: str, ts: pd.Timestamp) -> None:
        if asset_id in self.pumps:
            self.pumps[asset_id].reset()
        if asset_id == self.sp_asset:
            self.spindle_run_h = 0.0
            self.rebuild_pending = False

    def inspect_extra(self, ts: pd.Timestamp, asset_id: str) -> list[str]:
        found = []
        pump = self.pumps.get(asset_id)
        if pump is None:
            return found
        if (
            pump.cavitation is not None
            and pump.cavitation.active(ts)
            and self.cfg["maintenance"]["cavitation_found_if_active"]
        ):
            pump.cavitation.clear(ts, "inspection")
            found.append("cavitation")
        stage = maintenance.wear_stage(self.cfg, pump.wear_fraction(ts))
        if stage > 0 and self.rng.random() < maintenance.detection_probability(self.cfg, stage):
            found.append("wear")
        return found

    def on_pm_skipped(self, asset_id: str, task: str) -> None:
        if task == "spindle_lube_check":
            self.lube_missed[asset_id] = True

    def pm_effect(self, ts: pd.Timestamp, asset_id: str, task: str) -> None:
        if task == "spindle_lube_check":
            self.lube_missed[asset_id] = False
            ep = self.lube.get(asset_id)
            if ep is not None and ep.active(ts):
                ep.clear(ts, "lube_pm")
                self.log(ts, "lube_cleared", asset_id=asset_id, by="lube_pm")
        elif task == "coolant_check":
            for pump in self.pumps.values():
                if pump.cavitation is not None and pump.cavitation.active(ts):
                    pump.cavitation.clear(ts, "coolant_check")
        elif task == "tsc_pump_unit_replacement":
            dur = maintenance.repair_duration(self.cfg, "tsc_pump", self.rng)
            self.new_wo(
                asset_id,
                "pm",
                ts,
                started=ts,
                closed=ts + dur,
                task=task,
                downtime_h=dur / pd.Timedelta(hours=1),
            )
            self.take_down(asset_id, ts, ts + dur)
            self.schedule(
                ts + dur,
                "return_to_service",
                asset_id=asset_id,
                comps=list(self.asset_components[asset_id]),
                reason="pm_replacement",
                wo_id="pm",
            )
        elif task == "spindle_rebuild_older":
            dur = maintenance.repair_duration(self.cfg, "spindle_front", self.rng)
            self.new_wo(
                asset_id,
                "pm",
                ts,
                started=ts,
                closed=ts + dur,
                task=task,
                downtime_h=dur / pd.Timedelta(hours=1),
            )
            self.take_down(asset_id, ts, ts + dur)
            self.schedule(
                ts + dur,
                "return_to_service",
                asset_id=asset_id,
                comps=list(self.asset_components[asset_id]),
                reason="preventive_rebuild",
                wo_id="pm",
            )

    def point_scheduled(self, pid: str, i: int) -> bool:
        return self.sched["scheduled_min"].iat[i] > 0

    def point_condition(self, pid: str, i: int, ts: pd.Timestamp) -> dict[str, Any] | None:
        op = self.ops[i]
        if op["cut"] <= 0:
            return None
        p = self.points[pid]
        rated_hz = self.machine["rated_rpm"] / SECONDS_PER_MINUTE
        if p.point_class.startswith("spindle"):
            if op["speed"] < self.cfg["measurement"]["state_rule"]["spindle_min_rpm"]:
                return None
            ep = self.lube.get(self.sp_asset)
            sev = ep.severity(ts) if ep is not None else 0.0
            return {
                "shaft_hz": op["speed"] / SECONDS_PER_MINUTE,
                "rated_hz": rated_hz,
                "load_pct": op["load"],
                "load_pct_report": op["load"],
                "flutes": op["flutes"],
                "lube_rise": sev * (ep.hf_rise if ep else 0.0),
                "lube_temp": sev * (ep.temp_rise_c if ep else 0.0),
            }
        pump = self.pumps[p.asset_id]
        if (pump.kind == "tsc" and not op["tsc"]) or (pump.kind == "flood" and not op["flood"]):
            return None
        hz = self.reg.pumps[p.asset_id]["motor_rpm"] / SECONDS_PER_MINUTE
        return {
            "shaft_hz": hz,
            "rated_hz": hz,
            "cavitation_rise": pump.cavitation_rise(ts),
            "wear_frac": pump.wear_fraction(ts),
        }

    def hour_step(self, i: int, ts: pd.Timestamp) -> None:
        if i == 0:
            self.ops: dict[int, dict[str, Any]] = {}
        op = self.operating(i, ts)
        self.ops[i] = op
        cfg = self.cfg
        run = {"spindle": op["cut"] / SECONDS_PER_MINUTE}
        run["tsc"] = run["spindle"] * op["tsc"]
        run["flood"] = run["spindle"] * op["flood"]

        lub = cfg["lubrication"]
        ep = self.lube[self.sp_asset]
        if ep is not None and ep.finished(ts):
            self.lube[self.sp_asset] = ep = None
        if ep is None:
            rate = lub["episodes_per_spindle_year"][f"{self.machine['age_class']}_hmc"]
            rate *= lub["missed_lube_pm_rate_multiplier"] if self.lube_missed[self.sp_asset] else 1.0
            if self.rng.random() < pump_wear.hourly_probability(rate, 1.0, HOURS_PER_YEAR):
                ep = LubeEpisode.draw(cfg, self.sp_asset, ts, self.rng)
                self.lube[self.sp_asset] = ep
                self.log(
                    ts,
                    "lube_episode_start",
                    asset_id=self.sp_asset,
                    hf_rise=ep.hf_rise,
                    window_days=ep.window_days,
                )
        if ep is not None and ep.conversion_due(ts):
            ep.converted = True
            front = self.components[f"{self.mid}-SP-F"]
            if front.onset_ts is None:
                front.begin_onset(ts, "lubrication")
                self.log(
                    ts,
                    "bearing_onset",
                    component_id=front.component_id,
                    instance_id=front.instance_id,
                    cause="lubrication",
                    defect=front.defect,
                )

        pc = cfg["pump"]
        for asset_id, pump in self.pumps.items():
            k = pump.kind
            p_wear = {
                "seal": pump_wear.hourly_probability(
                    pc["seal_wear"]["onset_rate_per_pump_year"], 1.0, HOURS_PER_YEAR
                ),
                "impeller": pump_wear.hourly_probability(
                    pc["impeller_wear"]["onset_rate_per_pump_year"], 1.0, HOURS_PER_YEAR
                ),
            }
            for kind in pump_wear.maybe_start_wear(cfg, pump, ts, p_wear, self.rng):
                self.log(ts, "pump_wear_start", asset_id=asset_id, kind=kind)
            cav = pump.cavitation
            if cav is not None and not cav.active(ts):
                pump.cavitation = cav = None
            if cav is None and self.rng.random() < pump_wear.hourly_probability(
                pc["cavitation"]["rate_per_pump_year"][k], 1.0, HOURS_PER_YEAR
            ):
                pump.cavitation = pump_wear.Cavitation(
                    ts, float(self.rng.uniform(*pc["cavitation"]["hf_rise_frac"]))
                )
                self.log(ts, "cavitation_start", asset_id=asset_id, hf_rise=pump.cavitation.hf_rise)
            elif cav is not None:
                local_h = self.sh.frame["local_hour"].iat[i]
                if ts - cav.start >= pd.Timedelta(days=pc["cavitation"]["max_duration_days"]) or (
                    local_h == self.shift_start_h
                    and self.rng.random() < pc["cavitation"]["operator_topup_prob_per_day"]
                ):
                    cav.clear(ts, "operator")
            seal_rate = pc["seal_failure"]["rate_per_pump_year"][k] * (
                pc["seal_failure"]["wear_hazard_multiplier"] if pump.wear_active(ts) else 1.0
            )
            pump_run = run["tsc" if k == "tsc" else "flood"]
            if (
                pump_run > 0
                and self.rng.random() < pump_wear.hourly_probability(seal_rate, 1.0, HOURS_PER_YEAR)
                and asset_id not in self.failed
            ):
                self.unplanned_failure(
                    ts, asset_id, list(self.asset_components[asset_id]), "seal_failure", f"{k}_pump"
                )

        self.machine_crash(ts, op, self.components[f"{self.mid}-SP-F"], self.sp_asset)

        lube_active = ep is not None and ep.active(ts)
        life_rate = (
            1.0 / cfg["bearing_degradation"]["hazard_multipliers"]["lubrication_loss"] if lube_active else 1.0
        )
        for cid, c in self.components.items():
            r = (
                run["spindle"]
                if c.component_class.startswith("spindle")
                else run["tsc" if c.component_class == "tsc_pump" else "flood"]
            )
            lr = life_rate if c.component_class.startswith("spindle") else 1.0
            for ev in c.step(ts, r, lr):
                self.log(
                    ts,
                    ev["event"],
                    component_id=cid,
                    instance_id=c.instance_id,
                    **{k: v for k, v in ev.items() if k != "event"},
                )
            if c.failure_due is not None and ts >= c.failure_due and r > 0 and c.asset_id not in self.failed:
                self.unplanned_failure(
                    ts, c.asset_id, list(self.asset_components[c.asset_id]), "bearing", c.component_class
                )
        self.spindle_run_h += run["spindle"]
        if self.spindle_run_h >= self.rebuild_due_h and not self.rebuild_pending:
            self.rebuild_pending = True
            outcome, late = maintenance.pm_outcome(cfg, self.rng)
            late = late if outcome == "late" else 0.0
            when = maintenance.next_saturday_on_or_after(ts + pd.Timedelta(days=late), self.shift_start_h)
            self.out.pm_rows.append(
                {
                    "asset_id": self.sp_asset,
                    "task": "spindle_rebuild_older",
                    "basis": "spindle_running_h",
                    "due": ts,
                    "outcome": outcome,
                    "days_late": late,
                    "completed": when,
                }
            )
            self.schedule(when, "pm_done", asset_id=self.sp_asset, task="spindle_rebuild_older")

        for pid, p in self.points.items():
            c = p.component
            pump = self.pumps.get(p.asset_id)
            row = {
                "point_id": pid,
                "hour_utc": ts,
                "instance_id": c.instance_id,
                "stage": c.stage,
                "hi": c.h if c.onset_ts is not None else 0.0,
                "defect": c.defect,
                "onset_cause": c.onset_cause,
                "forced": c.forced,
                "lube_severity": (
                    self.lube[self.sp_asset].severity(ts)
                    if pid.endswith(("SP-F", "SP-R")) and self.lube[self.sp_asset]
                    else 0.0
                ),
                "cavitation": bool(pump and pump.cavitation_rise(ts) > 0),
                "wear_frac": pump.wear_fraction(ts) if pump else 0.0,
                "down": op["down"],
                "scheduled": self.point_scheduled(pid, i),
                "measured": None,
            }
            self.out.truth_hours.append(row)
            self.truth_index[(pid, i)] = row
        front = self.components[f"{self.mid}-SP-F"]
        th = cfg["bearing_degradation"]
        s4lo, hif = th["stage_thresholds_hi"]["stage4"], th["failure"]["hi_threshold"]
        s4 = float(np.clip((front.signature_hi() - s4lo) / (hif - s4lo), 0, 1)) if front.defect else 0.0
        tsc, cp = self.pumps[f"{self.mid}-TSC"], self.pumps[f"{self.mid}-CP"]
        self.machine_hour(
            i,
            ts,
            op,
            {
                "spindle_s4": s4,
                "spindle_stage": front.stage,
                "lube_temp_c": (ep.severity(ts) * ep.temp_rise_c) if ep is not None else 0.0,
                "tsc_wear": tsc.wear_fraction(ts),
                "tsc_cavitation": tsc.cavitation_rise(ts) > 0,
                "cp_cavitation": cp.cavitation_rise(ts) > 0,
            },
        )


class CompressorGroup(AssetGroup):
    def __init__(self, ctx, cfg, reg, shared, air: pd.DataFrame, overrides=None):
        super().__init__(ctx, cfg, reg, shared, "CMP", overrides)
        self.air = air
        self.cids = list(reg.compressors)
        self.down_until = {c: shared.window_start for c in self.cids}
        for cid in self.cids:
            c = reg.compressors[cid]
            m = self.add_component(
                f"{cid}-MDE",
                "compressor_motor",
                cid,
                f"{cid}-MDE",
                c["age_class"],
                c["motor_bearing_running_h_at_record_start"],
            )
            a = self.add_component(
                f"{cid}-AE",
                "compressor_airend",
                cid,
                f"{cid}-AE",
                c["age_class"],
                c["airend_running_h_at_record_start"],
            )
            self.add_point(f"{cid}-MDE", m)
            self.add_point(f"{cid}-AE", a)
            self.lube[cid] = None
        self.run_h = {c: 0.0 for c in self.cids}
        self.next_filter_h = {
            c: cfg["maintenance"]["pm_program"]["compressor"]["oil_filter_change"]["interval"]
            for c in self.cids
        }
        self.battery_gaps = battery_gaps(cfg, self.points, shared, ctx)
        for cid in self.cids:
            self.schedule_calendar_pm(
                cid, "oil_sample", cfg["maintenance"]["pm_program"]["compressor"]["oil_sample"]["interval"]
            )
        duty = next(c for c, v in reg.compressors.items() if v["role"] == "duty")
        rec = cfg["meta"]["record"]
        r0 = pd.Timestamp(str(rec["start"]), tz=LOCAL_TZ).tz_convert("UTC")
        r1 = pd.Timestamp(str(rec["end"]), tz=LOCAL_TZ).tz_convert("UTC")
        yr = pd.Timedelta(days=DAYS_PER_YEAR)
        t = r0
        while t < r1:
            ts = t + pd.Timedelta(seconds=float(self.rng.uniform(0, yr.total_seconds())))
            if shared.window_start <= ts < shared.window_end:
                self.schedule(ts, "airend_lube", cid=duty)
            t += yr
        years = (shared.window_end - shared.window_start) / pd.Timedelta(days=DAYS_PER_YEAR)
        self.duty_running_h_per_year = float(air[f"{duty}_running"].sum()) / years
        self.schedule_remounts()
        self.apply_overrides()

    def take_down(self, asset_id: str, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
        self.down_until[asset_id] = max(self.down_until[asset_id], t1)

    def maybe_trip(self, ts: pd.Timestamp, cid: str, run_h: float) -> None:
        c = self.cfg["compressor"]
        if run_h <= 0 or cid in self.failed or self.duty_running_h_per_year <= 0:
            return
        p = pump_wear.hourly_probability(
            c["other_stoppages_per_running_year"], run_h, self.duty_running_h_per_year
        )
        if self.rng.random() >= p:
            return
        dur = pd.Timedelta(hours=float(self.rng.uniform(*c["other_stoppage_repair_hours"])))
        wo = self.new_wo(
            cid,
            "unplanned_repair",
            ts,
            started=ts,
            closed=ts + dur,
            failure_mode="trip",
            downtime_h=dur / pd.Timedelta(hours=1),
            components="",
        )
        self.log(ts, "unplanned_failure", asset_id=cid, mode="trip", wo_id=wo["wo_id"], components="")
        self.open_wo[cid] = wo["wo_id"]
        self.failed.add(cid)
        self.take_down(cid, ts, ts + dur)
        self.schedule(ts + dur, "trip_cleared", asset_id=cid, wo_id=wo["wo_id"])

    def on_trip_cleared(self, ts: pd.Timestamp, asset_id: str, wo_id: str) -> None:
        self.failed.discard(asset_id)
        if self.open_wo.get(asset_id) == wo_id:
            self.open_wo.pop(asset_id)

    def planned_window(self, ts: pd.Timestamp, asset_id: str) -> pd.Timestamp:
        return ts

    def on_airend_lube(self, ts: pd.Timestamp, cid: str) -> None:
        if self.lube[cid] is None:
            self.lube[cid] = LubeEpisode.draw(self.cfg, cid, ts, self.rng)
            self.log(
                ts,
                "lube_episode_start",
                asset_id=cid,
                hf_rise=self.lube[cid].hf_rise,
                window_days=self.lube[cid].window_days,
            )

    def pm_effect(self, ts: pd.Timestamp, asset_id: str, task: str) -> None:
        if task == "oil_filter_change":
            ep = self.lube.get(asset_id)
            if ep is not None and ep.active(ts):
                ep.clear(ts, "oil_change")
                self.log(ts, "lube_cleared", asset_id=ep.asset_id, by="oil_change")

    def point_scheduled(self, pid: str, i: int) -> bool:
        return self.sh.plant_minutes[i] > 0

    def running(self, i: int, ts: pd.Timestamp) -> dict[str, bool]:
        r = {c: bool(self.air[f"{c}_running"].iat[i]) and ts >= self.down_until[c] for c in self.cids}
        lead = self.air["lead"].iat[i]
        if self.air[f"{lead}_running"].iat[i] and not r[lead]:
            for c in self.cids:
                if c != lead and ts >= self.down_until[c]:
                    r[c] = True
        return r

    def point_condition(self, pid: str, i: int, ts: pd.Timestamp) -> dict[str, Any] | None:
        p = self.points[pid]
        cid = p.asset_id
        if not self.ops[i][cid]:
            return None
        c = self.reg.compressors[cid]
        motor_hz = c["motor_rpm"] / SECONDS_PER_MINUTE
        load = float(self.air[f"{cid}_load_frac"].iat[i])
        if p.point_class == "compressor_motor":
            return {"shaft_hz": motor_hz, "rated_hz": motor_hz, "load_pct_report": load * 100}
        rotor = motor_hz * c["gear_ratio"]
        ep = self.lube[cid]
        sev = ep.severity(ts) if ep is not None else 0.0
        return {
            "shaft_hz": rotor,
            "rated_hz": rotor,
            "lobe_hz": rotor * c["male_rotor_lobes"],
            "load_pct_report": load * 100,
            "lube_rise": sev * (ep.hf_rise if ep else 0.0),
            "lube_temp": sev * (ep.temp_rise_c if ep else 0.0),
        }

    def hour_step(self, i: int, ts: pd.Timestamp) -> None:
        if i == 0:
            self.ops: dict[int, dict[str, bool]] = {}
        r = self.running(i, ts)
        self.ops[i] = r
        for cid in self.cids:
            ep = self.lube[cid]
            if ep is not None and ep.finished(ts):
                self.lube[cid] = ep = None
            if ep is not None and ep.conversion_due(ts):
                ep.converted = True
                ae = self.components[f"{cid}-AE"]
                if ae.onset_ts is None:
                    ae.begin_onset(ts, "lubrication")
                    self.log(
                        ts,
                        "bearing_onset",
                        component_id=ae.component_id,
                        instance_id=ae.instance_id,
                        cause="lubrication",
                        defect=ae.defect,
                    )
            run_h = 1.0 if r[cid] else 0.0
            self.run_h[cid] += run_h
            self.maybe_trip(ts, cid, run_h)
            if self.run_h[cid] >= self.next_filter_h[cid]:
                self.next_filter_h[cid] += self.cfg["maintenance"]["pm_program"]["compressor"][
                    "oil_filter_change"
                ]["interval"]
                outcome, late = maintenance.pm_outcome(self.cfg, self.rng)
                done = ts + pd.Timedelta(days=late) if outcome != "skipped" else pd.NaT
                self.out.pm_rows.append(
                    {
                        "asset_id": cid,
                        "task": "oil_filter_change",
                        "basis": "running_h",
                        "due": ts,
                        "outcome": outcome,
                        "days_late": late if outcome != "skipped" else np.nan,
                        "completed": done,
                    }
                )
                if outcome != "skipped":
                    self.schedule(done, "pm_done", asset_id=cid, task="oil_filter_change")
            lube_active = ep is not None and ep.active(ts)
            for comp_id in self.asset_components[cid]:
                c = self.components[comp_id]
                lr = (
                    1.0 / self.cfg["bearing_degradation"]["hazard_multipliers"]["lubrication_loss"]
                    if (lube_active and c.component_class == "compressor_airend")
                    else 1.0
                )
                for ev in c.step(ts, run_h, lr):
                    self.log(
                        ts,
                        ev["event"],
                        component_id=comp_id,
                        instance_id=c.instance_id,
                        **{k: v for k, v in ev.items() if k != "event"},
                    )
                if c.failure_due is not None and ts >= c.failure_due and run_h > 0 and cid not in self.failed:
                    self.unplanned_failure(ts, cid, [comp_id], "bearing", c.component_class)
        for pid, p in self.points.items():
            c = p.component
            ep = self.lube[p.asset_id]
            row = {
                "point_id": pid,
                "hour_utc": ts,
                "instance_id": c.instance_id,
                "stage": c.stage,
                "hi": c.h if c.onset_ts is not None else 0.0,
                "defect": c.defect,
                "onset_cause": c.onset_cause,
                "forced": c.forced,
                "lube_severity": ep.severity(ts) if (ep and p.point_class == "compressor_airend") else 0.0,
                "cavitation": False,
                "wear_frac": 0.0,
                "down": ts < self.down_until[p.asset_id],
                "scheduled": self.point_scheduled(pid, i),
                "measured": None,
            }
            self.out.truth_hours.append(row)
            self.truth_index[(pid, i)] = row


def battery_gaps(
    cfg: dict[str, Any], points: dict[str, PointRuntime], shared: Shared, ctx: RunContext
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    sb = cfg["sensor_behavior"]
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for pid, p in points.items():
        rng = rng_for(ctx.run_seed, "battery", pid)
        gaps = []
        t = p.install
        while True:
            life = pd.Timedelta(days=float(rng.uniform(*sb["battery_life_months"])) * DAYS_PER_YEAR / 12)
            gap = pd.Timedelta(days=float(rng.uniform(*sb["battery_gap_days"])))
            t0 = t + life
            if t0 >= shared.window_end:
                break
            gaps.append((t0, t0 + gap))
            t = t0 + gap
        out[pid] = gaps
    return out


def gateway_outages(
    cfg: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, ctx: RunContext
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    d = cfg["sensor_behavior"]["dropouts"]
    years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
    out = {}
    for gw in cfg["sensor_behavior"]["gateways"]:
        rng = rng_for(ctx.run_seed, "gateway", gw)
        k = rng.poisson(d["rate_per_gateway_year"] * years)
        starts = sorted(
            start + pd.Timedelta(seconds=float(s)) for s in rng.uniform(0, (end - start).total_seconds(), k)
        )
        out[gw] = [(s, s + pd.Timedelta(hours=float(rng.uniform(*d["duration_h"])))) for s in starts]
    return out

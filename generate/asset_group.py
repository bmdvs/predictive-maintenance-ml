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

from generate import conditions as cnd
from generate import maintenance, pump_wear
from generate.conditions import Condition
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
        self.sig = dict(cfg["bearing_signature"], fault_render=cfg["fault_conditions"]["render"])
        self.shift_start_h = int(_hm(cfg["shop"]["calendar"]["shifts_local"]["first"]["start"]))
        self.crng = rng_for(ctx.run_seed, "conditions", group_id)
        self.conditions: list[Condition] = []
        self.wo_index: dict[str, dict[str, Any]] = {}
        self.ingress_hazard: dict[str, tuple[int, float]] = {}
        self.pm_row: int | None = None

    # ---------------- conditions ----------------
    def start_condition(self, c: Condition) -> Condition:
        self.conditions.append(c)
        self.log(
            c.start,
            "condition_start",
            kind=c.kind,
            asset_id=c.asset_id,
            point_id=",".join(c.points) or None,
            cause=c.cause,
            ramp_h=c.ramp_h,
            **{f"p_{k}": v for k, v in c.params.items() if not isinstance(v, pd.Timestamp)},
        )
        return c

    def active_condition(self, asset_id: str, kind: str, ts: pd.Timestamp) -> Condition | None:
        for c in self.conditions:
            if c.asset_id == asset_id and c.kind == kind and c.active(ts):
                return c
        return None

    def clear_condition(
        self, c: Condition, ts: pd.Timestamp, by: str, wo_id: str | None = None, recovery_h: float = 0.0
    ) -> None:
        if c.cleared_at is not None:
            return
        c.clear(ts, by, recovery_h)
        self.log(
            ts,
            "condition_cleared",
            kind=c.kind,
            asset_id=c.asset_id,
            point_id=",".join(c.points) or None,
            by=by,
            wo_id=wo_id,
            level=c.level_at_clear,
            cause=c.cause,
        )

    def prune_conditions(self, ts: pd.Timestamp) -> None:
        self.conditions = [c for c in self.conditions if not c.finished(ts)]

    def point_faults(self, pid: str, ts: pd.Timestamp) -> list[dict[str, Any]]:
        return [
            cnd.fault_terms(c, ts)
            for c in self.conditions
            if c.kind in cnd.GRADUAL_FAULTS and pid in c.points and c.level(ts) > 0
        ]

    def point_condition_truth(self, pid: str, ts: pd.Timestamp) -> dict[str, Any]:
        kinds, level, ingress = [], 0.0, 0.0
        for c in self.conditions:
            if pid not in c.points:
                continue
            lv = c.level(ts)
            if lv <= 0:
                continue
            if c.kind in cnd.GRADUAL_FAULTS:
                kinds.append(c.kind)
                level = max(level, lv)
            elif c.kind == "coolant_ingress":
                ingress = max(ingress, lv)
        return {
            "fault_kind": "+".join(sorted(set(kinds))) or None,
            "fault_level": level,
            "ingress_level": ingress,
        }

    def condition_hazard(self, comp: BearingInstance, ts: pd.Timestamp) -> float:
        """Bearing hazard multiplier from untreated imbalance or gradual misalignment and from ingress."""
        fc = self.cfg["fault_conditions"]
        m = 1.0
        for c in self.conditions:
            if (
                c.kind in cnd.HAZARD_FAULTS
                and c.cause == "gradual"
                and comp.point_id in c.points
                and c.active(ts)
                and ts - c.start > pd.Timedelta(days=fc[c.kind]["untreated_days_for_hazard"])
            ):
                m *= c.params["hazard"]
        ing = self.active_condition(comp.asset_id, "coolant_ingress", ts)
        if ing is not None and comp.point_id in ing.points:
            self.ingress_hazard[comp.component_id] = (comp.instance_no, ing.params["hazard"])
        rec = self.ingress_hazard.get(comp.component_id)
        if rec is not None and rec[0] == comp.instance_no:
            m *= rec[1]
        return m

    def step_fault_onsets(self, ts: pd.Timestamp, units: list[tuple[str, list[str], str]]) -> None:
        """Imbalance, misalignment and looseness onsets per unit (asset id, its points, unit class)."""
        fc = self.cfg["fault_conditions"]
        for asset_id, pts, ucls in units:
            for kind in cnd.GRADUAL_FAULTS:
                spec = fc[kind]
                if ucls not in spec["unit_classes"] or self.active_condition(asset_id, kind, ts):
                    continue
                if self.crng.random() >= pump_wear.hourly_probability(
                    spec["onset_rate_per_unit_year"], 1.0, HOURS_PER_YEAR
                ):
                    continue
                points = pts if kind == "misalignment" else [pts[int(self.crng.integers(len(pts)))]]
                self.start_condition(cnd.draw_fault(self.cfg, kind, asset_id, points, ts, self.crng))

    def inspect_conditions(self, ts: pd.Timestamp, asset_id: str) -> list[str]:
        found = []
        for c in self.conditions:
            if c.asset_id != asset_id or not c.active(ts):
                continue
            if c.kind in cnd.GRADUAL_FAULTS:
                p = self.cfg["fault_conditions"][c.kind]["inspection_detection_prob"]
            elif c.kind == "rotary_union_leak":
                p = self.cfg["coolant_ingress"]["inspection_detection_prob"]
            else:
                continue
            tag = f"cond:{c.kind}"
            if tag not in found and self.crng.random() < p:
                found.append(tag)
        return found

    def condition_codes(self, asset_id: str, kinds: list[str], ts: pd.Timestamp) -> dict[str, Any]:
        """CMMS finding code and component for a corrective or condition-based condition repair."""
        if not kinds:
            return {}
        kind = kinds[0]
        if kind == "rotary_union_leak":
            if self.active_condition(asset_id, "coolant_ingress", ts):
                return {"code_override": "coolant_ingress", "component_override": "SPINDLE"}
            return {"code_override": "rotary_union_leak", "component_override": "ROTARY_UNION"}
        return {"code_override": kind}

    def corrective_duration(self, kinds: list[str]) -> pd.Timedelta:
        if "rotary_union_leak" in kinds:
            return pd.Timedelta(hours=float(self.crng.uniform(*self.cfg["coolant_ingress"]["repair_hours"])))
        return pd.Timedelta(
            hours=float(self.crng.uniform(*self.cfg["maintenance"]["repair_time"]["pump_hours"]))
        )

    def clear_for_repair(
        self, ts: pd.Timestamp, asset_id: str, replaced: bool, kinds: list[str], wo_id: str | None
    ) -> None:
        for c in list(self.conditions):
            if c.asset_id != asset_id or not c.active(ts):
                continue
            if c.kind in kinds:
                self.clear_condition(c, ts, "repair", wo_id)
            elif replaced and c.kind in cnd.GRADUAL_FAULTS:
                self.clear_condition(c, ts, "replacement", wo_id)
        if "rotary_union_leak" in kinds:
            ing = self.active_condition(asset_id, "coolant_ingress", ts)
            if ing is not None:
                self.clear_condition(ing, ts, "repair", wo_id)
            self.renew_rotary_union(ts)

    def renew_rotary_union(self, ts: pd.Timestamp) -> None:
        return None

    def post_repair_check(
        self, ts: pd.Timestamp, asset_id: str, comps: list[str], wo_id: str
    ) -> pd.Timestamp | None:
        """Time the post-repair check closes when it finds something to correct, else None."""
        return None

    def on_condition_repaired(
        self, ts: pd.Timestamp, asset_id: str, kinds: list[str], wo_id: str, by: str = "repair"
    ) -> None:
        self.clear_for_repair(ts, asset_id, False, kinds, wo_id)
        if self.open_wo.get(asset_id) == wo_id:
            self.open_wo.pop(asset_id)

    def on_corrective(self, ts: pd.Timestamp, asset_id: str, cond_kind: str, found: pd.Timestamp) -> None:
        """Planned corrective repair of a condition found outside an alert-driven inspection."""
        kind = cond_kind
        c = self.active_condition(asset_id, kind, ts)
        if c is None:
            return
        dur = self.corrective_duration([kind])
        wo = self.new_wo(
            asset_id,
            "corrective",
            found,
            started=ts,
            closed=ts + dur,
            findings=f"cond:{kind}",
            downtime_h=dur / pd.Timedelta(hours=1),
            **self.condition_codes(asset_id, [kind], ts),
        )
        self.log(ts, "corrective_repair", asset_id=asset_id, wo_id=wo["wo_id"], kind=kind)
        self.take_down(asset_id, ts, ts + dur)
        self.schedule(ts + dur, "condition_repaired", asset_id=asset_id, kinds=[kind], wo_id=wo["wo_id"])

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
        self.wo_index[wo["wo_id"]] = wo
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
        repair = [f for f in findings if f.startswith(("bearing:", "wear", "cond:"))]
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
        found += self.inspect_conditions(ts, asset_id)
        return found

    def inspect_extra(self, ts: pd.Timestamp, asset_id: str) -> list[str]:
        return []

    def on_planned_repair(
        self, ts: pd.Timestamp, asset_id: str, findings: list[str], inspection_wo: str
    ) -> None:
        if self.open_wo.get(asset_id) != inspection_wo:
            return
        hard = [f for f in findings if f.startswith(("bearing:", "wear"))]
        kinds = [
            f.split(":", 1)[1]
            for f in findings
            if f.startswith("cond:") and self.active_condition(asset_id, f.split(":", 1)[1], ts)
        ]
        if not hard and not kinds:
            self.open_wo.pop(asset_id, None)
            return
        comps = self.repair_scope(asset_id, hard) if hard else []
        if comps:
            dur = maintenance.repair_duration(self.cfg, self.components[comps[0]].component_class, self.rng)
            codes = {}
        else:
            dur = self.corrective_duration(kinds)
            codes = self.condition_codes(asset_id, kinds, ts)
        wo = self.new_wo(
            asset_id,
            "condition_based",
            ts,
            started=ts,
            closed=ts + dur,
            findings=",".join(hard + [f"cond:{k}" for k in kinds]),
            downtime_h=dur / pd.Timedelta(hours=1),
            source_wo=inspection_wo,
            **codes,
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
            kinds=kinds,
        )

    def repair_scope(self, asset_id: str, findings: list[str]) -> list[str]:
        return [f.split(":", 1)[1] for f in findings if f.startswith("bearing:")] or list(
            self.asset_components[asset_id]
        )

    def on_return_to_service(
        self,
        ts: pd.Timestamp,
        asset_id: str,
        comps: list[str],
        reason: str,
        wo_id: str,
        kinds: list[str] | None = None,
    ) -> None:
        if comps:
            self.replace(comps, ts, reason)
            self.after_replacement(asset_id, ts)
        self.failed.discard(asset_id)
        self.clear_for_repair(ts, asset_id, bool(comps), kinds or [], wo_id)
        check_close = self.post_repair_check(ts, asset_id, comps, wo_id) if comps else None
        if check_close is not None:
            self.open_wo[asset_id] = wo_id
            self.schedule(check_close, "post_repair_corrected", asset_id=asset_id, wo_id=wo_id, reason=reason)
            return
        if self.open_wo.get(asset_id) == wo_id:
            self.open_wo.pop(asset_id)
        if comps:
            self.rebaseline(asset_id, ts, reason)

    def on_post_repair_corrected(self, ts: pd.Timestamp, asset_id: str, wo_id: str, reason: str) -> None:
        c = self.active_condition(asset_id, "misalignment", ts)
        if c is not None:
            self.clear_condition(c, ts, "post_repair_check", wo_id)
        wo = self.wo_index.get(wo_id)
        if wo is not None:
            wo["post_repair_finding"] = "misalignment"
            wo["post_repair_corrected"] = ts
            wo["closed"] = max(wo.get("closed", ts), ts)
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
            self.schedule(
                row["completed"], "pm_done", asset_id=asset_id, task=task, row=len(self.out.pm_rows) - 1
            )
        self.on_pm_skipped(asset_id, task) if outcome == "skipped" else None
        nxt = due + pd.Timedelta(days=interval_days)
        self.schedule(nxt, "pm_due", asset_id=asset_id, task=task, interval_days=interval_days, due=nxt)

    def on_pm_skipped(self, asset_id: str, task: str) -> None:
        return None

    def on_pm_done(self, ts: pd.Timestamp, asset_id: str, task: str, row: int | None = None) -> None:
        self.log(ts, "pm_done", asset_id=asset_id, task=task)
        self.pm_row = row
        self.pm_effect(ts, asset_id, task)
        self.pm_row = None

    def record_pm_finding(self, finding: str) -> None:
        if self.pm_row is not None:
            self.out.pm_rows[self.pm_row]["finding"] = finding

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
            contamination_rise=cond.get("contamination_rise", 0.0),
            one_x_gain=cond.get("one_x_gain", 1.0),
            faults=cond.get("faults", []),
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
        if idx:
            self.prune_conditions(self.sh.frame["hour_utc"].iat[idx[-1]])

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
        self.fault_units = [
            (tsc_asset, [f"{machine_id}-TSC"], "tsc_pump"),
            (cp_asset, [f"{machine_id}-CP"], "flood_pump"),
        ]
        age = float(self.crng.uniform(*cfg["coolant_ingress"]["rotary_union_age_at_record_start_tsc_h"]))
        self.union_remaining_h = cnd.rotary_union_remaining_h(cfg, age, self.crng)
        self.union_found = False
        self.tool_factor: dict[str, float] = {}
        self.cooler_check_day: Any = None
        self.schedule_remounts()
        self.apply_overrides()

    def renew_rotary_union(self, ts: pd.Timestamp) -> None:
        self.union_remaining_h = cnd.rotary_union_remaining_h(self.cfg, 0.0, self.crng)
        self.union_found = False

    def program_one_x(self, program: str | None, ts: pd.Timestamp) -> float:
        if not program:
            return 1.0
        if program not in self.tool_factor:
            factor, high = cnd.tool_assembly_factor(
                self.cfg, rng_for(self.ctx.run_seed, "tool_assembly", program)
            )
            self.tool_factor[program] = factor
            if high:
                created = self.sched.loc[self.sched["program_id"] == program, "program_created"].iloc[0]
                self.log(
                    ts,
                    "tool_assembly_high_first_use",
                    asset_id=self.mid,
                    program=program,
                    factor=factor,
                    created_in_record=bool(pd.notna(created) and created >= self.sh.window_start),
                )
        return self.tool_factor[program]

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
            leak = self.active_condition(self.sp_asset, "rotary_union_leak", ts)
            if (
                leak is not None
                and not self.union_found
                and self.crng.random() < self.cfg["coolant_ingress"]["coolant_check_finds_leak_prob"]
            ):
                self.union_found = True
                self.record_pm_finding("rotary_union_leak")
                self.log(
                    ts,
                    "condition_found",
                    asset_id=self.sp_asset,
                    kind="rotary_union_leak",
                    by="coolant_check",
                )
                lead = pd.Timedelta(
                    days=float(self.crng.uniform(*self.cfg["maintenance"]["planned_repair_lead_days"]))
                )
                self.schedule(
                    self.planned_window(ts + lead, self.sp_asset),
                    "corrective",
                    asset_id=self.sp_asset,
                    cond_kind="rotary_union_leak",
                    found=ts,
                )
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
            contamination = max(
                (
                    c.level(ts) * c.params["hf_rise"]
                    for c in self.conditions
                    if c.kind == "coolant_ingress" and pid in c.points
                ),
                default=0.0,
            )
            return {
                "shaft_hz": op["speed"] / SECONDS_PER_MINUTE,
                "rated_hz": rated_hz,
                "load_pct": op["load"],
                "load_pct_report": op["load"],
                "flutes": op["flutes"],
                "lube_rise": sev * (ep.hf_rise if ep else 0.0),
                "lube_temp": sev * (ep.temp_rise_c if ep else 0.0),
                "contamination_rise": contamination,
                "one_x_gain": self.program_one_x(self.sched["program_id"].iat[i], ts),
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
            "faults": self.point_faults(pid, ts),
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
        if op["cut"] > 0:
            self.program_one_x(self.sched["program_id"].iat[i], ts)
        self.step_fault_onsets(ts, self.fault_units)
        leak, cooling = self.step_spindle_conditions(i, ts, op, run)

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
            lr = (life_rate if c.component_class.startswith("spindle") else 1.0) * self.condition_hazard(
                c, ts
            )
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
                **self.point_condition_truth(pid, ts),
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
                "tsc_leak_drop": leak.params["drop"] if leak is not None and leak.active(ts) else 0.0,
                "cooler_residual_c": cooling.level(ts) * cooling.params["rise_c"]
                if cooling is not None
                else 0.0,
            },
        )

    def step_spindle_conditions(
        self, i: int, ts: pd.Timestamp, op: dict[str, Any], run: dict[str, float]
    ) -> tuple[Condition | None, Condition | None]:
        """Rotary union leak and coolant ingress; spindle cooling degradation and overheat stops."""
        cfg, sp = self.cfg, self.sp_asset
        leak = self.active_condition(sp, "rotary_union_leak", ts)
        if leak is None:
            self.union_remaining_h -= run["tsc"]
            if self.union_remaining_h <= 0:
                leak = self.start_condition(cnd.draw_leak(cfg, sp, ts, self.crng))
        elif (
            leak.params["ingress"]
            and ts >= leak.params["ingress_at"]
            and self.active_condition(sp, "coolant_ingress", ts) is None
        ):
            leak.params["ingress"] = False
            self.start_condition(cnd.draw_ingress(cfg, sp, f"{self.mid}-SP-F", ts, self.crng))
        sc = cfg["spindle_cooling"]
        cooling = self.active_condition(sp, "spindle_cooling", ts)
        if cooling is None and self.crng.random() < pump_wear.hourly_probability(
            sc["onset_rate_per_hmc_year"], 1.0, HOURS_PER_YEAR
        ):
            cooling = self.start_condition(cnd.draw_spindle_cooling(cfg, sp, ts, self.crng))
        day = self.sh.frame["local_date"].iat[i]
        if cooling is not None and op["cut"] > 0 and sp not in self.failed and self.cooler_check_day != day:
            self.cooler_check_day = day
            if self.crng.random() < sc["overheat_stop_prob_per_day_at_full_rise"] * cooling.level(ts):
                dur = pd.Timedelta(hours=float(self.crng.uniform(*sc["repair_hours"])))
                wo = self.machine_stoppage(ts, sp, "spindle_cooler", (0.0, 0.0), dur=dur)
                self.schedule(
                    ts + dur, "condition_repaired", asset_id=sp, kinds=["spindle_cooling"], wo_id=wo["wo_id"]
                )
        return leak, cooling


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
        self.fault_units = [(cid, [f"{cid}-MDE", f"{cid}-AE"], "compressor") for cid in self.cids]
        self.fouling_rise = {cid: np.zeros(len(shared.frame)) for cid in self.cids}
        self.ctrl_events: list[dict[str, Any]] = []
        self.temp_high = {cid: False for cid in self.cids}
        self.local_month = shared.frame["hour_utc"].dt.tz_convert(LOCAL_TZ).dt.month.to_numpy()
        oc = cfg["oil_cooler_fouling"]
        for cid in self.cids:
            first = shared.window_start + pd.Timedelta(
                days=float(self.crng.uniform(0, oc["annual_cleaning_pm_days"]))
            )
            self.schedule_calendar_pm(cid, "oil_cooler_cleaning", oc["annual_cleaning_pm_days"], first)
        self.schedule_remounts()
        self.apply_overrides()

    def post_repair_check(
        self, ts: pd.Timestamp, asset_id: str, comps: list[str], wo_id: str
    ) -> pd.Timestamp | None:
        fc = self.cfg["fault_conditions"]["misalignment"]
        self.log(ts, "reinstallation", asset_id=asset_id, wo_id=wo_id, components=",".join(comps))
        if self.crng.random() >= fc["post_reinstallation_prob"]:
            return None
        pts = [f"{asset_id}-MDE", f"{asset_id}-AE"]
        self.start_condition(
            cnd.draw_fault(
                self.cfg, "misalignment", asset_id, pts, ts, self.crng, cause="post_reinstallation"
            )
        )
        return ts + pd.Timedelta(days=float(self.crng.uniform(*fc["post_repair_correction_days"])))

    def on_fouling_cleaned(self, ts: pd.Timestamp, asset_id: str, wo_id: str, by: str) -> None:
        f = self.active_condition(asset_id, "cooler_fouling", ts)
        if f is not None:
            rec = float(self.crng.uniform(*self.cfg["oil_cooler_fouling"]["cleaning_recovery_days"]))
            self.clear_condition(f, ts, by, wo_id, rec * HOURS_PER_DAY)

    def step_fouling(self, i: int, ts: pd.Timestamp, running: dict[str, bool]) -> None:
        """Oil cooler fouling onset and progression, discharge temperature warnings and trips."""
        oc, comp = self.cfg["oil_cooler_fouling"], self.cfg["compressor"]
        dt = comp["discharge_temp_c"]
        amb_mean = self.cfg["benign"]["ambient_room_c"]["annual_mean"]
        lo_m, hi_m = oc["summer_months"]
        summer = lo_m <= self.local_month[i] <= hi_m
        for cid in self.cids:
            role = self.reg.compressors[cid]["role"]
            f = self.active_condition(cid, "cooler_fouling", ts)
            if f is None and self.crng.random() < pump_wear.hourly_probability(
                oc["onset_rate_per_unit_year"][role], 1.0, HOURS_PER_YEAR
            ):
                f = self.start_condition(cnd.draw_fouling(self.cfg, cid, ts, self.crng))
            if f is not None:
                f.advance(1.0, oc["summer_rate_multiplier"] if summer else 1.0)
            rise = sum(
                c.level(ts) * c.params["rise_c"]
                for c in self.conditions
                if c.kind == "cooler_fouling" and c.asset_id == cid
            )
            self.fouling_rise[cid][i] = rise
            if not running[cid]:
                self.temp_high[cid] = False
                continue
            load = float(self.air[f"{cid}_load_frac"].iat[i]) or float(self.air["demand_frac"].iat[i])
            temp = (
                dt["base"]
                + dt["per_c_ambient"] * (float(self.air["room_c"].iat[i]) - amb_mean)
                + dt["per_load_frac"] * load
                + rise
            )
            if temp >= oc["trip_discharge_temp_c"] and cid not in self.failed:
                self.high_temp_trip(ts, cid, temp)
                self.temp_high[cid] = False
                continue
            high = temp >= oc["warning_discharge_temp_c"]
            if high and not self.temp_high[cid]:
                self.ctrl_events.append(
                    {
                        "compressor_id": cid,
                        "ts_utc": ts,
                        "event_type": "high_temp_warning",
                        "discharge_temp_c": round(temp, 1),
                        "stopped_machine": False,
                    }
                )
                self.log(ts, "high_temp_warning", asset_id=cid, discharge_temp_c=temp)
            self.temp_high[cid] = high

    def high_temp_trip(self, ts: pd.Timestamp, cid: str, temp: float) -> None:
        dur = pd.Timedelta(
            hours=float(self.crng.uniform(*self.cfg["compressor"]["other_stoppage_repair_hours"]))
        )
        wo = self.new_wo(
            cid,
            "unplanned_repair",
            ts,
            started=ts,
            closed=ts + dur,
            failure_mode="high_temp_trip",
            downtime_h=dur / pd.Timedelta(hours=1),
            components="",
        )
        self.log(
            ts, "unplanned_failure", asset_id=cid, mode="high_temp_trip", wo_id=wo["wo_id"], components=""
        )
        self.ctrl_events.append(
            {
                "compressor_id": cid,
                "ts_utc": ts,
                "event_type": "high_temp_trip",
                "discharge_temp_c": round(temp, 1),
                "stopped_machine": True,
            }
        )
        self.open_wo[cid] = wo["wo_id"]
        self.failed.add(cid)
        self.take_down(cid, ts, ts + dur)
        self.schedule(ts + dur, "trip_cleared", asset_id=cid, wo_id=wo["wo_id"])
        self.schedule(ts + dur, "fouling_cleaned", asset_id=cid, wo_id=wo["wo_id"], by="trip_repair")

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
        elif task == "oil_cooler_cleaning":
            f = self.active_condition(asset_id, "cooler_fouling", ts)
            if f is not None:
                rise = f.ramp(ts) * f.params["rise_c"]
                if rise > self.cfg["oil_cooler_fouling"]["pm_records_fouling_above_c"]:
                    self.record_pm_finding("oil_cooler_fouled")
                self.on_fouling_cleaned(ts, asset_id, None, "pm")

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
        faults = self.point_faults(pid, ts)
        if p.point_class == "compressor_motor":
            return {
                "shaft_hz": motor_hz,
                "rated_hz": motor_hz,
                "load_pct_report": load * 100,
                "faults": faults,
            }
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
            "faults": faults,
        }

    def hour_step(self, i: int, ts: pd.Timestamp) -> None:
        if i == 0:
            self.ops: dict[int, dict[str, bool]] = {}
        r = self.running(i, ts)
        self.ops[i] = r
        self.step_fault_onsets(ts, self.fault_units)
        self.step_fouling(i, ts, r)
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
                ) * self.condition_hazard(c, ts)
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
                **self.point_condition_truth(pid, ts),
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

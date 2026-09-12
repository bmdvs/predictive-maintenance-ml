"""Machines without wireless sensors (VMCs, lathes, mill-turns, grinder): operating state,
collisions with brinelling of the spindle, other stoppages, axis wear and calendar PM.

The spindle carries a bearing instance that degrades only after brinelling; it has no sensor and
shows only through the controller stream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from generate.asset_group import AssetGroup
from generate.machine_behavior import MachineMixin
from pdm.constants import SECONDS_PER_MINUTE


class FleetMachineGroup(MachineMixin, AssetGroup):
    def __init__(
        self, ctx, cfg, reg, shared, machine_id: str, schedule_frame: pd.DataFrame, axis_events=None
    ):
        super().__init__(ctx, cfg, reg, shared, machine_id, None)
        self.init_machine(machine_id, schedule_frame, axis_events or [])
        self.sp_asset = f"{machine_id}-SP"
        self.spindle = self.add_component(
            f"{machine_id}-SP", "spindle_front", self.sp_asset, None, self.machine["age_class"], 0.0
        )
        self.spindle.remaining_h = np.inf
        self.battery_gaps: dict[str, list] = {}
        pm = cfg["maintenance"]["pm_program"]["vmc_lathe_mill_turn"]
        for task, spec in pm.items():
            self.schedule_calendar_pm(machine_id, task, spec["interval"])

    def take_down(self, asset_id: str, t0: pd.Timestamp, t1: pd.Timestamp) -> None:
        self.down_until = max(self.down_until, t1)

    def after_replacement(self, asset_id: str, ts: pd.Timestamp) -> None:
        if asset_id == self.sp_asset:
            self.spindle.remaining_h = np.inf

    def point_scheduled(self, pid: str, i: int) -> bool:
        return False

    def point_condition(self, pid: str, i: int, ts: pd.Timestamp) -> dict[str, Any] | None:
        return None

    def hour_step(self, i: int, ts: pd.Timestamp) -> None:
        op = self.operating(i, ts)
        self.machine_crash(ts, op, self.spindle, self.sp_asset)
        run_h = op["cut"] / SECONDS_PER_MINUTE
        for ev in self.spindle.step(ts, run_h):
            self.log(
                ts,
                ev["event"],
                component_id=self.spindle.component_id,
                instance_id=self.spindle.instance_id,
                **{k: v for k, v in ev.items() if k != "event"},
            )
        c = self.spindle
        if (
            c.failure_due is not None
            and ts >= c.failure_due
            and run_h > 0
            and self.sp_asset not in self.failed
        ):
            self.unplanned_failure(ts, self.sp_asset, [c.component_id], "bearing", "spindle_front")
        th = self.cfg["bearing_degradation"]
        s4lo, hif = th["stage_thresholds_hi"]["stage4"], th["failure"]["hi_threshold"]
        s4 = float(np.clip((c.signature_hi() - s4lo) / (hif - s4lo), 0, 1)) if c.defect else 0.0
        self.machine_hour(
            i,
            ts,
            op,
            {
                "spindle_s4": s4,
                "spindle_stage": c.stage,
                "lube_temp_c": 0.0,
                "tsc_wear": 0.0,
                "tsc_cavitation": False,
                "cp_cavitation": False,
                "tsc_leak_drop": 0.0,
                "cooler_residual_c": 0.0,
            },
        )

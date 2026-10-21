"""Shared inputs for every asset group: hour frame, holidays, worked Saturdays, plant schedule,
gateway outages and ambient temperatures for a time window.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from generate import plant_air, shop_schedule
from generate.asset_group import Shared, gateway_outages
from generate.provenance import RunContext, rng_for


def build_shared(
    cfg: dict[str, Any], reg: Any, ctx: RunContext, start: pd.Timestamp, end: pd.Timestamp, rendered: set[str]
) -> tuple[Shared, dict[str, set]]:
    frame = shop_schedule.hour_frame(start, end)
    d0, d1 = frame["local_date"].iloc[0], frame["local_date"].iloc[-1]
    holidays = shop_schedule.holiday_dates(cfg["shop"]["calendar"]["holidays"], range(d0.year, d1.year + 1))
    sats = shop_schedule.worked_saturdays(cfg, d0, d1, rng_for(ctx.run_seed, "saturdays"))
    all_sats = set().union(*sats.values())
    plant = shop_schedule.scheduled_minutes(frame, cfg, holidays, all_sats)["scheduled_min"].to_numpy()
    heat_add, heat_episodes = plant_air.heat_spells(cfg, frame, rng_for(ctx.run_seed, "room_heat"))
    shared = Shared(
        frame=frame,
        holidays=holidays,
        plant_minutes=plant,
        gateway_outages=gateway_outages(cfg, start, end, ctx),
        shop_ambient=plant_air.shop_ambient_c(frame, cfg),
        room_ambient=plant_air.room_ambient_c(frame, cfg) + heat_add,
        rendered_points=rendered,
        window_start=start,
        window_end=end,
        heat_episodes=heat_episodes,
    )
    return shared, sats

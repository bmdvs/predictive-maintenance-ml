"""Compressor duty and compressor-room conditions by hour.

Inputs: compressor configuration, the plant's scheduled hours and a time window. Outputs: per
compressor running flag and load fraction, lead unit, room ambient temperature, and shop ambient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from generate.shop_schedule import LOCAL_TZ
from pdm.constants import DAYS_PER_YEAR, HOURS_PER_DAY


def seasonal_c(doy: np.ndarray, mean: float, amplitude: float, coldest_doy: float) -> np.ndarray:
    return mean - amplitude * np.cos(2 * np.pi * (doy - coldest_doy) / DAYS_PER_YEAR)


def room_ambient_c(frame: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    a = cfg["benign"]["ambient_room_c"]
    base = seasonal_c(
        frame["doy"].to_numpy(float), a["annual_mean"], a["annual_amplitude"], a["coldest_day_of_year"]
    )
    daily = a["daily_amplitude"] * np.cos(
        2 * np.pi * (frame["local_hour"].to_numpy(float) - a["daily_peak_hour_local"]) / HOURS_PER_DAY
    )
    return base + daily


def heat_spells(
    cfg: dict[str, Any], frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Additive compressor-room heat spells in summer months and their episode records."""
    hs = cfg["benign"]["compressor_room_heat"]
    rate = cfg["benign"]["episodes_per_year"]["compressor_room_heat"]
    hours = frame["hour_utc"]
    years = (hours.iloc[-1] - hours.iloc[0]) / pd.Timedelta(days=DAYS_PER_YEAR)
    months = hours.dt.tz_convert(LOCAL_TZ).dt.month.to_numpy()
    lo, hi = hs["months"]
    candidates = np.nonzero((months >= lo) & (months <= hi))[0]
    add = np.zeros(len(frame))
    episodes = []
    if candidates.size == 0:
        return add, episodes
    for _ in range(rng.poisson(rate * years)):
        i0 = int(rng.choice(candidates))
        dur = int(rng.uniform(*hs["days"]) * HOURS_PER_DAY)
        add[i0 : i0 + dur] += float(rng.uniform(*hs["add_c"]))
        episodes.append(
            {
                "event": "compressor_room_heat",
                "asset_id": "CMP",
                "ts": hours.iloc[i0],
                "end": hours.iloc[min(i0 + dur, len(frame) - 1)],
            }
        )
    return add, episodes


def shop_ambient_c(frame: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    a = cfg["benign"]["shop_ambient_c"]
    coldest = cfg["benign"]["ambient_room_c"]["coldest_day_of_year"]
    return seasonal_c(frame["doy"].to_numpy(float), a["annual_mean"], a["annual_amplitude"], coldest)


def _second_start(cfg: dict[str, Any]) -> float:
    h, m = cfg["shop"]["calendar"]["shifts_local"]["second"]["start"].split(":")
    return int(h) + int(m) / 60


def compressor_hours(
    cfg: dict[str, Any],
    frame: pd.DataFrame,
    plant_minutes: np.ndarray,
    compressor_ids: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Running state per compressor per hour.

    Args:
        frame: hour frame from shop_schedule.hour_frame.
        plant_minutes: scheduled minutes of the plant (any machine) per hour.
    """
    c = cfg["compressor"]
    pattern = c["lead_pattern_weeks"]
    week_no = ((frame["hour_utc"] - frame["hour_utc"].iloc[0]) / pd.Timedelta(weeks=1)).astype(int).to_numpy()
    lead = np.array([pattern[w % len(pattern)] for w in week_no])
    demand_tab = c["demand_frac_of_one_unit"]
    shift = np.where(
        frame["weekday"].to_numpy() == 5,
        "saturday",
        np.where(frame["local_hour"].to_numpy() < _second_start(cfg), "first_shift", "second_shift"),
    )
    demand = np.where(plant_minutes > 0, [demand_tab[s] for s in shift], demand_tab["off_shift"])
    running_any = demand > 0
    trim = running_any & (rng.random(len(frame)) < c["lag_trim_share_of_lead_hours"])
    out = frame[["hour_utc"]].copy()
    out["lead"] = lead
    out["demand_frac"] = demand
    out["room_c"] = room_ambient_c(frame, cfg)
    for cid in compressor_ids:
        is_lead = lead == cid
        out[f"{cid}_running"] = running_any & (is_lead | trim)
        out[f"{cid}_load_frac"] = (
            np.where(is_lead, demand, np.where(trim, demand, 0.0)) * out[f"{cid}_running"]
        )
    return out

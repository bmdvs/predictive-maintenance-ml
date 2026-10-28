"""Shop calendar, jobs and programs: the ERP and controller view of each machine by hour.

Inputs: production and shop configuration, the register and a time window. Outputs: per-machine
hourly frames with scheduled minutes, shift, job, program, material, spindle speed and load,
coolant use and benign production events (heavy new programs, coolant drift, titanium job).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from pdm.constants import DAYS_PER_YEAR, HOURS_PER_DAY, SECONDS_PER_MINUTE

LOCAL_TZ = "America/Chicago"
_DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def holiday_dates(names: list[str], years: range) -> set[date]:
    out: set[date] = set()
    for y in years:
        thanks = _nth_weekday(y, 11, 3, 4)
        table = {
            "new_years_day": date(y, 1, 1),
            "memorial_day": _last_weekday(y, 5, 0),
            "independence_day": date(y, 7, 4),
            "labor_day": _nth_weekday(y, 9, 0, 1),
            "thanksgiving": thanks,
            "day_after_thanksgiving": thanks + timedelta(days=1),
            "christmas_eve": date(y, 12, 24),
            "christmas_day": date(y, 12, 25),
        }
        out |= {table[n] for n in names}
    return out


def _hm(text: str) -> float:
    h, m = text.split(":")
    return int(h) + int(m) / SECONDS_PER_MINUTE


@dataclass
class Program:
    program_id: str
    material: str
    speed_frac: float
    load_pct: float
    flutes: int
    tsc_frac: float
    flood_frac: float
    created: pd.Timestamp
    heavy: bool = False


def hour_frame(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """UTC hours covering [start, end) with local date, hour and weekday."""
    idx = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    local = idx.tz_convert(LOCAL_TZ)
    return pd.DataFrame(
        {
            "hour_utc": idx,
            "local_date": local.date,
            "local_hour": local.hour,
            "weekday": local.weekday,
            "doy": local.dayofyear,
        }
    )


def worked_saturdays(
    cfg: dict[str, Any], start: date, end: date, rng: np.random.Generator
) -> dict[str, set[date]]:
    """Saturdays worked: one cell-wide draw for the HMC cell, one draw per other machine class key."""
    p = cfg["shop"]["calendar"]["saturday_first_shift_prob"]
    sats = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    sats = [d for d in sats if d.weekday() == 5]
    return {key: {d for d in sats if rng.random() < prob} for key, prob in p.items()}


def scheduled_minutes(
    frame: pd.DataFrame, cfg: dict[str, Any], holidays: set[date], saturdays: set[date]
) -> pd.DataFrame:
    cal = cfg["shop"]["calendar"]
    first, second = cal["shifts_local"]["first"], cal["shifts_local"]["second"]
    s1, e1, e2 = _hm(first["start"]), _hm(first["end"]), _hm(second["end"])
    workdays = {_DAY_INDEX[d] for d in cal["workdays"]}
    h0 = frame["local_hour"].to_numpy(dtype=float)
    h1 = h0 + 1
    is_work = np.array(
        [
            (wd in workdays) and (d not in holidays)
            for wd, d in zip(frame["weekday"], frame["local_date"], strict=False)
        ]
    )
    is_sat = np.array([d in saturdays for d in frame["local_date"]])
    end_hour = np.where(is_work, e2, np.where(is_sat, e1, s1))
    minutes = np.clip(np.minimum(h1, end_hour) - np.maximum(h0, s1), 0, 1) * SECONDS_PER_MINUTE
    out = frame.copy()
    out["scheduled_min"] = np.where(is_work | is_sat, minutes, 0.0)
    out["shift"] = np.where(out["scheduled_min"] > 0, np.where(h0 < e1, "first", "second"), "")
    return out


class ProgramPool:
    def __init__(
        self, cfg: dict[str, Any], machine: dict[str, Any], rng: np.random.Generator, start: pd.Timestamp
    ):
        self.cfg = cfg
        self.machine = machine
        self.rng = rng
        self.cls = machine["class"]
        self.count = 0
        prod = cfg["production"]
        self.mix = prod["material_mix"][self.cls]
        size = prod["programs_active_per_machine"][self.cls]
        self.programs: list[Program] = [self.new_program(start - pd.Timedelta(days=365)) for _ in range(size)]

    def draw_material(self) -> str:
        mats = [m for m, w in self.mix.items() if w > 0]
        w = np.array([self.mix[m] for m in mats], dtype=float)
        return str(self.rng.choice(mats, p=w / w.sum()))

    def new_program(
        self, created: pd.Timestamp, material: str | None = None, load_gain: float = 1.0
    ) -> Program:
        prod = self.cfg["production"]
        material = material or self.draw_material()
        regime = prod["cutting_regime_by_material"][material]
        lo, hi = prod["flutes_by_material"][material]
        self.count += 1
        is_hmc = self.cls == "hmc"
        return Program(
            program_id=f"{self.machine['machine_id']}-P{self.count:04d}",
            material=material,
            speed_frac=float(self.rng.uniform(*regime["speed_frac"])),
            load_pct=float(
                max(regime["load_pct"][0] + regime["load_pct"][1] * self.rng.standard_normal(), 1.0)
            )
            * load_gain,
            flutes=int(self.rng.integers(lo, hi + 1)),
            tsc_frac=self.coolant_share(prod["tsc_use_share_of_cutting"]) if is_hmc else 0.0,
            flood_frac=self.coolant_share(prod["flood_use_share_of_cutting"]),
            created=created,
            heavy=load_gain > 1.0,
        )

    def coolant_share(self, mean: float) -> float:
        spread = self.cfg["production"]["coolant_share_spread"]
        return float(np.clip(mean + self.rng.uniform(-spread, spread), 0.0, 1.0))

    def pick(self, material: str) -> Program:
        same = [p for p in self.programs if p.material == material]
        if not same:
            p = self.new_program(self.programs[-1].created, material)
            self.programs.append(p)
            return p
        return same[int(self.rng.integers(len(same)))]

    def add(self, program: Program) -> None:
        self.programs.append(program)
        limit = self.cfg["production"]["programs_active_per_machine"][self.cls]
        if len(self.programs) > limit:
            self.programs.pop(0)


def machine_schedule(
    cfg: dict[str, Any],
    machine: dict[str, Any],
    frame: pd.DataFrame,
    holidays: set[date],
    saturdays: set[date],
    rng: np.random.Generator,
    heavy_program_times: list[pd.Timestamp],
    titanium_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    coolant_drift: list[tuple[pd.Timestamp, pd.Timestamp, float]],
) -> tuple[pd.DataFrame, dict[str, Program], list[dict[str, Any]]]:
    """Hourly schedule for one machine.

    Returns:
        (hourly frame, programs by id, benign events)
    """
    prod = cfg["production"]
    cls = machine["class"]
    util = prod["utilization_cutting_over_scheduled"][cls]
    setup_share = prod["setup_share_of_noncutting"]
    cut_frac = util / (1.0 - setup_share * (1.0 - util))
    sched = scheduled_minutes(frame, cfg, holidays, saturdays)
    start = sched["hour_utc"].iloc[0]
    pool = ProgramPool(cfg, machine, rng, start)
    programs = {p.program_id: p for p in pool.programs}
    benign: list[dict[str, Any]] = []

    hours_per_day = float(
        sched.groupby("local_date")["scheduled_min"].sum().replace(0, np.nan).mean() / SECONDS_PER_MINUTE
    )
    month_h = DAYS_PER_YEAR / 12 * HOURS_PER_DAY
    new_rate_h = prod["new_programs_per_machine_month"][cls] / month_h
    jl = prod["job_length_days_lognormal"]
    mean_job_h = jl["median"] * np.exp(jl["sigma"] ** 2 / 2) * hours_per_day
    setup_per_job_h = setup_share * (1.0 - util) * mean_job_h

    n = len(sched)
    cols = {k: np.empty(n, dtype=object) for k in ("job_id", "program_id", "material")}
    setup = np.zeros(n, dtype=bool)
    cutting = np.zeros(n)
    pending_new = 0
    pending_heavy = sorted(heavy_program_times)
    job_left = 0.0
    setup_left = 0.0
    job_no = 0
    current: Program | None = None
    last_ts = start
    for i, (ts, smin) in enumerate(zip(sched["hour_utc"], sched["scheduled_min"], strict=False)):
        pending_new += rng.poisson(new_rate_h * (ts - last_ts) / pd.Timedelta(hours=1)) if i else 0
        last_ts = ts
        if smin <= 0:
            continue
        if job_left <= 0:
            job_no += 1
            job_h = float(jl["median"] * np.exp(jl["sigma"] * rng.standard_normal())) * hours_per_day
            job_left = job_h
            setup_left = setup_per_job_h
            in_ti = titanium_window is not None and titanium_window[0] <= ts < titanium_window[1]
            material = "titanium" if in_ti else pool.draw_material()
            if pending_heavy and pending_heavy[0] <= ts:
                pending_heavy.pop(0)
                gain = float(rng.uniform(*cfg["benign"]["heavy_new_program_load_gain"]))
                current = pool.new_program(ts, material, gain)
                pool.add(current)
                benign.append(
                    {
                        "event": "heavy_new_program",
                        "asset_id": machine["machine_id"],
                        "ts": ts,
                        "program_id": current.program_id,
                        "load_gain": gain,
                    }
                )
            elif pending_new > 0 or in_ti:
                pending_new = max(pending_new - 1, 0)
                current = pool.new_program(ts, material)
                pool.add(current)
            else:
                current = pool.pick(material)
            programs[current.program_id] = current
        frac_h = smin / SECONDS_PER_MINUTE
        job_left -= frac_h
        cols["job_id"][i] = f"{machine['machine_id']}-J{job_no:05d}"
        cols["program_id"][i] = current.program_id
        cols["material"][i] = current.material
        if setup_left > 0:
            setup[i] = True
            setup_left -= frac_h
        else:
            cutting[i] = cut_frac * smin

    out = sched.assign(**cols, setup=setup, cutting_min=cutting)
    prog = out["program_id"].map(lambda p: programs.get(p) if p else None)
    rated = machine["rated_rpm"]
    out["speed_rpm"] = [p.speed_frac * rated if p else 0.0 for p in prog]
    drift_gain = np.ones(n)
    for d0, d1, g in coolant_drift:
        drift_gain[((out["hour_utc"] >= d0) & (out["hour_utc"] < d1)).to_numpy()] *= g
    out["coolant_drift_gain"] = drift_gain
    out["load_pct"] = [p.load_pct if p else 0.0 for p in prog] * drift_gain
    out["flutes"] = [p.flutes if p else 0 for p in prog]
    out["tsc_frac"] = [p.tsc_frac if p else 0.0 for p in prog]
    out["flood_frac"] = [p.flood_frac if p else 0.0 for p in prog]
    out["program_created"] = [p.created if p else pd.NaT for p in prog]
    out["heavy_program"] = [bool(p and p.heavy) for p in prog]
    return out, programs, benign


def cell_benign_draws(
    cfg: dict[str, Any],
    machine_ids: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    rng: np.random.Generator,
) -> tuple[
    dict[str, list[pd.Timestamp]], dict[str, list[tuple[pd.Timestamp, pd.Timestamp, float]]], tuple | None
]:
    """Heavy new programs, coolant drift episodes and the titanium job window for the HMC cell."""
    ben = cfg["benign"]
    years = (end - start) / pd.Timedelta(days=DAYS_PER_YEAR)
    span = (end - start).total_seconds()
    heavy: dict[str, list[pd.Timestamp]] = {m: [] for m in machine_ids}
    for _ in range(rng.poisson(ben["episodes_per_year"]["heavy_new_program_hmc"] * years)):
        m = machine_ids[int(rng.integers(len(machine_ids)))]
        heavy[m].append(start + pd.Timedelta(seconds=float(rng.uniform(0, span))))
    drift: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, float]]] = {m: [] for m in machine_ids}
    for _ in range(rng.poisson(ben["episodes_per_year"]["coolant_concentration_drift_hmc"] * years)):
        m = machine_ids[int(rng.integers(len(machine_ids)))]
        t0 = start + pd.Timedelta(seconds=float(rng.uniform(0, span)))
        dur = pd.Timedelta(days=float(rng.uniform(*ben["coolant_drift_duration_days"])))
        drift[m].append((t0, t0 + dur, float(rng.uniform(*ben["coolant_drift_spindle_load_gain"]))))
    ti = ben["titanium_job"]
    w0, w1 = (
        pd.Timestamp(ti["start_window"][0], tz=LOCAL_TZ),
        pd.Timestamp(ti["start_window"][1], tz=LOCAL_TZ),
    )
    t0 = w0 + pd.Timedelta(seconds=float(rng.uniform(0, (w1 - w0).total_seconds())))
    titanium = (t0.tz_convert("UTC"), (t0 + pd.Timedelta(weeks=ti["duration_weeks"])).tz_convert("UTC"))
    return heavy, drift, titanium

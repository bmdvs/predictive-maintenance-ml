"""Controller-history export: 1 Hz MTConnect stream per machine, per-minute aggregates for the
full record, 1 Hz rows for the shadow quarter, the controller event log and 1 kHz spindle-load
windows around detected events.

Inputs: the machine's hourly schedule, its per-hour machine state and its event timeline.
Outputs: bronze controller tables under <bronze>/controller/ and minute-level regime truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from generate.machine_behavior import AXES_BY_CLASS
from generate.provenance import RunContext, rng_for
from pdm.constants import DAYS_PER_YEAR, SECONDS_PER_HOUR, SECONDS_PER_MINUTE
from pdm.ingest.mtconnect_stream import HZ1_COLUMNS, aggregate_minutes

_ALL_AXES = ("x", "y", "z", "b")
_CUTTING_EVENTS = {"collision", "overload_below_crash", "tool_breakage"}


def _program_attrs(
    cfg: dict[str, Any], ctx: RunContext, program: str, material: str, axes: list[str]
) -> dict[str, Any]:
    cs = cfg["controller_signals"]
    r = rng_for(ctx.run_seed, "program_signals", program)
    base = cs["axis_load_pct_by_material"][material]
    axis_base = {
        a: base * cs["axis_gain"][a] * max(1 + cs["axis_load_program_sd_frac"] * r.standard_normal(), 0.0)
        for a in axes
    }
    fo = cs["feed_override"]
    if r.random() < fo["nominal_share"]:
        feed = 100.0
    else:
        feed = float(
            r.choice(np.arange(fo["range_pct"][0], fo["range_pct"][1] + fo["step_pct"], fo["step_pct"]))
        )
    return {"axis": axis_base, "feed": feed}


def _failure_event_type(cfg: dict[str, Any], ev: dict[str, Any]) -> str | None:
    fmap = cfg["controller_signals"]["failure_event_type"]
    mode = ev.get("mode")
    if mode == "crash":
        return None
    if mode == "bearing":
        return fmap["bearing_spindle"] if str(ev["asset_id"]).endswith("-SP") else fmap["bearing_pump"]
    return fmap.get(mode)


def _khz_window(
    cfg: dict[str, Any], etype: str, base_load: float, peak: float, idle: float, rng: np.random.Generator
) -> np.ndarray:
    ks = cfg["controller_signals"]["khz_shape"]
    rate, half = ks["rate_hz"], ks["window_s"]
    t = np.arange(-half * rate, half * rate) / rate
    noise = cfg["controller_signals"]["spindle_load_noise_frac"]
    x = base_load * (1 + noise * rng.standard_normal(t.size))
    if etype == "collision":
        x = np.where(t < 0, x, idle + (peak - idle) * np.exp(-t / ks["collision_decay_s"]))
    elif etype == "overload_below_crash":
        x = x + (peak - base_load) * np.exp(-0.5 * (t / ks["overload_width_s"]) ** 2)
    else:
        x = np.where(t < 0, x, idle * (1 + noise * rng.standard_normal(t.size)))
    return np.clip(x, 0, None).astype(np.float32)


def render_machine(
    cfg: dict[str, Any],
    reg: Any,
    ctx: RunContext,
    machine_id: str,
    sched: pd.DataFrame,
    machine_hours: pd.DataFrame,
    events: list[dict[str, Any]],
    shop_ambient: np.ndarray,
    shadow: tuple[pd.Timestamp, pd.Timestamp],
    bronze_root: Path,
    truth_dir: Path,
) -> dict[str, int]:
    cs = cfg["controller_signals"]
    rng = rng_for(ctx.run_seed, "controller", machine_id)
    machine = reg.machines[machine_id]
    is_hmc = machine["class"] == "hmc"
    axes = AXES_BY_CLASS[machine["class"]]
    rated = float(machine["rated_rpm"])
    limit = float(machine["crash_load_limit_pct"])
    tsc_drop = cfg["pump"]["tsc_pressure_drop_frac_at_wear_end"]
    stamp = ctx.stamp()
    root = bronze_root / "controller"
    alarm_rates = {}
    for k, v in cfg["controller_events"]["alarms_per_machine_year"].items():
        r = v["hmc" if is_hmc else "other"] if isinstance(v, dict) else v
        if r > 0:
            alarm_rates[k] = r
    sched_s_year = max(
        float(sched["scheduled_min"].sum())
        * SECONDS_PER_MINUTE
        / ((sched["hour_utc"].iloc[-1] - sched["hour_utc"].iloc[0]) / pd.Timedelta(days=DAYS_PER_YEAR)),
        1.0,
    )
    tool_rate_h = cfg["controller_events"]["tool_breakage_per_cutting_hour"]
    hour_pos = {ts: i for i, ts in enumerate(sched["hour_utc"])}

    engine_by_hour: dict[int, list[dict[str, Any]]] = {}
    seal_failures: list[pd.Timestamp] = []
    for ev in events:
        etype = None
        if ev["event"] == "collision":
            etype = "collision"
        elif ev["event"] == "unplanned_failure":
            etype = _failure_event_type(cfg, ev)
            if ev.get("mode") == "seal_failure" and str(ev["asset_id"]).endswith("TSC"):
                seal_failures.append(ev["ts"])
        if etype is None:
            continue
        i = hour_pos.get(ev["ts"].floor("h"))
        if i is not None:
            engine_by_hour.setdefault(i, []).append(
                {"type": etype, "source": "engine", "failure": ev["event"] == "unplanned_failure"}
            )
    seal_leads = [
        (t - pd.Timedelta(hours=float(rng.uniform(*cfg["pump"]["seal_failure"]["pressure_drop_lead_h"]))), t)
        for t in seal_failures
    ]

    tau_s = cs["spindle_temp"]["time_constant_min"] * SECONDS_PER_MINUTE
    alpha = 1 - np.exp(-1.0 / tau_s)
    month_aggs: dict[str, list[pd.DataFrame]] = {}
    month_hz: dict[str, list[pd.DataFrame]] = {}
    event_rows: list[dict[str, Any]] = []
    khz_index: list[dict[str, Any]] = []
    counts = {"hz1_rows": 0, "minute_rows": 0, "events": 0, "khz_windows": 0}

    sched = sched.reset_index(drop=True)
    for _, day in sched[sched["scheduled_min"] > 0].groupby("local_date", sort=True):
        parts: dict[str, list[np.ndarray]] = {
            k: []
            for k in (
                "ts",
                "exe",
                "prog",
                "speed",
                "sload",
                "temp_target",
                "feed",
                "hyd",
                "cool",
                "alarm",
                *[f"ax_{a}" for a in _ALL_AXES],
            )
        }
        for i in day.index:
            row = sched.loc[i]
            mh = machine_hours.loc[i]
            m_count = int(round(row["scheduled_min"]))
            n = m_count * int(SECONDS_PER_MINUTE)
            t0 = row["hour_utc"]
            ts = t0 + pd.to_timedelta(np.arange(n), unit="s")
            down = bool(mh["down"])
            active_min = 0 if down else min(m_count, int(round(mh["cut_min"])))
            start_min = int(rng.integers(0, m_count - active_min + 1))
            exe = np.full(n, "stopped" if down else "ready", dtype=object)
            active = np.zeros(n, dtype=bool)
            sec = int(SECONDS_PER_MINUTE)
            active[start_min * sec : (start_min + active_min) * sec] = True
            exe[active] = "active"
            prog = str(row["program_id"]) if row["program_id"] else ""
            attrs = (
                _program_attrs(cfg, ctx, prog, row["material"], axes)
                if prog
                else {"axis": {a: 0.0 for a in axes}, "feed": 100.0}
            )
            speed = np.where(
                active, row["speed_rpm"] * (1 + cs["speed_noise_frac"] * rng.standard_normal(n)), 0.0
            )
            s4 = float(mh.get("spindle_s4", 0.0) or 0.0)
            sload = np.where(
                active,
                row["load_pct"] * (1 + cs["spindle_load_noise_frac"] * rng.standard_normal(n))
                + s4 * cs["spindle_load_gain_at_failure_pct"],
                cs["idle_spindle_load_pct"],
            )
            ax = {}
            for a in _ALL_AXES:
                if a not in axes:
                    ax[a] = np.full(n, np.nan)
                    continue
                mult = float(mh.get(f"axis_mult_{a}", 1.0))
                ax[a] = np.where(
                    active,
                    attrs["axis"][a] * mult * (1 + cs["axis_load_noise_frac"] * rng.standard_normal(n)),
                    cs["idle_axis_load_pct"],
                )
            st = cs["spindle_temp"]
            target = (
                shop_ambient[i]
                + st["rise_c_at_rated"] * speed / rated
                + s4 * st["rise_c_at_failure"]
                + float(mh.get("lube_temp_c", 0.0) or 0.0)
                + float(mh.get("cooler_residual_c", 0.0) or 0.0)
            )
            feed = np.where(active, attrs["feed"], 100.0)
            hb, tb = cs["hydraulic_bar"], cs["tsc_bar"]
            hyd = hb["nominal"] + hb["noise"] * rng.standard_normal(n) if is_hmc else np.full(n, np.nan)
            if is_hmc:
                factor = (
                    (1 - tsc_drop * float(mh.get("tsc_wear", 0.0)))
                    * (1 - tb["cavitation_drop_frac"] * bool(mh.get("tsc_cavitation", False)))
                    * (1 - float(mh.get("tsc_leak_drop", 0.0) or 0.0))
                )
                tsc_on = active & (np.cumsum(active) <= float(row["tsc_frac"]) * active.sum())
                cool = np.where(tsc_on, tb["nominal"] * factor + tb["noise"] * rng.standard_normal(n), 0.0)
                for a0, a1 in seal_leads:
                    sel = (ts >= a0) & (ts < a1) & tsc_on
                    if sel.any():
                        prog_frac = ((ts[sel] - a0) / (a1 - a0)).to_numpy(dtype=float)
                        cool[sel] *= 1 - tb["seal_lead_drop_frac"] * prog_frac
            else:
                cool = np.full(n, np.nan)
            alarm = np.full(n, "", dtype=object)

            hour_events = list(engine_by_hour.get(i, []))
            for etype, rate in alarm_rates.items():
                for _ in range(rng.poisson(rate / sched_s_year * n)):
                    hour_events.append(
                        {"type": "servo" if etype == "servo" else etype, "source": "alarm", "failure": False}
                    )
            for _ in range(rng.poisson(tool_rate_h * active.sum() / SECONDS_PER_HOUR)):
                hour_events.append({"type": "tool_breakage", "source": "alarm", "failure": False})
            act_idx = np.nonzero(active)[0]
            for ev in hour_events:
                etype = ev["type"]
                pool = act_idx if (etype in _CUTTING_EVENTS or act_idx.size) else np.arange(n)
                if pool.size == 0:
                    if etype in _CUTTING_EVENTS:
                        continue
                    pool = np.arange(n)
                k = int(rng.choice(pool))
                lo, hi = cs["event_duration_s"][etype]
                dur = int(rng.integers(lo, hi + 1))
                stops = bool(cs["event_stops_machine"][etype]) or ev["failure"]
                base_load = float(sload[k])
                if etype in cs["peak_frac_of_limit"]:
                    peak = limit * float(rng.uniform(*cs["peak_frac_of_limit"][etype]))
                else:
                    peak = base_load
                if etype != "operator_stop":
                    alarm[k] = etype
                end = n if ev["failure"] else min(n, k + dur)
                if stops:
                    exe[k:end] = "interrupted"
                    speed[k:end] = 0.0
                    sload[k:end] = cs["idle_spindle_load_pct"]
                    for a in axes:
                        ax[a][k:end] = cs["idle_axis_load_pct"]
                    feed[k:end] = 100.0
                    if is_hmc:
                        cool[k:end] = 0.0
                if is_hmc and etype == "hydraulic_low_pressure":
                    hyd[k:end] -= hb["fault_drop"] if ev["failure"] else hb["alarm_drop"]
                ets = ts[k]
                event_rows.append(
                    {
                        "machine_id": machine_id,
                        "ts_utc": ets,
                        "event_type": etype,
                        "peak_load_pct": round(peak, 1),
                        "duration_s": float((end - k) if stops else dur),
                        "program": prog,
                        "stopped_machine": stops,
                        "source": ev["source"],
                        "event_source": "execution_state" if etype == "operator_stop" else "controller_alarm",
                        **stamp,
                    }
                )
                if etype in cs["khz_event_types"] and active[k]:
                    w = _khz_window(cfg, etype, base_load, peak, cs["idle_spindle_load_pct"], rng)
                    rel = Path("khz") / machine_id / f"{ets:%Y%m%dT%H%M%S}_{etype}.npz"
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        root / rel,
                        spindle_load_pct=w,
                        rate_hz=cfg["controller_signals"]["khz_shape"]["rate_hz"],
                    )
                    khz_index.append(
                        {
                            "machine_id": machine_id,
                            "event_ts_utc": ets,
                            "event_type": etype,
                            "path": rel.as_posix(),
                            **stamp,
                        }
                    )
            for key, val in (
                ("ts", ts.to_numpy()),
                ("exe", exe),
                ("prog", np.full(n, prog, dtype=object)),
                ("speed", speed),
                ("sload", sload),
                ("temp_target", target),
                ("feed", feed),
                ("hyd", hyd),
                ("cool", cool),
                ("alarm", alarm),
            ):
                parts[key].append(val)
            for a in _ALL_AXES:
                parts[f"ax_{a}"].append(ax[a])

        cat = {k: np.concatenate(v) for k, v in parts.items()}
        target = cat["temp_target"]
        temp, _ = lfilter([alpha], [1, -(1 - alpha)], target, zi=[target[0] * (1 - alpha)])
        temp = temp + cs["spindle_temp"]["noise_c"] * rng.standard_normal(temp.size)
        hz = pd.DataFrame(
            {
                "machine_id": machine_id,
                "ts_utc": pd.DatetimeIndex(cat["ts"]).tz_localize("UTC")
                if pd.DatetimeIndex(cat["ts"]).tz is None
                else pd.DatetimeIndex(cat["ts"]),
                "execution": cat["exe"],
                "program": cat["prog"],
                "spindle_speed_rpm": np.round(cat["speed"]),
                "spindle_load_pct": np.round(np.clip(cat["sload"], 0, None), 1),
                **{f"{a}_load_pct": np.round(np.clip(cat[f"ax_{a}"], 0, None), 1) for a in _ALL_AXES},
                "spindle_temp_c": np.round(temp, 1),
                "feed_override_pct": cat["feed"],
                "hydraulic_pressure_bar": np.round(cat["hyd"], 1),
                "coolant_pressure_bar": np.round(np.clip(cat["cool"], 0, None), 1),
                "alarm_code": cat["alarm"],
            }
        )[HZ1_COLUMNS]
        agg = aggregate_minutes(hz)
        month = str(day["hour_utc"].iloc[0].strftime("%Y-%m"))
        month_aggs.setdefault(month, []).append(agg)
        if shadow[0] <= hz["ts_utc"].iloc[0] < shadow[1]:
            month_hz.setdefault(month, []).append(hz)

    regime_parts = []
    mat = sched.set_index("program_id")["material"].dropna()
    mat = mat[~mat.index.duplicated()]
    for month, frames in month_aggs.items():
        df = pd.concat(frames, ignore_index=True).assign(**stamp)
        d = root / "minute_agg" / f"month={month}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"{machine_id}.parquet", index=False)
        counts["minute_rows"] += len(df)
        regime_parts.append(df[["machine_id", "minute_utc", "program", "execution", "seconds_in_state"]])
    for month, frames in month_hz.items():
        df = pd.concat(frames, ignore_index=True).assign(**stamp)
        d = root / "hz1" / f"month={month}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"{machine_id}.parquet", index=False)
        counts["hz1_rows"] += len(df)
    if regime_parts:
        reg_df = pd.concat(regime_parts, ignore_index=True)
        reg_df["material"] = reg_df["program"].map(mat)
        (truth_dir / "regime").mkdir(parents=True, exist_ok=True)
        reg_df.assign(**stamp).to_parquet(truth_dir / "regime" / f"{machine_id}.parquet", index=False)
    if event_rows:
        ev = pd.DataFrame(event_rows)
        (root / "events").mkdir(parents=True, exist_ok=True)
        ev.drop(columns=["source"]).to_parquet(root / "events" / f"{machine_id}.parquet", index=False)
        (truth_dir / "controller_events").mkdir(parents=True, exist_ok=True)
        ev.to_parquet(truth_dir / "controller_events" / f"{machine_id}.parquet", index=False)
        counts["events"] = len(ev)
    if khz_index:
        (root / "khz_index").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(khz_index).to_parquet(root / "khz_index" / f"{machine_id}.parquet", index=False)
        counts["khz_windows"] = len(khz_index)
    return counts

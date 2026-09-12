"""Realism checks on an exported run, compared against the truth archive.

Inputs: bronze cm_platform tables, the truth archive and configuration. Output: a list of check
results (number, name, expected, measured, pass) plus supporting tables for the run report.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdm.config import Register
from pdm.features.kinematics import FAULT_KEYS, bearing_fault_orders
from pdm.paths import REPO_ROOT
from pdm.rules.iso_zones import zone_boundaries

_FAULT_FOR_DEFECT = {"outer_race": "bpfo", "inner_race": "bpfi", "ball": "bsf"}
_ISO_KEY = {
    "spindle_front": "spindle",
    "spindle_rear": "spindle",
    "tsc_pump": "tsc_pump",
    "flood_pump": "flood_pump",
    "compressor_motor": "compressor",
    "compressor_airend": "compressor",
}


@dataclass
class CheckResult:
    number: str
    name: str
    expected: str
    measured: str
    passed: bool | None
    detail: pd.DataFrame | None = field(default=None, repr=False)


def load_scalars(bronze_root: Path) -> pd.DataFrame:
    return pd.read_parquet(bronze_root / "cm_platform" / "scalars")


def hand_orders(n: int, d: float, D: float, alpha_deg: float) -> dict[str, float]:
    c = d / D * math.cos(alpha_deg * math.pi / 180.0)
    return {
        "bpfo": 0.5 * n * (1 - c),
        "bpfi": 0.5 * n * (1 + c),
        "bsf": 0.5 * D / d * (1 - c * c),
        "ftf": 0.5 * (1 - c),
    }


def check_kinematics(cfg: dict[str, Any], reg: Register) -> CheckResult:
    tol = cfg["checks"]["expected_ranges"]["kinematics_tolerance_frac"]
    rows = []
    for part, g in reg.bearings.items():
        code = bearing_fault_orders(g).as_dict()
        hand = hand_orders(g.n, g.d_mm, g.pitch_mm, g.contact_deg)
        for k in FAULT_KEYS:
            rows.append(
                {
                    "bearing": part,
                    "fault": k,
                    "code_order": code[k],
                    "hand_order": hand[k],
                    "rel_err": abs(code[k] - hand[k]) / hand[k],
                }
            )
    df = pd.DataFrame(rows)
    sanity = {"bpfo": 8.4, "bpfi": 10.6, "bsf": 4.2, "ftf": 0.44}
    front = df[df.bearing == "7014C-P4"].set_index("fault")["code_order"]
    sanity_err = max(abs(front[k] - v) / v for k, v in sanity.items())
    worst = df["rel_err"].max()
    return CheckResult(
        "1",
        "Kinematics vs hand computation",
        f"within {tol:.0%}",
        f"max error {worst:.2e}; HMC front set vs CLAUDE.md sanity values {sanity_err:.2%}",
        bool(worst <= tol and sanity_err <= tol),
        df,
    )


def _merge(scalars: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    meas = scalars[~scalars["no_measurement"]]
    return meas.merge(truth, on=["point_id", "hour_utc"], how="left", suffixes=("", "_truth"))


def no_conditions(m: pd.DataFrame) -> pd.Series:
    """Rows with no fault condition or coolant ingress on the point (all rows for older archives)."""
    ok = pd.Series(True, index=m.index)
    if "fault_kind" in m:
        ok &= m["fault_kind"].isna()
    if "ingress_level" in m:
        ok &= m["ingress_level"].fillna(0) == 0
    return ok


def healthy_rows(m: pd.DataFrame) -> pd.Series:
    return (
        (m["stage"] == 0)
        & (m["lube_severity"] == 0)
        & (~m["cavitation"].astype(bool))
        & (m["wear_frac"] == 0)
        & (m["radio_noise"].isna())
        & no_conditions(m)
    )


def check_healthy(cfg: dict[str, Any], reg: Register, m: pd.DataFrame) -> CheckResult:
    er = cfg["checks"]["expected_ranges"]
    ratio = cfg["checks"]["envelope_detection_ratio"]
    exceed_max = cfg["checks"]["healthy_exceed_share_max"]
    keys = cfg["checks"]["healthy_env_fault_keys"]
    healthy = m[healthy_rows(m)]
    rows = []
    for pid, g in healthy.groupby("point_id"):
        pclass = reg.point_class(pid)
        iso = cfg["rules"]["iso_class_by_point_class"][_ISO_KEY[pclass]]
        _, bc, _ = zone_boundaries(iso["group"], iso["foundation"])
        d = g[g["axis"] != "res"]
        r = g[g["axis"] == "res"]
        vmax = d.groupby("hour_utc")["vel_rms_mm_s"].max()
        env_any = (r[[f"res_envr_{k}" for k in keys]] > ratio).any(axis=1)
        ftf_share = (r["res_envr_ftf"] > ratio).mean()
        rows.append(
            {
                "point_id": pid,
                "hours": len(r),
                "vel_p95_mm_s": vmax.quantile(0.95),
                "zone_bc_mm_s": bc,
                "kurt_p05": d["kurtosis"].quantile(0.05),
                "kurt_p95": d["kurtosis"].quantile(0.95),
                "crest_p05": d["crest_factor"].quantile(0.05),
                "crest_p95": d["crest_factor"].quantile(0.95),
                "env_fault_peak_share": env_any.mean(),
                "ftf_peak_share_reported": ftf_share,
            }
        )
    df = pd.DataFrame(rows)
    k_lo, k_hi = er["healthy_kurtosis"]
    c_lo, c_hi = er["healthy_crest_factor"]
    df["vel_ok"] = df["vel_p95_mm_s"] < df["zone_bc_mm_s"]
    df["kurt_ok"] = (df["kurt_p05"] >= k_lo) & (df["kurt_p95"] <= k_hi)
    df["crest_ok"] = (df["crest_p05"] >= c_lo) & (df["crest_p95"] <= c_hi)
    df["env_ok"] = df["env_fault_peak_share"] <= exceed_max
    ok = bool(df[["vel_ok", "kurt_ok", "crest_ok", "env_ok"]].all().all())
    measured = (
        f"vel p95 max {df.vel_p95_mm_s.max():.2f} mm/s; kurtosis p05-p95 {df.kurt_p05.min():.2f}-{df.kurt_p95.max():.2f}; "
        f"crest p05-p95 {df.crest_p05.min():.2f}-{df.crest_p95.max():.2f}; env fault-peak share max {df.env_fault_peak_share.max():.3f}"
    )
    expected = f"zone A/B; kurtosis {k_lo}-{k_hi}; crest {c_lo}-{c_hi}; env peaks > {ratio:g}x floor in <= {exceed_max:.0%} of hours (p05-p95 rule)"
    return CheckResult("2", "Healthy signature", expected, measured, ok, df)


def _fault_hz(reg: Register, pid: str, speed_rpm: pd.Series, defect: str) -> pd.Series:
    orders = bearing_fault_orders(reg.bearings[reg.sensors[pid].bearing]).as_dict()
    return speed_rpm / 60.0 * orders[_FAULT_FOR_DEFECT[defect]]


def check_defect(cfg: dict[str, Any], reg: Register, m: pd.DataFrame) -> CheckResult:
    c = cfg["checks"]
    er = c["expected_ranges"]
    ratio = c["envelope_detection_ratio"]
    line_min = c["sideband_line_min_ratio"]
    tested_kinds = set(c["sideband_tested_kinds"])
    hw = cfg["indicators"]["envelope"]["band_half_width_frac"]
    harmonics = cfg["indicators"]["envelope"]["harmonics"]
    excl = c["inner_race_exclude_shaft_orders"]
    top_n = cfg["indicators"]["top_peaks"]
    need = er["stage2_envelope_peak_share_min"]
    s2 = m[(m["stage"] == 2) & (m["axis"] == "res") & (m["radio_noise"].isna())]
    rows = []
    for (pid, defect), g in s2.groupby(["point_id", "defect"]):
        f0 = _fault_hz(reg, pid, g["speed_rpm_at_measurement"], defect).to_numpy()
        fr = (g["speed_rpm_at_measurement"] / 60.0).to_numpy()
        pk_f = g[[f"envpk_{i + 1:02d}_hz" for i in range(top_n)]].to_numpy(dtype=float)
        if defect == "inner_race":
            is_shaft = np.zeros(pk_f.shape, dtype=bool)
            for k in excl:
                is_shaft |= np.abs(pk_f - k * fr[:, None]) <= hw * k * fr[:, None]
            pk_f = np.where(is_shaft, np.nan, pk_f)
            first = np.array([r[~np.isnan(r)][0] if (~np.isnan(r)).any() else np.nan for r in pk_f])
        else:
            first = pk_f[:, 0]
        hit = np.zeros(len(g), dtype=bool)
        for h in range(1, harmonics + 1):
            hit |= np.abs(first - h * f0) <= hw * h * f0
        kind = reg.bearings[reg.sensors[pid].bearing].kind
        row = {
            "point_id": pid,
            "bearing_kind": kind,
            "defect": defect,
            "hours": len(g),
            "top_peak_at_fault_share": hit.mean(),
        }
        if defect == "inner_race":
            sb = ((g["res_envp_bpfi_sb_lo"] > ratio) | (g["res_envp_bpfi_sb_hi"] > ratio)).to_numpy()
            gate = (g["res_envp_bpfi"] >= line_min).to_numpy()
            row["sideband_tested"] = kind in tested_kinds
            row["sideband_share_all_hours"] = sb.mean()
            row["line_ge_12x_hours"] = int(gate.sum())
            row["sideband_share_line_ge_12x"] = sb[gate].mean() if gate.any() else np.nan
            if len(g) >= 3:
                thirds = pd.qcut(g["hi"].rank(method="first"), 3, labels=["first", "second", "third"])
                for lab in ("first", "second", "third"):
                    row[f"sideband_share_{lab}_third"] = sb[(thirds == lab).to_numpy()].mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return CheckResult("3", "Defect signature (stage 2)", f">= {need:.0%}", "no stage-2 hours", None, df)
    ok = bool((df["top_peak_at_fault_share"] >= need).all())
    sb_txt = "no inner-race stage-2 hours"
    if "sideband_tested" in df:
        inner = df[df["defect"] == "inner_race"]
        tested = inner[inner["sideband_tested"].astype(bool)]
        gated = tested.dropna(subset=["sideband_share_line_ge_12x"])
        if len(gated):
            ok = ok and bool((gated["sideband_share_line_ge_12x"] >= need).all())
        untested = inner[~inner["sideband_tested"].astype(bool)]
        sb_txt = (
            f"sidebands (peak in BPFI -/+ 1x band above {ratio:g}x local floor) tested on deep-groove and cylindrical "
            f"points on stage-2 hours with the BPFI line >= {line_min:g}x the floor, pass at >= {need:.0%}: "
            + (
                ", ".join(
                    f"{r.point_id} {r.sideband_share_line_ge_12x:.0%} of {r.line_ge_12x_hours} h"
                    for r in gated.itertuples()
                )
                or "no gated hours"
            )
            + "; angular-contact (report-only): "
            + (
                ", ".join(f"{r.point_id} {r.sideband_share_all_hours:.0%}" for r in untested.itertuples())
                or "none"
            )
        )
    by = df.groupby("defect")["top_peak_at_fault_share"].min().round(2).to_dict()
    measured = f"top peak at fault frequency or harmonic, min by defect {by} (inner race excludes shaft 1x-4x); {sb_txt}"
    return CheckResult(
        "3",
        "Defect signature (stage 2)",
        f">= {need:.0%} top peak; deep-groove and cylindrical sidebands >= {need:.0%} where line >= {line_min:g}x floor",
        measured,
        ok,
        df,
    )


def _first_consecutive(ts: pd.Series, cond: pd.Series, required: int = 2) -> pd.Timestamp | None:
    run = 0
    for t, c in zip(ts, cond, strict=False):
        run = run + 1 if c else 0
        if run >= required:
            return t
    return None


def four_stage_instances(events: pd.DataFrame) -> pd.DataFrame:
    events = events.reindex(
        columns=sorted(
            set(events.columns) | {"mode", "components", "component_id", "instance_id", "cause", "defect"}
        )
    )
    fails = events[(events["event"] == "unplanned_failure") & (events["mode"] == "bearing")]
    onsets = events[events["event"] == "bearing_onset"][
        ["component_id", "instance_id", "ts", "cause", "defect"]
    ]
    rows = []
    for _, f in fails.iterrows():
        for comp in str(f["components"]).split(","):
            inst = onsets[onsets["component_id"] == comp]
            inst = inst[inst["ts"] <= f["ts"]].tail(1)
            if inst.empty:
                continue
            rows.append(
                {
                    "component_id": comp,
                    "instance_id": inst["instance_id"].iat[0],
                    "onset_ts": inst["ts"].iat[0],
                    "failure_ts": f["ts"],
                    "cause": inst["cause"].iat[0],
                    "defect": inst["defect"].iat[0],
                }
            )
    return pd.DataFrame(rows)


def _converting_lube_start(events: pd.DataFrame, comp: str, onset: pd.Timestamp) -> pd.Timestamp | None:
    if "-SP-" in comp:
        asset = comp.rsplit("-", 1)[0]
    elif comp.endswith("-AE"):
        asset = comp[:-3]
    else:
        asset = comp
    lube = events[
        (events["event"] == "lube_episode_start") & (events["asset_id"] == asset) & (events["ts"] <= onset)
    ]
    return lube["ts"].max() if not lube.empty else None


def _concurrent_wear(
    m: pd.DataFrame, events: pd.DataFrame, comp: str, t0: pd.Timestamp, t1: pd.Timestamp, limit: float
) -> str | None:
    g = m[(m["point_id"] == comp) & m["hour_utc"].between(t0, t1)]
    if g.empty or g["wear_frac"].max() <= limit:
        return None
    w = events[(events["event"] == "pump_wear_start") & (events["asset_id"] == comp) & (events["ts"] <= t1)]
    kinds = sorted(set(w["kind"].dropna())) if "kind" in w else []
    return f"{'/'.join(kinds) or 'pump'} wear {g['wear_frac'].max():.2f}"


def _mechanism_hours(
    truth: pd.DataFrame | None, pid: str, t0: pd.Timestamp, t1: pd.Timestamp, kinds: list[str]
) -> dict[str, int]:
    """Hours in (t0, t1] with each competing mechanism active on the point, from truth."""
    if truth is None:
        return {}
    g = truth[(truth["point_id"] == pid) & (truth["hour_utc"] > t0) & (truth["hour_utc"] <= t1)]
    out: dict[str, int] = {}
    for k in kinds:
        if k == "cavitation":
            n = int(g["cavitation"].astype(bool).sum())
        elif k == "coolant_ingress":
            n = int((g["ingress_level"].fillna(0) > 0).sum()) if "ingress_level" in g else 0
        else:
            n = int(g["fault_kind"].fillna("").str.contains(k).sum()) if "fault_kind" in g else 0
        if n:
            out[k] = n
    return out


def check_four_stage(
    cfg: dict[str, Any],
    reg: Register,
    m: pd.DataFrame,
    events: pd.DataFrame,
    sensor_events: pd.DataFrame,
    truth: pd.DataFrame | None = None,
) -> CheckResult:
    c = cfg["checks"]
    after_peak_kinds = list(c.get("four_stage_exclude_after_kurtosis_peak", []))
    ratio = c["envelope_detection_ratio"]
    nsig = c["hf_detection_sigma"]
    fall_min = c["stage34_fall_frac_min"]
    tie = pd.Timedelta(hours=c["four_stage_tie_h"])
    wear_limit = c["four_stage_exclude_pump_wear_above"]
    win = f"{c['rolling_window_h']}h"
    inst = four_stage_instances(events)
    rows = []
    for _, it in inst.iterrows():
        pid = it["component_id"]
        if pid not in set(m["point_id"]):
            continue
        g = m[(m["point_id"] == pid) & (m["radio_noise"].isna())]
        res = g[g["axis"] == "res"].set_index("hour_utc").sort_index()
        dirx = g[g["axis"] != "res"].groupby("hour_utc")[["vel_rms_mm_s", "kurtosis"]].max().sort_index()
        se = sensor_events[(sensor_events["point_id"] == pid) & (sensor_events["ts"] <= it["onset_ts"])]
        b0 = se[se["event"].isin(["install", "baselining_start", "remount"])]["ts"].max()
        b1 = b0 + pd.Timedelta(days=cfg["measurement"]["baselining_days"])
        base = res.loc[(res.index >= b0) & (res.index < b1)]
        hf_thr = base["hf_rms_g"].mean() + nsig * base["hf_rms_g"].std()
        life = res.loc[(res.index >= it["onset_ts"]) & (res.index < it["failure_ts"])]
        dlife = dirx.loc[(dirx.index >= it["onset_ts"]) & (dirx.index < it["failure_ts"])]
        pclass = reg.point_class(pid)
        iso = cfg["rules"]["iso_class_by_point_class"][_ISO_KEY[pclass]]
        _, bc, _ = zone_boundaries(iso["group"], iso["foundation"])
        lube_start = (
            _converting_lube_start(events, pid, it["onset_ts"]) if it["cause"] == "lubrication" else None
        )
        if lube_start is not None:
            t_hf = lube_start
        else:
            t_hf = _first_consecutive(life.index.to_series(), life["hf_rms_g"] > hf_thr)
        env_any = (life[[f"res_envr_{k}" for k in c["healthy_env_fault_keys"]]] > ratio).any(axis=1)
        t_env = _first_consecutive(life.index.to_series(), env_any)
        crest_rm = life["crest_factor"].rolling(win, min_periods=c["rolling_min_readings"]).median()
        kurt_rm = dlife["kurtosis"].rolling(win, min_periods=c["rolling_min_readings"]).median()
        t_crest = crest_rm.idxmax() if crest_rm.notna().any() else None
        t_kurt = kurt_rm.idxmax() if kurt_rm.notna().any() else None
        t_zone = _first_consecutive(dlife.index.to_series(), dlife["vel_rms_mm_s"] >= bc)
        seq = [t_hf, t_env, t_crest, t_zone, it["failure_ts"]]
        ordered = all(s is not None for s in seq) and all(
            a <= b + tie for a, b in zip(seq, seq[1:], strict=False)
        )
        peaks_before_zone = t_zone is not None and all(
            tp is not None and tp <= t_zone + tie for tp in (t_crest, t_kurt)
        )
        stage_at = g[g["axis"] == "res"].set_index("hour_utc")["stage"]
        crest_base = base["crest_factor"].median()
        kurt_base = g[(g["axis"] != "res") & g["hour_utc"].between(b0, b1)]["kurtosis"].median()

        def fall(series: pd.Series, baseline: float) -> tuple[float, float]:
            if series.dropna().empty:
                return np.nan, np.nan
            peak, last = series.max(), series.dropna().iloc[-1]
            return (peak - last) / (peak - baseline) if peak > baseline else np.nan, (peak - last) / peak

        cf_ex, cf_pk = fall(crest_rm, crest_base)
        kf_ex, kf_pk = fall(kurt_rm, kurt_base)
        competing = None
        if pclass.endswith("pump"):
            competing = _concurrent_wear(m, events, pid, it["onset_ts"], it["failure_ts"], wear_limit)
        life_cav = _mechanism_hours(truth, pid, it["onset_ts"], it["failure_ts"], ["cavitation"]).get(
            "cavitation", 0
        )
        after_peak = (
            _mechanism_hours(truth, pid, t_kurt, it["failure_ts"], after_peak_kinds)
            if t_kurt is not None
            else {}
        )
        if after_peak:
            text = ", ".join(f"{k} {n} h" for k, n in after_peak.items()) + " after kurtosis peak"
            competing = f"{competing}; {text}" if competing else text
        rows.append(
            {
                "component_id": pid,
                "instance_id": it["instance_id"],
                "cause": it["cause"],
                "defect": it["defect"],
                "onset": it["onset_ts"],
                "t_hf": t_hf,
                "t_env": t_env,
                "t_crest_peak": t_crest,
                "t_kurt_peak": t_kurt,
                "t_zone_c": t_zone,
                "failure": it["failure_ts"],
                "ordered": ordered,
                "peaks_before_zone_c": peaks_before_zone,
                "crest_peak_stage": stage_at.get(t_crest) if t_crest is not None else None,
                "kurt_peak_stage": stage_at.get(t_kurt) if t_kurt is not None else None,
                "crest_fall_of_excess": cf_ex,
                "crest_fall_of_peak": cf_pk,
                "kurt_fall_of_excess": kf_ex,
                "kurt_fall_of_peak": kf_pk,
                "cavitation_h_in_life": life_cav,
                "cavitation_h_after_kurt_peak": after_peak.get("cavitation", 0),
                "excluded_competing_mechanism": competing,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return CheckResult(
            "4",
            "Four-stage ordering",
            "ordered; peaks before zone C; fall >= 30%",
            "no failed instance on a rendered point",
            None,
            df,
        )
    df["fall_ok"] = (df["crest_fall_of_excess"] >= fall_min) & (df["kurt_fall_of_excess"] >= fall_min)
    tested = df[df["excluded_competing_mechanism"].isna()]
    excluded = df[df["excluded_competing_mechanism"].notna()]
    ok = (
        bool((tested["ordered"] & tested["peaks_before_zone_c"] & tested["fall_ok"]).all())
        if len(tested)
        else None
    )
    measured = (
        f"tested {len(tested)}: ordered {int(tested.ordered.sum())}/{len(tested)}; kurtosis and crest peaks before zone C "
        f"{int(tested.peaks_before_zone_c.sum())}/{len(tested)}; fall >= {fall_min:.0%} of rise "
        f"{int(tested.fall_ok.sum())}/{len(tested)}"
    )
    if len(excluded):
        measured += "; excluded " + "; ".join(
            f"{r.instance_id} ({r.excluded_competing_mechanism})" for r in excluded.itertuples()
        )
    return CheckResult(
        "4",
        "Four-stage ordering",
        f"HF <= envelope <= crest peak <= zone C <= failure (24 h ties); kurtosis and crest peak before zone C; "
        f"both fall >= {fall_min:.0%} of rise; excluded: pump wear above {wear_limit}, or "
        f"{', '.join(after_peak_kinds) or 'none'} on the point between the kurtosis peak and failure",
        measured,
        ok,
        df,
    )


def check_independence() -> CheckResult:
    bad = []
    for path in (REPO_ROOT / "src" / "pdm").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "generate" for a in node.names):
                bad.append(f"{path.name}: import generate")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "generate":
                bad.append(f"{path.name}: from generate")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "runs/" in node.value:
                bad.append(f"{path.name}: 'runs/' literal")
    return CheckResult(
        "16",
        "Independence from truth",
        "no imports of generate, no runs/ paths in src/pdm",
        "none found" if not bad else "; ".join(bad),
        not bad,
    )


def run_checks(
    cfg: dict[str, Any], reg: Register, ctx: Any, bronze_root: Path, truth_dir: Path, pilot: bool
) -> dict[str, Any]:
    scalars = load_scalars(bronze_root)
    truth = pd.read_parquet(truth_dir / "point_hours.parquet").drop(columns=["run_id", "config_hash"])
    events = pd.read_parquet(truth_dir / "events.parquet")
    sensor_events = pd.read_parquet(bronze_root / "cm_platform" / "sensor_events" / "sensor_events.parquet")
    m = _merge(scalars, truth)
    results = [
        check_kinematics(cfg, reg),
        check_healthy(cfg, reg, m),
        check_defect(cfg, reg, m),
        check_four_stage(cfg, reg, m, events, sensor_events, truth),
        check_independence(),
    ]
    extra: dict[str, Any] = {}
    if not pilot:
        from generate.checks import realism_full

        full, extra = realism_full.run_full_checks(
            cfg, reg, bronze_root, truth_dir, scalars, truth, events, sensor_events, m
        )
        results = results[:4] + full + results[4:]
    return {
        "results": results,
        **extra,
        "scalars": scalars,
        "truth": truth,
        "events": events,
        "merged": m,
        "sensor_events": sensor_events,
    }

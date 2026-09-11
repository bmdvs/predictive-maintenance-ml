"""Run report in the CLAUDE.md section 15 format: check table, config diff, headline counts,
timing, and pilot figures. Written to runs/<run_id>/run_report.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from generate.checks import pilot_figures
from generate.provenance import RUNS_DIR, RunContext
from pdm.config import load_register


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


def previous_run(ctx: RunContext) -> Path | None:
    prior = sorted(
        p
        for p in RUNS_DIR.glob(f"{ctx.label[:-2]}*")
        if p.name != ctx.run_id and (p / "resolved_config.json").exists()
    )
    return prior[-1] if prior else None


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def full_timing(
    timing: dict[str, Any], counts: dict[str, int], bronze_root: Path, truth_dir: Path
) -> pd.DataFrame:
    gb = 1024.0**3
    ms = timing["render_extract_s"] / max(timing["bursts_rendered"], 1) * 1000
    groups = timing.get("groups", {})
    slow = max(groups, key=groups.get) if groups else ""
    rows = [
        ("elapsed (min)", f"{timing['elapsed_s'] / 60:.1f}"),
        ("bursts rendered", f"{timing['bursts_rendered']:,}"),
        ("render plus extraction per burst (ms)", f"{ms:.1f}"),
        ("slowest group", f"{slow} {groups.get(slow, 0) / 60:.1f} min"),
        ("bronze size (GB)", f"{_dir_bytes(bronze_root) / gb:.2f}"),
        ("truth archive size (GB)", f"{_dir_bytes(truth_dir) / gb:.2f}"),
        *[(f"rows: {k}", f"{v:,}") for k, v in sorted(counts.items())],
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def config_diff(ctx: RunContext) -> str:
    prior = sorted(
        p
        for p in RUNS_DIR.glob(f"{ctx.label[:-2]}*")
        if p.name != ctx.run_id and (p / "resolved_config.json").exists()
    )
    if not prior:
        return "none (first run of this series)"
    with open(prior[-1] / "resolved_config.json", encoding="utf-8") as fh:
        a = _flatten(json.load(fh))
    with open(ctx.run_dir / "resolved_config.json", encoding="utf-8") as fh:
        b = _flatten(json.load(fh))
    lines = [
        f"{k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if str(a.get(k)) != str(b.get(k))
    ]
    return f"vs {prior[-1].name}\n" + ("\n".join(lines) if lines else "none")


def _md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return format(v, floatfmt) if pd.notna(v) else ""
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d %H:%M")
        return "" if v is None else str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))


def headline_counts(chk: dict[str, Any], counts: dict[str, int]) -> pd.DataFrame:
    ev = chk["events"]
    sc = chk["scalars"]
    alerts = (
        pd.read_parquet(Path(chk["bronze_root"]) / "cm_platform" / "alerts" / "alerts.parquet")
        if counts.get("alerts")
        else pd.DataFrame()
    )
    wo = pd.read_parquet(Path(chk["truth_dir"]) / "work_orders.parquet")
    wo = wo.reindex(columns=sorted(set(wo.columns) | {"type", "findings", "no_fault_found"}))
    insp = wo[wo["type"] == "inspection"]
    insp_done = insp[insp["findings"].fillna("") != "superseded"] if "findings" in insp else insp
    rows = [
        ("bearing onsets (all simulated components)", int((ev["event"] == "bearing_onset").sum())),
        ("unplanned failures", int((ev["event"] == "unplanned_failure").sum())),
        ("planned condition-based repairs", int((ev["event"] == "planned_repair").sum())),
        ("lubrication episodes", int((ev["event"] == "lube_episode_start").sum())),
        ("cavitation episodes", int((ev["event"] == "cavitation_start").sum())),
        ("collisions", int((ev["event"] == "collision").sum())),
        ("platform alerts (rendered points)", len(alerts)),
        ("alert-driven inspections", len(insp_done)),
        (
            "inspections closing no_fault_found",
            int(insp_done.get("no_fault_found", pd.Series(dtype=bool)).fillna(False).sum()),
        ),
        ("scalar rows", counts.get("scalars", 0)),
        ("retained bursts", counts.get("bursts", 0)),
    ]
    nm = sc.groupby("point_id")["no_measurement"].mean()
    rows += [(f"no_measurement share {p}", f"{v:.1%}") for p, v in nm.items()]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def timing_projection(cfg: dict[str, Any], chk: dict[str, Any], timing: dict[str, float]) -> pd.DataFrame:
    sc = chk["scalars"]
    meas = sc[(~sc["no_measurement"]) & (sc["axis"] == "res")]
    days = (sc["hour_utc"].max() - sc["hour_utc"].min()) / pd.Timedelta(days=1)
    per_day = meas.groupby("point_id").size() / days
    ms = timing["render_extract_s"] / max(timing["bursts_rendered"], 1) * 1000
    record_days = (
        pd.Timestamp(str(cfg["meta"]["record"]["end"])) - pd.Timestamp(str(cfg["meta"]["record"]["start"]))
    ).days + 1
    sp = per_day.filter(like="SP-").mean()
    tsc = per_day.filter(like="TSC").mean()
    flood = (
        tsc * cfg["production"]["flood_use_share_of_cutting"] / cfg["production"]["tsc_use_share_of_cutting"]
    )
    cmp_ = per_day.filter(like="CMP").mean()
    hmc_group = (2 * sp + tsc + flood) * record_days
    total = 4 * hmc_group + 4 * cmp_ * record_days
    overhead_s = timing["elapsed_s"] - timing["render_extract_s"]
    wall_min = (hmc_group * ms / 1000 + overhead_s * record_days / days) / 60
    rows = [
        ("measured render plus extraction per burst (ms)", f"{ms:.1f}"),
        ("pilot bursts rendered", f"{timing['bursts_rendered']:,}"),
        ("pilot elapsed (s)", f"{timing['elapsed_s']:.0f}"),
        ("projected bursts, full record", f"{total:,.0f}"),
        ("projected wall time, slowest of 5 parallel groups (min)", f"{wall_min:.0f}"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def write(
    ctx: RunContext,
    cfg: dict[str, Any],
    chk: dict[str, Any],
    counts: dict[str, int],
    timing: dict[str, float],
    bronze_root: Path,
    truth_dir: Path,
    pilot: bool,
) -> Path:
    chk["bronze_root"], chk["truth_dir"] = str(bronze_root), str(truth_dir)
    results = sorted(chk["results"], key=lambda r: int(r.number))
    figs_dir = ctx.run_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    reg = load_register()
    fig_files: list[Path] = []
    if pilot:
        forced = chk["truth"].loc[chk["truth"]["forced"].astype(bool), "point_id"].unique()
        for pid in forced:
            fig_files += pilot_figures.burst_figures(cfg, reg, bronze_root, chk["truth"], pid, figs_dir)
            four = next(r for r in results if r.number == "4").detail
            tf = pilot_figures.trend_figure(chk["merged"], four, pid, figs_dir)
            if tf:
                fig_files.append(tf)
    passed = sum(1 for r in results if r.passed)
    lines = [
        f"Run: {ctx.run_id}   Config hash: {ctx.config_hash}   Generator version: {ctx.generator_version}",
        f"Checks: {passed}/{len(results)}",
        "",
        "| # | Check | Expected | Measured | Result |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        res = "pass" if r.passed else ("not exercised" if r.passed is None else "FAIL")
        lines.append(f"| {r.number} | {r.name} | {r.expected} | {r.measured} | {res} |")
    lines += [
        "",
        "Config diff from previous run:",
        "```",
        config_diff(ctx),
        "```",
        "",
    ]
    if pilot:
        lines += [
            "Headline counts (pilot scope):",
            "",
            _md_table(headline_counts(chk, counts)),
            "",
            "Timing:",
            "",
            _md_table(timing_projection(cfg, chk, timing)),
            "",
        ]
    else:
        head = chk["headline"].copy()
        head.to_parquet(ctx.run_dir / "headline_counts.parquet", index=False)
        prior = previous_run(ctx)
        if prior is not None and (prior / "headline_counts.parquet").exists():
            prev = pd.read_parquet(prior / "headline_counts.parquet")[["quantity", "measured"]].rename(
                columns={"measured": f"previous ({prior.name[:5]})"}
            )
            head = head.merge(prev, on="quantity", how="left")
        if "episodes" in chk:
            lines += [
                "Degradation episodes reaching stage 2, by asset class and outcome:",
                "",
                _md_table(chk["episodes"]),
                "",
            ]
        lines += [
            "Headline counts (section 16):",
            "",
            _md_table(head),
            "",
            "Timing and volume:",
            "",
            _md_table(full_timing(timing, counts, bronze_root, truth_dir)),
            "",
        ]
    for r in results:
        if r.detail is not None and not r.detail.empty:
            lines += [f"Check {r.number} detail:", "", _md_table(r.detail), ""]
    if fig_files:
        lines += ["Figures:", ""] + [f"- figures/{p.name}" for p in fig_files] + [""]
    path = ctx.run_dir / "run_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

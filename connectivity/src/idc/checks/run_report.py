"""Assemble the run report (``runs/run_report_<stamp>.md``) from a chain run's artifacts.

Inputs: the orchestrator summary JSON, the newest acceptance JSON written after the run
started, the landing writer reports written during the run, and optional static sections
(``runs/report_sections/*.md``, included in name order) recording pre-launch test results,
decisions and the file inventory.

Usage:
    python -m idc.checks.run_report --summary runs/chain_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _newest(paths: list[Path], after: str) -> Path | None:
    stamped = [p for p in paths if p.stem.split("_")[-1] >= after]
    return max(stamped, key=lambda p: p.stem) if stamped else None


def _duration(start: str | None, end: str | None) -> str:
    if not start or not end:
        return "-"
    delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60} h {minutes % 60:02d} min"


def build(summary_path: Path) -> str:
    summary = _load(summary_path)
    reports = summary_path.parent
    stamp = summary["stamp"]
    lines: list[str] = ["# Connectivity chain: run report", ""]
    lines += [f"Run `{stamp}`; status **{summary['status']}**; started {summary['started_utc']} "
              f"UTC, finished {summary.get('finished_utc', '-')} UTC.", ""]

    acceptance_path = _newest(sorted(reports.glob("acceptance_*.json")), stamp)
    lines += ["## Acceptance", ""]
    if acceptance_path is None:
        lines += ["Acceptance did not run.", ""]
    else:
        acc = _load(acceptance_path)
        lines += [f"{acc['passed']} of {acc['targets']} targets pass "
                  f"(`{acceptance_path.name}`).", "",
                  "| Result | Table | Rows expected | Rows landed | Path |",
                  "|---|---|---|---|---|"]
        for r in acc["results"]:
            exp = f"{r['rows_expected']:,}" if r["rows_expected"] is not None else "-"
            act = f"{r['rows_actual']:,}" if r["rows_actual"] is not None else "-"
            lines.append(f"| {'PASS' if r['ok'] else 'FAIL'} | {r['table']} | {exp} | {act} | "
                         f"{r['path']} |")
        failures = [r for r in acc["results"] if not r["ok"]]
        if failures:
            lines += ["", "Failure details:", ""]
            lines += [f"- `{r['path']}`: {r['detail']}" for r in failures]
        lines.append("")

    hmc = summary.get("hmc", {})
    cmp = summary.get("compressor", {})
    lines += ["## Paths and timing", "",
              "| Data set | Path | Start (UTC) | Finish (UTC) | Duration |",
              "|---|---|---|---|---|"]
    mm = hmc.get("machine_months", {})
    starts = [r["started_utc"] for runs in mm.values() for r in runs]
    ends = [r["finished_utc"] for runs in mm.values() for r in runs]
    if mm:
        lines.append(f"| HMC hz1 and minute_agg | MTConnect, bridge, MQTT JSON | {min(starts)} | "
                     f"{hmc.get('finished_utc', max(ends))} | "
                     f"{_duration(min(starts), hmc.get('finished_utc', max(ends)))} |")
    chunks = cmp.get("chunks", [])
    if chunks:
        lines.append(f"| Compressor minute and room | {cmp.get('path')} | "
                     f"{chunks[0]['started_utc']} | {cmp.get('finished_utc', '-')} | "
                     f"{_duration(chunks[0]['started_utc'], cmp.get('finished_utc'))} |")
    lines.append("")

    lines += ["## Events", "",
              f"- Bridge restarts: {len(hmc.get('bridge_restarts', []))}",
              f"- Machine-months rerun after a sequence gap: "
              f"{len(hmc.get('sequence_gap_reruns', []))}",
              f"- Machine-months rerun after verification: {len(hmc.get('verify_reruns', []))}",
              f"- Trial resets: {len(cmp.get('trial_resets', []))}",
              f"- Modbus server restarts: {len(cmp.get('server_restarts', []))}",
              f"- Compressor gap ranges rerun: {len(cmp.get('gap_reruns', []))}",
              f"- Compressor gaps after rerun: {cmp.get('gaps_after_rerun', 'not computed')}", ""]
    for key, title in (("bridge_restarts", "Bridge restarts"),
                       ("sequence_gap_reruns", "Sequence-gap reruns"),
                       ("verify_reruns", "Verification reruns")):
        if hmc.get(key):
            lines += [f"{title}:", ""] + [f"- {json.dumps(e)}" for e in hmc[key]] + [""]
    for key, title in (("trial_resets", "Trial resets"), ("server_restarts", "Server restarts"),
                       ("gap_reruns", "Gap reruns")):
        if cmp.get(key):
            lines += [f"{title}:", ""] + [f"- {json.dumps(e)}" for e in cmp[key]] + [""]

    lines += ["## Writer reports", ""]
    for pattern in ("controller_landing_*.json", "compressor_landing_*.json"):
        for path in sorted(reports.glob(pattern)):
            if path.stem.split("_")[-1] < stamp:
                continue
            body = _load(path)
            keys = ("messages", "records", "accepted", "duplicates", "conflict_count", "gaps",
                    "link_losses", "rebirth_requests", "minutes_landed")
            brief = {k: body[k] if k != "gaps" else (len(body[k]) if isinstance(body[k], list)
                                                     else body[k])
                     for k in keys if k in body}
            lines.append(f"- `{path.name}`: {json.dumps(brief)}")
    lines.append("")

    if summary.get("errors"):
        lines += ["## Errors", ""]
        for e in summary["errors"]:
            lines += [f"- {e['path']} at {e['utc']}:", "", "```", e["error"].strip(), "```", ""]

    sections = reports / "report_sections"
    if sections.exists():
        for path in sorted(sections.glob("*.md")):
            lines += [path.read_text(encoding="utf-8").rstrip(), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_report")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    stamp = _load(args.summary)["stamp"]
    out = args.out or args.summary.parent / f"run_report_{stamp}.md"
    out.write_text(build(args.summary), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

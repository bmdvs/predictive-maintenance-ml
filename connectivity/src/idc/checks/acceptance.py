"""Acceptance check: every landed target against the bronze export it must reproduce.

Targets come from the ``acceptance`` list in the chain configuration (paths relative to both
``export_root`` and ``landing_root``). Each is compared with ``compare_files``: column order,
dtypes, row count, then exact values. Prints a pass/fail table, writes
``<run_reports>/acceptance_<stamp>.json`` and exits 1 if any target fails or is missing.

Usage:
    python -m idc.checks.acceptance [--config config/chain.yaml]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from idc.checks.compare_exports import compare_files, load_config


def run_acceptance(config: dict) -> list[dict]:
    """Compare every acceptance target and return one result record per target."""
    export_root = Path(config["paths"]["export_root"])
    landing_root = Path(config["paths"]["landing_root"])
    results: list[dict] = []
    for target in config["acceptance"]:
        expected = export_root / target["path"]
        actual = landing_root / target["path"]
        record = {"table": target["table"], "path": target["path"], "ok": False,
                  "rows_expected": None, "rows_actual": None, "detail": ""}
        if not actual.exists():
            record["detail"] = "landed file missing"
        else:
            try:
                result = compare_files(target["table"], expected, actual)
            except Exception as exc:  # report and continue with the remaining targets
                record["detail"] = f"{type(exc).__name__}: {exc}"
            else:
                record.update(ok=result.ok, rows_expected=result.rows_expected,
                              rows_actual=result.rows_actual, detail=result.detail)
        results.append(record)
    return results


def format_table(results: list[dict]) -> str:
    lines = [f"{'status':6s} {'table':11s} {'rows expected':>13s} {'rows landed':>12s}  path"]
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        exp = f"{r['rows_expected']:,d}" if r["rows_expected"] is not None else "-"
        act = f"{r['rows_actual']:,d}" if r["rows_actual"] is not None else "-"
        lines.append(f"{status:6s} {r['table']:11s} {exp:>13s} {act:>12s}  {r['path']}")
        if not r["ok"]:
            lines.append(f"       {r['detail']}")
    passed = sum(r["ok"] for r in results)
    lines.append(f"{passed} of {len(results)} targets pass")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acceptance")
    parser.add_argument("--config", type=Path, default=Path("config/chain.yaml"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    results = run_acceptance(config)
    print(format_table(results))
    run_dir = Path(config["paths"]["run_reports"])
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = run_dir / f"acceptance_{stamp}.json"
    out.write_text(json.dumps({"finished_utc": stamp, "targets": len(results),
                               "passed": sum(r["ok"] for r in results), "results": results},
                              indent=2), encoding="utf-8")
    print(f"report {out}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

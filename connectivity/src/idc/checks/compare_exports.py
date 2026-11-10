"""Exact comparison of landed Parquet files against the bronze exports they must reproduce.

The landing writers rebuild the controller and compressor tables from the live chain; this
module decides whether they match. Two files are equal only if they hold the same rows, in the
same column order, with the same dtypes (including timestamp unit and time zone) and exactly
the same values. Rows are sorted on the table's key columns before comparison because the
chain lands rows in arrival order, which can differ from the export's write order.

Usage:
    python -m idc.checks.compare_exports describe <parquet>
    python -m idc.checks.compare_exports compare <table> <expected> <actual> [--day YYYY-MM-DD]
    python -m idc.checks.compare_exports selfcheck [--config config/chain.yaml]

`selfcheck` compares every acceptance target against itself and recomputes each minute_agg
target from its hz1 file with the pipeline's own aggregation. Both must pass before the
comparator is trusted to judge chain output.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml
from pandas.testing import assert_frame_equal

# Key columns per bronze table. A key identifies one row; sorting on it makes row order
# irrelevant to the comparison.
TABLE_KEYS: dict[str, list[str]] = {
    "ctrl_hz1": ["machine_id", "ts_utc"],
    "ctrl_minute": ["machine_id", "minute_utc", "program", "execution"],
    "cmp_minute": ["compressor_id", "minute_utc"],
    "cmp_room": ["minute_utc"],
}


@dataclass
class ComparisonResult:
    """Outcome of one file comparison; `detail` names the first difference on failure."""

    table: str
    expected: Path
    actual: Path
    ok: bool
    rows_expected: int
    rows_actual: int
    detail: str = ""

    def summary(self) -> str:
        status = "OK  " if self.ok else "FAIL"
        line = (
            f"{status} {self.table:11s} rows expected={self.rows_expected:>9,d} "
            f"actual={self.rows_actual:>9,d}  {self.actual}"
        )
        return line if self.ok else f"{line}\n      {self.detail}"


def describe(path: Path) -> str:
    """Return the Parquet schema as text, with the timestamp unit and time zone visible."""
    schema = pq.read_schema(path)
    meta = pq.read_metadata(path)
    lines = [f"{path}", f"rows={meta.num_rows:,d}  row_groups={meta.num_row_groups}"]
    for field in schema:
        lines.append(f"  {field.name:28s} {field.type}")
    return "\n".join(lines)


def load_sorted(path: Path, table: str) -> pd.DataFrame:
    """Read a bronze file and sort it on the table's key columns.

    Column order and dtypes are left exactly as stored; only row order is normalized.
    """
    keys = TABLE_KEYS[table]
    df = pd.read_parquet(path)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"{path}: key columns missing for {table}: {missing}")
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame) -> str | None:
    """Return None if the frames are exactly equal, otherwise a short description of why not.

    The checks run from coarse to fine so the message names the first thing that differs:
    column set and order, dtypes, row count, then values.
    """
    if list(expected.columns) != list(actual.columns):
        return (
            "column order differs:\n"
            f"      expected {list(expected.columns)}\n"
            f"      actual   {list(actual.columns)}"
        )
    dtype_diffs = [
        f"{c}: expected {expected[c].dtype}, actual {actual[c].dtype}"
        for c in expected.columns
        if str(expected[c].dtype) != str(actual[c].dtype)
    ]
    if dtype_diffs:
        return "dtypes differ: " + "; ".join(dtype_diffs)
    if len(expected) != len(actual):
        return f"row count differs: expected {len(expected):,d}, actual {len(actual):,d}"
    try:
        assert_frame_equal(expected, actual, check_exact=True)
    except AssertionError as exc:
        return _first_value_difference(expected, actual) or str(exc).splitlines()[0]
    return None


def _first_value_difference(expected: pd.DataFrame, actual: pd.DataFrame) -> str | None:
    """Locate the first differing cell so a failure points at a column and a row."""
    for column in expected.columns:
        e, a = expected[column], actual[column]
        both_nan = e.isna() & a.isna()
        differs = ~both_nan & ((e != a) | (e.isna() != a.isna()))
        if differs.any():
            i = int(differs.idxmax())
            return (
                f"values differ in column '{column}': {int(differs.sum()):,d} rows; "
                f"first at row {i}: expected {e.iloc[i]!r}, actual {a.iloc[i]!r}"
            )
    return None


# Time column per bronze table, used to restrict a comparison to one UTC day.
TIME_KEY = {"ctrl_hz1": "ts_utc", "ctrl_minute": "minute_utc", "cmp_minute": "minute_utc",
            "cmp_room": "minute_utc"}


def restrict_day(frame: pd.DataFrame, table: str, day: str) -> pd.DataFrame:
    """Keep the rows of one UTC day, so a partial landing can be judged against a month file."""
    start = pd.Timestamp(day, tz="UTC")
    col = frame[TIME_KEY[table]]
    return frame[(col >= start) & (col < start + pd.Timedelta(days=1))].reset_index(drop=True)


def compare_files(table: str, expected_path: Path, actual_path: Path,
                  day: str | None = None) -> ComparisonResult:
    """Compare two bronze files of the same table exactly, optionally on one UTC day."""
    expected = load_sorted(expected_path, table)
    actual = load_sorted(actual_path, table)
    if day:
        expected = restrict_day(expected, table, day)
        actual = restrict_day(actual, table, day)
    detail = compare_frames(expected, actual)
    return ComparisonResult(
        table=table,
        expected=expected_path,
        actual=actual_path,
        ok=detail is None,
        rows_expected=len(expected),
        rows_actual=len(actual),
        detail=detail or "",
    )


def recompute_minutes(hz1_path: Path) -> pd.DataFrame:
    """Rebuild a minute_agg frame from an hz1 file with the pipeline's own aggregation.

    The aggregation is imported from predictive-maintenance-ml rather than reimplemented so
    that both sides of a comparison are produced by the same code.
    """
    from pdm.ingest.mtconnect_stream import HZ1_COLUMNS, aggregate_minutes

    hz = pd.read_parquet(hz1_path)
    minutes = aggregate_minutes(hz[HZ1_COLUMNS])
    minutes["run_id"] = hz["run_id"].iloc[0]
    minutes["config_hash"] = hz["config_hash"].iloc[0]
    return minutes


def load_config(path: Path) -> dict:
    """Load the chain config, applying chain.local.yaml from the same directory if present.

    Overrides merge one level deep: keys inside a mapping section update that section, and any
    other top-level value replaces the original. Relative entries under ``paths`` resolve from
    the connectivity root, the parent of the config directory, so commands run from anywhere.
    """
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = path.with_name("chain.local.yaml")
    if local.exists():
        override = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        for section, values in override.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section].update(values)
            else:
                config[section] = values
    root = path.resolve().parent.parent
    config["paths"] = {key: str((root / value).resolve()) if isinstance(value, str) else value
                       for key, value in config.get("paths", {}).items()}
    return config


def selfcheck(config_path: Path) -> int:
    """Verify the comparator on known-equal inputs. Returns 1 if any check fails, else 0."""
    config = load_config(config_path)
    export_root = Path(config["paths"]["export_root"])
    results: list[ComparisonResult] = []

    for target in config["acceptance"]:
        path = export_root / target["path"]
        results.append(compare_files(target["table"], path, path))

    for target in config["acceptance"]:
        if target["table"] != "ctrl_minute":
            continue
        minute_path = export_root / target["path"]
        hz1_path = Path(str(minute_path).replace("minute_agg", "hz1"))
        if not hz1_path.exists():
            results.append(
                ComparisonResult(
                    "ctrl_minute", minute_path, hz1_path, False, 0, 0,
                    f"hz1 file not found for recomputation: {hz1_path}",
                )
            )
            continue
        expected = load_sorted(minute_path, "ctrl_minute")
        recomputed = recompute_minutes(hz1_path)
        recomputed = recomputed.sort_values(
            TABLE_KEYS["ctrl_minute"], kind="mergesort"
        ).reset_index(drop=True)
        detail = compare_frames(expected, recomputed)
        results.append(
            ComparisonResult(
                "ctrl_minute", minute_path, hz1_path, detail is None,
                len(expected), len(recomputed), (detail or "") + "  (recomputed from hz1)",
            )
        )

    for r in results:
        print(r.summary())
    failed = sum(not r.ok for r in results)
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compare_exports")
    sub = parser.add_subparsers(dest="command", required=True)

    p_describe = sub.add_parser("describe", help="print a Parquet file's schema")
    p_describe.add_argument("path", type=Path)

    p_compare = sub.add_parser("compare", help="compare an actual file against an expected one")
    p_compare.add_argument("table", choices=sorted(TABLE_KEYS))
    p_compare.add_argument("expected", type=Path)
    p_compare.add_argument("actual", type=Path)
    p_compare.add_argument("--day", default=None, help="compare one UTC day only, YYYY-MM-DD")

    p_self = sub.add_parser("selfcheck", help="compare the acceptance targets against themselves")
    p_self.add_argument("--config", type=Path, default=Path("config/chain.yaml"))

    args = parser.parse_args(argv)
    if args.command == "describe":
        print(describe(args.path))
        return 0
    if args.command == "compare":
        result = compare_files(args.table, args.expected, args.actual, args.day)
        print(result.summary())
        return 0 if result.ok else 1
    return selfcheck(args.config)


if __name__ == "__main__":
    sys.exit(main())

"""Model report appendix C and I rows from appendix.yaml, and the check that traces every number in them.

Inputs: appendix.yaml, the Data object (gold snapshot and config). Outputs: the two tables as DataFrames and
a list of check problems (an untraced number or point id, or a trace that does not match the record).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yaml

from pdm.reports.data import Data

SOURCE = Path(__file__).with_name("appendix.yaml")
WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
NUMBER = re.compile(r"(?<![A-Za-z0-9.-])\d+(?:\.\d+)?(?![A-Za-z0-9])")
WORD = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.I)
POINT_ID = re.compile(r"\b(?:HMC|CMP)-\d{2}(?:-[A-Z]+)+\b")


def _window_only(d: Data) -> list[str]:
    f = d.st.read_pd("monitoring", "feature_drift")
    f = f[f["as_of"] == f["as_of"].max()]
    return sorted(f.loc[f["reference"].str.contains("window only"), "point_id"].unique())


RECORD: dict[str, Callable[[Data], object]] = {
    "inspection_hours": lambda d: d.inspection_hours,
    "window_only_points": _window_only,
    "window_only_spindle_points": lambda d: sum("-SP-" in p for p in _window_only(d)),
}


def load(path: Path = SOURCE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def definitions(content: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [(r["term"], r["text"]) for r in content["definitions"]], columns=["Term", "Definition"]
    )


def limitations(content: dict) -> list[str]:
    return [r["text"] for r in content["limitations"]]


def _value(d: Data, source: str) -> object:
    kind, _, key = source.partition(":")
    if kind == "config":
        v: object = d.cfg
        for k in key.split("."):
            v = v[k]  # type: ignore[index]
        return v
    if kind == "record":
        return RECORD[key](d)
    raise KeyError(f"unknown source {source!r}")


def _matches(shown: str, value: object) -> bool:
    if not isinstance(value, (int, float)):
        return False
    decimals = len(shown.split(".")[1]) if "." in shown else 0
    return f"{float(value):.{decimals}f}" == f"{float(shown):.{decimals}f}"


def check(d: Data, content: dict | None = None) -> tuple[list[str], int]:
    """Every number and point id in the appendix rows traced to the record. Returns problems and the count traced."""
    content = content or load()
    problems, traced = [], 0
    rows = [("C", r) for r in content["definitions"]] + [("I", r) for r in content["limitations"]]
    for app, r in rows:
        text, trace = r["text"], {str(k): v for k, v in (r.get("trace") or {}).items()}
        label = f"appendix {app}: {text[:50]}"
        found = NUMBER.findall(text) + [w.lower() for w in WORD.findall(text)]
        for tok in found:
            if tok not in trace:
                problems.append(f"{label}: number {tok!r} has no trace")
                continue
            shown = str(WORDS.get(tok, tok))
            v = _value(d, trace[tok])
            if not _matches(shown, v):
                problems.append(f"{label}: {tok!r} against {trace[tok]} = {v!r}")
            else:
                traced += 1
        for k in set(trace) - set(found) - {"ids", "ids_all"}:
            problems.append(f"{label}: trace {k!r} not in the text")
        ids = POINT_ID.findall(text)
        key = "ids_all" if "ids_all" in trace else "ids"
        if ids and key not in trace:
            problems.append(f"{label}: point ids without a trace")
        elif key in trace:
            want = set(_value(d, trace[key]))  # type: ignore[arg-type]
            bad = set(ids) - want if key == "ids" else set(ids) ^ want
            if bad:
                problems.append(f"{label}: point ids against {trace[key]}: {sorted(bad)}")
            else:
                traced += len(ids)
    return problems, traced

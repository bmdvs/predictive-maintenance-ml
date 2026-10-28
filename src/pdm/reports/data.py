"""Shared inputs for the pages, read once from the gold snapshot through the pipeline's own functions.

Inputs: exports/gold_snapshot (SnapshotStore) and config.yaml. Outputs: the Data object with the tables,
conditions, alert episodes and derived frames the three pages and the README use.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from pdm import tasks
from pdm.config import load_config
from pdm.evaluation import events as ev
from pdm.reports.snapshot import SnapshotStore

TZ = "America/Chicago"
END = pd.Timestamp("2026-07-01 05:00", tz="UTC")
Q2 = pd.Timestamp("2026-04-01 05:00", tz="UTC")
AS_OF = "2026-06"
METHODS = ("rule_floor", "nbm", "controller", "model_layer_instrumented")
LABEL = {
    "rule_floor": "Vendor rules",
    "nbm": "Per-point model",
    "controller": "Controller health",
    "model_layer_instrumented": "Model layer",
    "model_layer": "Model layer, all assets",
    "model_layer_uninstrumented": "Model layer, uninstrumented",
    "controller_uninstrumented": "Controller health, uninstrumented",
    "vendor": "Vendor rules",
    "iso": "ISO zones",
}
LAYER_LABEL = {
    "vendor": "Vendor rules",
    "iso": "Vendor rules",
    "nbm": "Per-point model",
    "controller": "Controller health",
}
YEAR, QUARTER = "2025-07_to_2026-06", "2026-Q2"


def utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def naive_date(s: pd.Series) -> pd.Series:
    """A date column as naive midnight timestamps (dates exported as UTC midnight keep their day)."""
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.dt.normalize()


def local(ts: Any, fmt: str = "%Y-%m-%d") -> str:
    if ts is None or pd.isna(ts):
        return ""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(TZ).strftime(fmt)


class Data:
    def __init__(self) -> None:
        self.st = SnapshotStore()
        self.cfg = load_config()
        self.cfg["evaluation"]["as_of"] = AS_OF
        self.op = tasks._operating(self.cfg)

    def g(self, table: str) -> pd.DataFrame:
        return self.st.read_pd("gold", table)

    def m(self, table: str) -> pd.DataFrame:
        return self.st.read_pd("monitoring", table)

    @cached_property
    def inputs(self) -> dict[str, Any]:
        return tasks._evaluation_inputs(self.st, self.cfg)

    @cached_property
    def conds(self) -> pd.DataFrame:
        return self.inputs["conds"]

    @cached_property
    def episodes(self) -> pd.DataFrame:
        parts = []
        for m, (ep, n) in tasks._monitored_episodes(self.cfg, self.inputs).items():
            e = ep.copy()
            e["opened"], e["closed"] = utc(e["opened"]), utc(e["closed"])
            parts.append(e.assign(method=m, n_assets=n))
        return pd.concat(parts, ignore_index=True)

    @cached_property
    def wo(self) -> pd.DataFrame:
        return tasks._wo(self.st)

    @cached_property
    def machines(self) -> pd.DataFrame:
        return self.st.read_pd("bronze", "reg_machines")

    @cached_property
    def sensors(self) -> pd.DataFrame:
        return self.st.read_pd("bronze", "reg_sensors")

    @cached_property
    def cases(self) -> pd.DataFrame:
        c = self.g("cases")
        for k in ("opened", "closed", "last_signal", "first_alarm", "chain_opened"):
            if k in c:
                c[k] = utc(c[k])
        return c

    @cached_property
    def requests(self) -> pd.DataFrame:
        w = self.g("work_requests")
        w["created"] = utc(w["created"])
        return w

    @cached_property
    def status_at_end(self) -> pd.Series:
        s = self.g("asset_status")
        s["ts"] = utc(s["ts"])
        return s[s["ts"] < END].sort_values("ts").groupby("asset_id")["status"].last()

    @cached_property
    def open_cases(self) -> pd.DataFrame:
        c = self.cases
        return c[(c["opened"] < END) & (c["closed"] >= END)].set_index("asset_id")

    def events(self, table: str, period: str | None = None) -> pd.DataFrame:
        """Evaluation events at each method's operating point."""
        e = self.g(table)
        if period is not None:
            e = e[e["period"] == period]
        keep = [
            np.isclose(float(lv), float(self.op.get(m, -1)))
            for m, lv in zip(e["method"], e["level"], strict=True)
        ]
        return e[keep].copy()

    @cached_property
    def outcomes(self) -> pd.DataFrame:
        """Work request outcomes as of the record end: confirmed, still open, or closed with no finding recorded."""
        d = self.g("diagnosis_agreement_shadow")
        q = d[d["period"] == QUARTER].merge(
            self.requests[["request_id", "case_id", "created"]], on="request_id", suffixes=("", "_wr")
        )
        open_end = self.cases.set_index("case_id")["closed"] >= END
        q["case_open_at_end"] = q["case_id"].map(open_end).fillna(False).astype(bool)
        q["outcome_end"] = np.where(
            q["outcome"] == "confirmed",
            "Confirmed finding",
            np.where(q["case_open_at_end"], "Still open at the record end", "Closed, no finding recorded"),
        )
        q["month"] = q["created_wr"].dt.tz_convert(TZ).dt.strftime("%Y-%m")
        return q

    @cached_property
    def inspection_hours(self) -> float:
        return float(tasks._context(self.st)["inspection_hours_each"])

    @cached_property
    def silver_2026(self) -> pd.DataFrame:
        s = self.st.read_pd("silver", "hourly_indicators")
        s["hour_utc"] = utc(s["hour_utc"])
        return s[~s["implausible"].fillna(False).astype(bool)]

    @cached_property
    def pressure(self) -> pd.DataFrame:
        """Daily through-spindle coolant pressure per HMC: pump-on (gold) and the advisory's duty-weighted series (silver)."""
        on = self.g("tsc_pressure_daily")[["machine_id", "local_date", "pressure_median_bar"]].rename(
            columns={"pressure_median_bar": "pump_on_bar"}
        )
        ch = self.st.read_pd("silver", "controller_hourly")
        ch = ch[ch["coolant_pressure_bar"] > 0].copy()
        ch["local_date"] = utc(ch["hour_utc"]).dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
        duty = (
            ch.groupby(["machine_id", "local_date"])["coolant_pressure_bar"]
            .median()
            .rename("advisory_bar")
            .reset_index()
        )
        on["local_date"] = naive_date(on["local_date"])
        return on.merge(duty, on=["machine_id", "local_date"], how="outer").sort_values(
            ["machine_id", "local_date"]
        )

    @cached_property
    def pending(self) -> pd.DataFrame:
        """False episodes per method for the year and the quarter as published, with those whose 30-day
        outcome window is still open at the record end in a separate column (definitions unchanged)."""
        w = pd.Timedelta(days=self.cfg["evaluation"]["alert_precision_window_days"])
        b = tasks.splits.boundaries(self.cfg)
        periods = {"year": (b["validation"][0], END), "quarter": (b["shadow"][0], END)}
        rows = []
        for m in METHODS:
            ep = self.episodes[self.episodes["method"] == m]
            n = int(ep["n_assets"].iat[0])
            for name, (a, z) in periods.items():
                e = ep[(ep["opened"] >= a) & (ep["opened"] < z)].reset_index(drop=True)
                fe = ev.false_episodes(e, self.conds).to_numpy() if len(e) else np.array([], bool)
                pend = (e["opened"] + w > END).to_numpy() if len(e) else np.array([], bool)
                months = (z - a) / pd.Timedelta(days=365.25 / 12)
                rows.append(
                    {
                        "method": m,
                        "period": name,
                        "episodes": len(e),
                        "false": int(fe.sum()),
                        "false_rate": fe.sum() / (n * months),
                        "pending": int((fe & pend).sum()),
                        "false_closed_rate": (fe.sum() - (fe & pend).sum()) / (n * months),
                    }
                )
        return pd.DataFrame(rows)

    @cached_property
    def leaks(self) -> pd.DataFrame:
        """Per rotary-union leak work order: the pump-on low span (daily pump-on median below 95% of the median
        45 to 135 days before the work order, contiguous within 3 days, the span starting by the opening),
        its drop and lead, and the advisory series over the same span."""
        w = self.wo[self.wo["failure_code"] == "rotary_union_leak"].copy()
        p = self.pressure
        period = self.conds.set_index("wo_id")["period"].to_dict()
        rows = []
        for r in w.itertuples():
            mch = r.asset_id[:6]
            o = pd.Timestamp(r.opened).tz_convert(TZ).tz_localize(None).normalize()
            c = pd.Timestamp(r.closed).tz_convert(TZ).tz_localize(None).normalize()
            g = p[p["machine_id"] == mch].set_index("local_date")
            base = g.loc[
                (g.index < o - pd.Timedelta(days=45)) & (g.index >= o - pd.Timedelta(days=135)), "pump_on_bar"
            ].median()
            low = g[
                (g["pump_on_bar"] < 0.95 * base)
                & (g.index >= o - pd.Timedelta(days=45))
                & (g.index <= c + pd.Timedelta(days=2))
            ]
            days = low.index.to_series()
            grp = (days.diff().dt.days > 3).cumsum()
            starts = days.groupby(grp).min()
            before = starts[starts <= o + pd.Timedelta(days=2)]
            span = low[grp == (before.index[-1] if len(before) else grp.iloc[0])]
            s0, s1 = span.index.min(), span.index.max()
            adv = g.loc[(g.index < s0) & (g.index >= s0 - pd.Timedelta(days=30)), "advisory_bar"]
            rows.append(
                {
                    "wo_id": r.wo_id,
                    "asset_id": r.asset_id,
                    "machine_id": mch,
                    "period": period.get(r.wo_id, ""),
                    "opened": o,
                    "span_start": s0,
                    "span_end": s1,
                    "span_days": len(span),
                    "before_bar": base,
                    "span_bar": span["pump_on_bar"].median(),
                    "lowest_bar": span["pump_on_bar"].min(),
                    "drop_bar": base - span["pump_on_bar"].median(),
                    "drop_pct": 100 * (base - span["pump_on_bar"].median()) / base,
                    "lead_days": (o - s0).days,
                    "advisory_spread_bar": adv.std(),
                }
            )
        return pd.DataFrame(rows).sort_values("opened").reset_index(drop=True)

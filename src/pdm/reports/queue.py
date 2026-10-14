"""The Maintenance Queue screen (docs/index.html): the queue of assets with an open Vendor Rules or ML Model
status at the snapshot hour, and below it the detail section of the selected asset.

Inputs: the build cache (pdm.reports.queue_data, prod tables; bursts per pdm.reports.queue_bursts).
Output: docs/index.html, with the snapshot, active list and check facts in build/queue_cache/page.json.

python -m pdm.reports.queue
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from pdm import log
from pdm.config import load_config
from pdm.features import kinematics
from pdm.features.burst import ExtractionSettings
from pdm.reports import queue_bursts as qb
from pdm.reports import queue_svg as sv
from pdm.reports.queue_state import BANDS, METHODS, RANK, TZ, QueueState, asset_kind, local, machine_of
from pdm.reports.style import DOCS

LOG = log.get_logger("pdm.reports.queue")
PRODUCT = "Maintenance Center"
USER = "maint.planner"
LEVEL_CLASS = {"ALARM": "red", "ALERT": "amber", "WATCH": "gray", "OK": "gray"}
TYPE_LABEL = {
    "pm": "PM",
    "inspection": "Inspection",
    "unplanned_repair": "Unplanned repair",
    "condition_based": "Condition-based repair",
    "corrective": "Corrective repair",
    "project": "Project",
    "rebuild_order": "Rebuild order",
}
MEASURES = [
    ("vel", "vel_rms_mm_s", "Velocity RMS", "mm/s", 20.0),
    ("hf", "hf_rms_g", "HF accel", "g", 20.0),
    *[(k, f"env_{k}_1_g2", f"Envelope {n}", "g²", 10.0) for k, n, _ in BANDS],
]
RULE_MEASURE = {"vel": "vel", "hf": "hf", "zone": "vel", "envelope": "env", "temp": "temp"}


def h(s: object) -> str:
    return sv.esc(s)


def chip(level: str, extra: str = "") -> str:
    return f'<span class="chip {LEVEL_CLASS.get(level, "gray")}">{h(level)}</span>{extra}'


def days_open(opened: pd.Timestamp, T: pd.Timestamp) -> int:
    return int((T - opened) / pd.Timedelta(days=1))


def outcome(r: pd.Series) -> str:
    """The work order's outcome as recorded: the inspection finding, the repair action or the PM task."""
    notes = str(r["notes"] or "").strip().rstrip(".")
    if r["wo_type"] == "inspection":
        if pd.isna(r["closed"]):
            return "open"
        return notes.split(": ", 1)[-1] if ": " in notes else notes
    if r["wo_type"] == "pm":
        return notes.split(": ", 1)[-1] if ": " in notes else notes
    return notes.split("; ")[-1] if notes else ("open" if pd.isna(r["closed"]) else "completed")


def wo_label(r: pd.Series) -> str:
    return f"{r['wo_id']}, {TYPE_LABEL.get(r['wo_type'], r['wo_type'])}: {outcome(r)}"


RULE_NAME = {
    "vel_alarm": "velocity alarm",
    "vel_alert": "velocity alert",
    "hf_alarm": "HF alarm",
    "hf_alert": "HF alert",
    "envelope": "envelope fault band",
    "temp_alarm": "temperature alarm",
    "temp_alert": "temperature warning",
    "zone_C": "ISO zone C",
    "zone_D": "ISO zone D",
}


def cause_label(cause: str | None) -> str:
    """The ALARM cause in plain terms: the Vendor Rules alarm-level rules that fired, or the ML Model criterion."""
    if not cause:
        return "cause not recorded"
    if cause.startswith("vendor "):
        rules = [r for r in cause[len("vendor ") :].split(",") if r]
        alarm = [r for r in rules if r.endswith("alarm") or r == "zone_D"] or rules
        return ", ".join(RULE_NAME.get(r, r) for r in alarm)
    return cause


def request_number(rid: str) -> str:
    return str(rid).replace("|", "-")


def family_label(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


class Page:
    def __init__(self, state: QueueState | None = None) -> None:
        self.q = state or QueueState()
        self.T = self.q.T
        self.t0 = self.T - pd.Timedelta(days=90)
        self.cfg = load_config()
        self.xs = ExtractionSettings.from_config(self.cfg)
        self.lv = self.q.levels()
        self.facts: dict[str, Any] = {"assets": {}}
        self.uid = 0

    # ------------------------------------------------------------------ queue rows
    def rows(self) -> list[dict[str, Any]]:
        out = []
        for a, m in self.lv.items():
            v, ml = m.get("vendor_rules"), m.get("ml_model")
            key = ("vendor_rules", v.level) if v else ("ml_model", ml.level)
            lead = v or ml
            out.append(
                {
                    "asset": a,
                    "v": v,
                    "ml": ml,
                    "rank": RANK[key],
                    "opened": lead.opened,
                    "wr": self.q.work_request(a, ml) if ml else None,
                }
            )
        return sorted(out, key=lambda r: (r["rank"], r["opened"], r["asset"]))

    def leading(self, asset: str, v: Any, ml: Any) -> str:
        """What drives the row. With a Vendor Rules episode open: at its most recent alert, the fired measure furthest
        above its healthy level (dB above the Vendor Rules healthy level; the reading for a temperature rule). With the
        ML Model alone: the score against its threshold at the last exceedance."""
        pts = self.q.points(asset)
        if v is not None:
            al = self.q.vendor_alerts
            al = al[
                (al["asset_id"] == asset) & (al["alert_ts"] >= v.opened) & (al["alert_ts"] <= self.T)
            ].sort_values("alert_ts")
            r = al.iloc[-1]
            pid, ts = r["point_id"], r["alert_ts"]
            where = f"{self.q.point_label(pid)}, " if len(pts) > 1 else ""
            fired = {x.split("_")[0] for x in str(r["rules"]).split(",") if x}
            row = self.q.indicators[
                (self.q.indicators["point_id"] == pid) & (self.q.indicators["hour_utc"] == ts)
            ]
            w = self.q.window_at(pid, ts)
            best: tuple[float, str] | None = None
            for key, col, name, _, factor in MEASURES:
                kind = key if key in ("vel", "hf") else "envelope"
                if not ({kind} & fired or (kind == "vel" and "zone" in fired)) or w is None or row.empty:
                    continue
                base, now = w[col], row[col].iloc[0]
                if pd.notna(base) and base > 0 and pd.notna(now) and now > 0:
                    db = factor * math.log10(now / base)
                    best = (db, f"{name} {db:+.1f} dB") if best is None or db > best[0] else best
            if best is None and "temp" in fired and not row.empty:
                best = (0.0, f"Housing temp {row['temp_c'].iloc[0]:.1f} °C")
            return f"{best[1]} ({where}{local(ts, '%b %d').replace(' 0', ' ')})" if best else "Not available"
        sc = self.q.scores
        g = sc[
            sc["point_id"].isin(pts)
            & (sc["hour_utc"] >= ml.opened)
            & (sc["hour_utc"] <= self.T)
            & (sc["state"] == "scored")
        ]
        g = g[g["score"] > g["threshold"]]
        if g.empty:
            return "Not available"
        r = g.sort_values("hour_utc").iloc[-1]
        where = f"{self.q.point_label(r['point_id'])}, " if len(pts) > 1 else ""
        return f"Score {r['score'] / r['threshold']:.1f}x threshold ({where}{local(r['hour_utc'], '%b %d').replace(' 0', ' ')})"

    def notes(self, asset: str, ml: Any) -> list[str]:
        out = []
        w = self.q.watch
        w = w[
            (w["asset_id"] == asset)
            & (w["hour_utc"] > self.T - pd.Timedelta(hours=24))
            & (w["hour_utc"] <= self.T)
        ]
        if ml is None and len(w):
            first = w["hour_utc"].min()
            out.append(f"ML Model WATCH since {local(first, '%Y-%m-%d %H:%M')}")
        seg = self.q.segments
        for pid in self.q.points(asset):
            s = seg[(seg["point_id"] == pid) & (seg["segment_start"] <= self.T)].sort_values("segment_start")
            if len(s) and bool(s["plausibility_flag"].iloc[-1]):
                out.append(
                    f"Post-repair inspection flag: {self.q.point_label(pid)} level outside its prior range since {local(s['segment_start'].iloc[-1])}"
                )
        return out

    def wo_cell(self, asset: str, wr: pd.Series | None) -> str:
        parts = []
        for r in self.q.open_work_orders(asset).itertuples():
            parts.append(
                f'<a class="wo" href="#" data-asset="{h(asset)}" data-wo="{h(r.wo_id)}">{h(r.wo_id)}</a> <span class="muted">{h(TYPE_LABEL.get(r.wo_type, r.wo_type))}, open</span>'
            )
        if wr is not None:
            parts.append(
                f'<a class="wo" href="#" data-asset="{h(asset)}" data-wr="1">{h(request_number(wr["request_id"]))}</a> <span class="tag">Shadow</span>'
            )
        if not parts:
            w = self.q.asset_work_orders(asset)
            w = w[w["wo_type"] != "pm"]
            if len(w):
                r = w.iloc[-1]
                parts.append(
                    f'<span class="muted">None open; last</span> <a class="wo" href="#" data-asset="{h(asset)}" data-wo="{h(r["wo_id"])}">{h(r["wo_id"])}</a>'
                )
            else:
                parts.append('<span class="muted">None open</span>')
        return "<br>".join(parts)

    def queue_table(self, rows: list[dict[str, Any]]) -> str:
        head = (
            "<tr><th>Asset</th><th>Vendor Rules<div class='sub'>Triggers inspection</div></th><th>ML Model <span class='tag'>Shadow</span></th>"
            "<th>Leading indicator</th><th>Suspected condition</th><th>Recommended inspection</th><th>Work request / work order</th></tr>"
        )
        filt = (
            "<tr class='filters'>"
            + "".join(
                f"<th><div class='fbox'>{x}<span class='caret'>▾</span></div></th>"
                for x in ("All assets", "All", "All", "All", "All", "All", "All")
            )
            + "</tr>"
        )
        body = []
        for r in rows:
            a, v, ml, wr = r["asset"], r["v"], r["ml"], r["wr"]
            vcell = (
                (chip(v.level) + f"<div class='sub'>Opened {local(v.opened)}</div>")
                if v
                else "<span class='muted'>OK</span>"
            )
            mcell = (
                (chip(ml.level) + f"<div class='sub'>Opened {local(ml.opened)}</div>")
                if ml
                else "<span class='muted'>OK</span>"
            )
            if wr is not None:
                cond = f"<b>{h(family_label(wr['suspected_condition']))}</b>" + (
                    f"<div class='sub'>{h(wr['suspected_detail'])}</div>" if wr["suspected_detail"] else ""
                )
                insp = h(family_label(str(wr["recommended_inspection"])))
            else:
                cond, insp = (
                    "<span class='muted'>Not diagnosed</span>",
                    "<span class='muted'>Not specified</span>",
                )
            notes = self.notes(a, ml)
            only = "" if v else " <span class='tag only'>ML Model only</span>"
            note_html = "".join(f"<div class='note'>{h(n)}</div>" for n in notes)
            body.append(
                f"<tr class='qrow' data-asset='{h(a)}'><td><div class='aname'>{h(a)}{only}</div><div class='sub'>{h(self.q.asset_name(a))} · {h(self.q.location(a))}</div>{note_html}</td>"
                f"<td>{vcell}</td><td>{mcell}</td><td>{h(self.leading(a, v, ml))}</td><td>{cond}</td><td>{insp}</td><td>{self.wo_cell(a, wr)}</td></tr>"
            )
            self.facts["assets"][a] = {
                "vendor": v.level if v else "OK",
                "ml": ml.level if ml else "OK",
                "suspected": None if wr is None else wr["suspected_condition"],
                "request": None if wr is None else request_number(wr["request_id"]),
                "notes": notes,
            }
        cols = "".join(f"<col style='width:{w}'>" for w in ("15%", "9%", "9%", "14%", "20%", "15%", "18%"))
        return f"<table class='queue'><colgroup>{cols}</colgroup><thead>{head}{filt}</thead><tbody>{''.join(body)}</tbody></table>"

    def ok_list(self) -> str:
        ok = [a for a in self.q.assets if a not in self.lv]
        items = []
        for a in ok:
            n = self.notes(a, None)
            w = self.q.asset_work_orders(a)
            last = w.iloc[-1] if len(w) else None
            lw = (
                f"Last work order {last['wo_id']}, {local(last['opened'])}"
                if last is not None
                else "No work orders"
            )
            items.append(
                f"<tr><td class='aname'>{h(a)}</td><td>{h(self.q.asset_name(a))} · {h(self.q.location(a))}</td><td>{chip('OK')}</td><td class='muted'>{h(lw)}</td><td>{''.join(f'<div class=note>{h(x)}</div>' for x in n)}</td></tr>"
            )
            self.facts.setdefault("ok", {})[a] = n
        return f"<details class='oklist'><summary>OK <span class='count'>{len(ok)} assets</span></summary><table class='plain'>{''.join(items)}</table></details>"

    # ------------------------------------------------------------------ detail: events and charts
    def events(self, asset: str, score: bool = False) -> list[tuple[pd.Timestamp, str, str]]:
        ev = []
        e = self.q.episodes
        e = e[e["asset_id"] == asset]
        for r in e.itertuples():
            name = METHODS[r.method]
            if r.opened < self.t0 < r.closed:
                lvl = "ALARM" if pd.notna(r.alarm_ts) and r.alarm_ts <= self.t0 else "ALERT"
                ev.append(
                    (self.t0, f"{name} {lvl} since {local(r.opened)}", sv.RED if lvl == "ALARM" else sv.AMBER)
                )
            same = pd.notna(r.alarm_ts) and r.alarm_ts - r.opened <= pd.Timedelta(hours=1)
            if self.t0 <= r.opened <= self.T:
                ev.append(
                    (
                        r.opened,
                        f"{name} ALERT and ALARM" if same else f"{name} ALERT",
                        sv.RED if same else sv.AMBER,
                    )
                )
            if pd.notna(r.alarm_ts) and self.t0 <= r.alarm_ts <= self.T and not same:
                ev.append((r.alarm_ts, f"{name} ALARM", sv.RED))
        w = self.q.asset_work_orders(asset)
        w = w[(w["opened"] >= self.t0) & ((w["wo_type"] != "pm") | w["failure_code"].notna())]
        for _, r in w.iterrows():
            ev.append((r["opened"], wo_label(r), sv.MUTED))
        if score:
            seg = self.q.segments
            for pid in self.q.points(asset):
                s = seg[
                    (seg["point_id"] == pid)
                    & (seg["segment_start"] >= self.t0)
                    & (seg["segment_start"] <= self.T)
                ]
                for r in s.itertuples():
                    ev.append((r.segment_start, f"Re-centered, {self.q.point_label(pid)}", sv.MUTED))
                    if r.plausibility_flag:
                        ev.append((r.level_ready, f"Plausibility flag, {self.q.point_label(pid)}", sv.MUTED))
            wt = self.q.watch
            wt = wt[
                (wt["asset_id"] == asset)
                & (wt["hour_utc"] >= self.t0)
                & (wt["reason"] == "unpersisted exceedance")
            ]
            open_eps = e[e["method"] == "ml_model"]
            starts = []
            last = None
            for t in sorted(wt["hour_utc"]):
                inside = ((open_eps["opened"] <= t) & (open_eps["closed"] > t)).any()
                if not inside and (last is None or t - last > pd.Timedelta(hours=72)):
                    starts.append(t)
                last = t
            for t in starts:
                ev.append((t, "WATCH", sv.MUTED))
        return sorted(ev, key=lambda x: x[0])

    def _chart(self, title: str, unit: str, chart: sv.TimeChart, level: str) -> str:
        self.uid += 1
        return (
            f"<div class='chartbox'><div class='ctitle'>{h(title)} <span class='unit'>{h(unit)}</span> {chip(level)}</div>"
            f"{chart.svg(str(self.uid))}</div>"
        )

    def _new(self, asset: str, score: bool = False, log: bool = False) -> sv.TimeChart:
        c = sv.TimeChart(self.t0, self.T, log=log)
        for ts, text, col in self.events(asset, score):
            c.event(ts, text, col)
        return c

    def score_chart(self, asset: str) -> str:
        c = self._new(asset, score=True, log=True)
        sc = self.q.scores
        e = self.q.episodes
        e = e[(e["asset_id"] == asset) & (e["method"] == "ml_model") & e["alarm_ts"].notna()]
        pts = self.q.points(asset)
        for i, pid in enumerate(pts):
            g = sc[(sc["point_id"] == pid) & (sc["hour_utc"] >= self.t0) & (sc["hour_utc"] <= self.T)]
            r = (g["score"] / g["threshold"]).where(g["state"] == "scored")
            alarm = np.zeros(len(g), dtype=bool)
            for x in e.itertuples():
                alarm |= ((g["hour_utc"] >= x.alarm_ts) & (g["hour_utc"] < x.closed)).to_numpy()
            c.fit(r.to_numpy())
            c.series(
                g["hour_utc"],
                r,
                sv.POINT_COLORS[i],
                [
                    (pd.Series(1.0, index=g.index), "alert"),
                    (pd.Series(np.where(alarm, 1.0, np.inf), index=g.index), "alarm"),
                ],
            )
        c.fit(np.array([1.0]))
        c.hline(self.t0, self.T, 1.0, sv.MUTED, "4 3", "Threshold")
        lvl = self.lv.get(asset, {}).get("ml_model")
        level = (
            lvl.level
            if lvl
            else (
                "WATCH"
                if self.notes(asset, None) and any("WATCH" in n for n in self.notes(asset, None))
                else "OK"
            )
        )
        return self._chart("ML Model score (shadow)", "score / threshold" + self._who(asset), c, level)

    def _who(self, asset: str) -> str:
        pts = self.q.points(asset)
        return "" if len(pts) == 1 else ", " + " dark, ".join(self.q.point_label(p) for p in pts) + " light"

    def _windows(self, pid: str) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Series]]:
        w = self.q.vendor_windows
        w = w[w["point_id"] == pid].sort_values("ready")
        out = []
        for i, r in enumerate(w.itertuples(index=False)):
            a = max(r.ready, self.t0)
            b = w["start"].iloc[i + 1] if i + 1 < len(w) else self.T
            b = min(b, self.T)
            if b > a:
                out.append((a, b, w.iloc[i]))
        return out

    def measure_chart(self, asset: str, col: str, title: str, unit: str, lim: str | None) -> str:
        c = self._new(asset)
        ind = self.q.indicators
        pts = self.q.points(asset)
        multi = len(pts) > 1
        worst = "OK"
        for i, pid in enumerate(pts):
            name = self.q.point_label(pid) + " " if multi else ""
            g = ind[(ind["point_id"] == pid) & (ind["hour_utc"] >= self.t0) & (ind["hour_utc"] <= self.T)]
            g = g[~g["implausible"].fillna(False).astype(bool)].reset_index(drop=True)
            col_c = sv.POINT_COLORS[i]
            al = pd.Series(np.inf, index=g.index)
            am = pd.Series(np.inf, index=g.index)
            wins = self._windows(pid)
            for j, (a, b, w) in enumerate(wins):
                last_piece = j == len(wins) - 1
                m = (g["hour_utc"] >= a) & (g["hour_utc"] < b)
                lo, hi = w.get(f"{col}_p05"), w.get(f"{col}_p95")
                if pd.notna(lo) and pd.notna(hi):
                    c.band(a, b, lo, hi, col_c, self._lab(name, "level") if last_piece else None)
                    c.fit(np.array([lo, hi]))
                if lim:
                    L = self.q.vendor_limits(pid, b - pd.Timedelta(hours=1))
                    if np.isfinite(L.get(f"{lim}_alert", np.inf)):
                        al[m], am[m] = L[f"{lim}_alert"], L[f"{lim}_alarm"]
                        c.hline(
                            a,
                            b,
                            L[f"{lim}_alert"],
                            sv.AMBER,
                            "5 3",
                            self._lab(name, "alert") if last_piece else None,
                        )
                        c.hline(
                            a,
                            b,
                            L[f"{lim}_alarm"],
                            sv.RED,
                            "5 3",
                            self._lab(name, "alarm") if last_piece else None,
                        )
                        c.fit(np.array([L[f"{lim}_alert"], L[f"{lim}_alarm"]]))
            c.fit(g[col].to_numpy())
            c.series(g["hour_utc"], g[col], col_c, [(al, "alert"), (am, "alarm")] if lim else None)
            last = g.dropna(subset=[col]).tail(1)
            if len(last) and lim:
                vv = float(last[col].iloc[0])
                if vv >= float(am.iloc[last.index[0]]):
                    worst = "ALARM"
                elif vv >= float(al.iloc[last.index[0]]) and worst != "ALARM":
                    worst = "ALERT"
        return self._chart(title, unit + self._who(asset), c, worst)

    @staticmethod
    def _lab(name: str, what: str) -> str:
        return f"{name}{what}" if name else what.capitalize()

    def temp_chart(self, asset: str) -> str:
        c = self._new(asset)
        ind = self.q.indicators
        pts = self.q.points(asset)
        multi = len(pts) > 1
        worst = "OK"
        for i, pid in enumerate(pts):
            name = self.q.point_label(pid) + " " if multi else ""
            g = ind[
                (ind["point_id"] == pid) & (ind["hour_utc"] >= self.t0) & (ind["hour_utc"] <= self.T)
            ].reset_index(drop=True)
            s = self.q.vendor_settings(pid)
            wins = self._windows(pid)
            for j, (a, b, w) in enumerate(wins):
                lo, hi = w.get("temp_c_p05"), w.get("temp_c_p95")
                if pd.notna(lo) and pd.notna(hi):
                    c.band(
                        a,
                        b,
                        lo,
                        hi,
                        sv.POINT_COLORS[i],
                        self._lab(name, "level") if j == len(wins) - 1 else None,
                    )
                    c.fit(np.array([lo, hi]))
            same = multi and i > 0 and self.q.vendor_settings(pts[0]).temp_warning_c == s.temp_warning_c
            if not same:
                c.hline(
                    self.t0,
                    self.T,
                    s.temp_warning_c,
                    sv.AMBER,
                    "5 3",
                    f"{name}alert" if multi and not self._same_temp(pts) else "Alert",
                )
                c.hline(
                    self.t0,
                    self.T,
                    s.temp_alarm_c,
                    sv.RED,
                    "5 3",
                    f"{name}alarm" if multi and not self._same_temp(pts) else "Alarm",
                )
            v = g["temp_c"].where(~g["implausible"].fillna(False).astype(bool))
            c.fit(v.to_numpy())
            lim_view = max(float(np.nanmax(v.to_numpy())) if v.notna().any() else 0, 0) * 1.0
            if s.temp_warning_c <= lim_view * 1.25:
                c.fit(np.array([s.temp_warning_c]))
            c.series(
                g["hour_utc"],
                v,
                sv.POINT_COLORS[i],
                [
                    (pd.Series(s.temp_warning_c, index=g.index), "alert"),
                    (pd.Series(s.temp_alarm_c, index=g.index), "alarm"),
                ],
            )
            last = v.dropna().tail(1)
            if len(last):
                if float(last.iloc[0]) >= s.temp_alarm_c:
                    worst = "ALARM"
                elif float(last.iloc[0]) >= s.temp_warning_c and worst != "ALARM":
                    worst = "ALERT"
        return self._chart("Housing temperature", "°C" + self._who(asset), c, worst)

    def _same_temp(self, pts: list[str]) -> bool:
        return len({self.q.vendor_settings(p).temp_warning_c for p in pts}) == 1

    def resid_chart(self, asset: str) -> str:
        c = self._new(asset)
        sc = self.q.scores
        lr = pd.read_parquet(self.q.cache / "level_rows.parquet")
        lr["hour_utc"] = pd.to_datetime(lr["hour_utc"], utc=True)
        seg = self.q.segments
        pts = self.q.points(asset)
        multi = len(pts) > 1
        for i, pid in enumerate(pts):
            name = self.q.point_label(pid) + " " if multi else ""
            g = sc[(sc["point_id"] == pid) & (sc["hour_utc"] >= self.t0) & (sc["hour_utc"] <= self.T)]
            s = seg[(seg["point_id"] == pid) & (seg["segment_start"] <= self.T)].sort_values("segment_start")
            for j, r in enumerate(s.itertuples()):
                final = j == len(s) - 1
                a = max(r.level_ready, self.t0) if pd.notna(r.level_ready) else None
                b = s["segment_start"].iloc[j + 1] if j + 1 < len(s) else self.T
                if a is None or min(b, self.T) <= a:
                    continue
                x = lr[
                    (lr["point_id"] == pid)
                    & (lr["hour_utc"] >= r.segment_start)
                    & (lr["hour_utc"] < r.level_ready)
                ]["temp_resid_c"].dropna()
                if len(x) >= 5:
                    lo, hi = float(x.quantile(0.05)), float(x.quantile(0.95))
                    c.band(
                        a,
                        min(b, self.T),
                        lo,
                        hi,
                        sv.POINT_COLORS[i],
                        self._lab(name, "level") if final else None,
                    )
                    c.fit(np.array([lo, hi]))
            v = g["temp_resid_c"].where(g["state"].isin(["scored", "plausibility"]))
            c.fit(v.to_numpy())
            c.series(g["hour_utc"], v, sv.POINT_COLORS[i])
        return self._chart("Temperature residual", "°C" + self._who(asset), c, "OK")

    def controller_charts(self, asset: str) -> list[str]:
        kind = asset_kind(asset)
        out = []
        mach = machine_of(asset)
        if kind == "spindle":
            d = self.q.t("controller_daily")
            d = d[d["machine_id"] == mach]
            out.append(
                self._daily(
                    asset, d, "local_date", "thermistor_residual_c", "Spindle thermistor residual", "°C"
                )
            )
        if kind in ("spindle", "tsc_pump"):
            d = self.q.t("tsc_pressure")
            d = d[d["machine_id"] == mach]
            out.append(
                self._daily(
                    asset, d, "local_date", "pressure_median_bar", "TSC coolant pressure, pump on", "bar"
                )
            )
        if kind == "compressor":
            d = self.q.t("discharge")
            d = d[d["compressor_id"] == asset]
            out.append(
                self._daily(
                    asset,
                    d,
                    "local_date",
                    "discharge_residual_c",
                    "Compressor discharge temperature residual",
                    "°C",
                )
            )
        return [x for x in out if x]

    def _daily(self, asset: str, d: pd.DataFrame, dcol: str, vcol: str, title: str, unit: str) -> str:
        day = pd.to_datetime(d[dcol])
        if day.dt.tz is not None:
            day = day.dt.tz_convert("UTC").dt.tz_localize(None)
        ts = (day.dt.normalize() + pd.Timedelta(hours=12)).dt.tz_localize(TZ).dt.tz_convert("UTC")
        d = d.assign(ts=ts.to_numpy())
        d = d[(d["ts"] >= self.t0) & (d["ts"] <= self.T)].sort_values("ts")
        if d[vcol].notna().sum() == 0:
            return ""
        c = self._new(asset)
        c.fit(d[vcol].to_numpy())
        c.series(d["ts"], d[vcol], sv.POINT_COLORS[0], gap_hours=72)
        return self._chart(title, unit, c, "OK")

    # ------------------------------------------------------------------ detail: bands and spectrum
    def _plan(self, asset: str) -> dict[str, dict[str, Any]]:
        p = pd.DataFrame(json.loads((self.q.cache / "burst_plan.json").read_text(encoding="utf-8")))
        p["hour_utc"] = pd.to_datetime(p["hour_utc"], utc=True)
        return {r["role"]: r for r in p[p["asset_id"] == asset].to_dict("records")}

    def band_section(self, asset: str) -> str:
        op = self.q.alert_opening(asset)
        if op is None:
            return ""
        pid = op["point_id"]
        ind = self.q.indicators
        g = ind[ind["point_id"] == pid]
        w = self.q.window_at(pid, self.T)
        L = self.q.vendor_limits(pid, self.T)
        groups = []
        for k, name, loc in BANDS:
            col = f"env_{k}_1_g2"
            base = w[col] if w is not None else np.nan
            since = g[(g["hour_utc"] >= op["opened"]) & (g["hour_utc"] <= self.T)][col].mean()
            week = g[(g["hour_utc"] > self.T - pd.Timedelta(days=7)) & (g["hour_utc"] <= self.T)][col].mean()
            if not (pd.notna(base) and base > 0 and pd.notna(since) and pd.notna(week)):
                continue
            lim = L.get(f"env_{k}_alert", np.nan)

            def col_of(v: float, lim: float = lim) -> str:
                return sv.AMBER if pd.notna(lim) and v >= lim else sv.GRAY

            groups.append(
                {
                    "name": f"{name} ({loc.lower()})",
                    "values": [
                        ("Healthy level", base, sv.REF),
                        ("Since alert opened", since, col_of(since)),
                        ("Past 7 days", week, col_of(week)),
                    ],
                    "limit": lim,
                }
            )
        if not groups:
            return ""
        self.facts["assets"][asset]["bands"] = [g["name"] for g in groups]
        return (
            f"<div class='sec'><div class='sech'>Fault-band energies <span class='unit'>{h(self.q.point_label(pid))} point, envelope, g² · columns: healthy level, average since the alert opened, past 7 days · dashed: Vendor Rules envelope limit</span></div>"
            f"<div class='chartbox narrow'>{sv.bars(groups, width=760, height=200)}</div></div>"
        )

    def spectrum_section(self, asset: str) -> str:
        plan = self._plan(asset)
        if not {"alert", "current", "reference"} <= set(plan):
            return ""
        rows = pd.read_parquet(self.q.cache / "burst_rows.parquet")
        rows["hour_utc"] = pd.to_datetime(rows["hour_utc"], utc=True)
        pid = plan["alert"]["point_id"]
        orders = self.q.fault_orders(pid)

        def spec(role: str) -> tuple[np.ndarray, np.ndarray, float, pd.Timestamp, str]:
            r = plan[role]
            hour = r["hour_utc"]
            arr, src = qb.load(pid, hour, r["retained"], self.q.cache)
            st = rows[(rows["point_id"] == pid) & (rows["hour_utc"] == hour)]
            speed = float(st["speed_rpm_at_measurement"].dropna().iloc[0]) / 60.0
            band = qb.band_at(self.q.sensor_events, pid, hour)
            f, a = qb.envelope_resultant(arr, self.xs.fs, band, self.xs)
            return f, a, speed, hour, src

        rf, ra, _, rh, _ = spec("reference")
        panels = []
        srcs = {}
        for role, title in (("alert", "At the alert"), ("current", "Current")):
            f, a, speed, hour, src = spec(role)
            srcs[role] = src
            fz = kinematics.fault_frequencies_hz(orders, speed)
            st = rows[(rows["point_id"] == pid) & (rows["hour_utc"] == hour) & (rows["axis"] == "res")]
            L = self.q.vendor_limits(pid, hour)
            markers = []
            for k, name, _ in BANDS:
                val = (
                    float(st[f"env_{k}_1_g2"].iloc[0])
                    if len(st) and pd.notna(st[f"env_{k}_1_g2"].iloc[0])
                    else np.nan
                )
                if np.isnan(val):
                    continue
                lim = L.get(f"env_{k}_alert", np.nan)
                col = sv.AMBER if pd.notna(lim) and val >= lim else sv.GRAY
                for hm in (1, 2, 3):
                    markers.append((hm * fz[k], name if hm == 1 else f"{hm}x{name}", col, hm == 1))
            band = qb.band_at(self.q.sensor_events, pid, hour)
            fmax = min(band[1] - band[0], 3.4 * max(fz["bpfo"], fz["bpfi"]))
            panels.append(
                f"<div class='chartbox half'><div class='ctitle'>{title} <span class='unit'>{local(hour, '%Y-%m-%d %H:%M')} · {speed * 60:,.0f} rpm · gray: healthy reference {local(rh, '%Y-%m-%d')}</span></div>"
                f"{sv.spectrum(f, a, rf, ra, fmax, markers)}</div>"
            )
        self.facts["assets"][asset]["bursts"] = srcs
        return (
            f"<div class='sec'><div class='sech'>Envelope spectrum <span class='unit'>{h(self.q.point_label(pid))} point, g</span></div>"
            f"<div class='row2'>{''.join(panels)}</div></div>"
        )

    # ------------------------------------------------------------------ detail: header and history
    def header(self, r: dict[str, Any]) -> str:
        a, v, ml, wr = r["asset"], r["v"], r["ml"], r["wr"]
        chips = []
        for name, lv, tag in (("Vendor Rules", v, ""), ("ML Model", ml, " <span class='tag'>Shadow</span>")):
            if lv is None:
                chips.append(f"<div class='hchip'><span class='hl'>{name}{tag}</span>{chip('OK')}</div>")
                continue
            extra = f"<span class='sub'>Opened {local(lv.opened)} · {days_open(lv.opened, self.T)} days open</span>"
            if lv.alarm_ts is not None:
                extra += (
                    f"<span class='sub'>ALARM {local(lv.alarm_ts)}, {h(cause_label(lv.alarm_cause))}</span>"
                )
            chips.append(
                f"<div class='hchip'><span class='hl'>{name}{tag}</span>{chip(lv.level)}{extra}</div>"
            )
        if wr is not None:
            cond = f"{h(family_label(wr['suspected_condition']))}" + (
                f" <span class='muted'>({h(wr['suspected_detail'])})</span>" if wr["suspected_detail"] else ""
            )
            insp = h(family_label(str(wr["recommended_inspection"])))
            req = f"<span id='wr-{h(a)}'>{h(request_number(wr['request_id']))}</span> <span class='tag'>Shadow</span> <span class='muted'>created {local(wr['created'], '%Y-%m-%d %H:%M')}</span>"
        else:
            cond, insp, req = (
                "<span class='muted'>Not diagnosed</span>",
                "<span class='muted'>Not specified</span>",
                "<span class='muted'>None</span>",
            )
        ow = self.q.open_work_orders(a)
        wos = (
            ", ".join(
                f"<a class='wo' href='#' data-asset='{h(a)}' data-wo='{h(x)}'>{h(x)}</a>" for x in ow["wo_id"]
            )
            or "<span class='muted'>None open</span>"
        )
        return (
            f"<div class='dhead'><div class='dtitle'>{h(a)} <span class='muted'>{h(self.q.asset_name(a))} · {h(self.q.location(a))}</span></div>"
            f"<div class='hchips'>{''.join(chips)}</div>"
            f"<table class='kv'><tr><td>Suspected condition</td><td>{cond}</td></tr><tr><td>Recommended inspection</td><td>{insp}</td></tr>"
            f"<tr><td>Work request</td><td>{req}</td></tr><tr><td>Open work orders</td><td>{wos}</td></tr></table></div>"
        )

    def history(self, asset: str) -> str:
        w = self.q.asset_work_orders(asset)
        w = w[w["opened"] > self.T - pd.Timedelta(days=365)].sort_values("opened", ascending=False)
        if w.empty:
            return ""
        rows = "".join(
            f"<tr id='wo-{h(asset)}-{h(r['wo_id'])}'><td class='mono'>{h(r['wo_id'])}</td><td>{local(r['opened'])}</td><td>{h(TYPE_LABEL.get(r['wo_type'], r['wo_type']))}</td>"
            f"<td class='mono'>{h(r['failure_code'] or '')}</td><td>{h(family_label(outcome(r)))}</td></tr>"
            for _, r in w.iterrows()
        )
        return (
            f"<div class='sec'><div class='sech'>Maintenance history <span class='unit'>12 months</span></div>"
            f"<table class='plain hist'><thead><tr><th>CMMS number</th><th>Opened</th><th>Type</th><th>Failure code</th><th>Outcome</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    def detail(self, r: dict[str, Any]) -> str:
        a = r["asset"]
        kind = asset_kind(a)
        trends = [
            self.score_chart(a),
            self.measure_chart(a, "vel_rms_mm_s", "Velocity RMS", "mm/s", "vel"),
            self.measure_chart(a, "hf_rms_g", "HF acceleration RMS", "g", "hf"),
            self.temp_chart(a),
            self.resid_chart(a),
        ]
        secs = [
            self.header(r),
            f"<div class='sec'><div class='sech'>Measurement trends <span class='unit'>90 days to {local(self.T, '%Y-%m-%d %H:%M')}</span></div>{''.join(trends)}</div>",
        ]
        try:
            spec = self.spectrum_section(a)
        except FileNotFoundError as e:
            self.facts["assets"][a]["missing_burst"] = str(e)
            spec = ""
        for s in (self.band_section(a), spec):
            if s:
                secs.append(s)
        ctrl = self.controller_charts(a)
        if ctrl:
            secs.append(
                f"<div class='sec'><div class='sech'>Controller signals <span class='unit'>daily</span></div>{''.join(ctrl)}</div>"
            )
        hist = self.history(a)
        if hist:
            secs.append(hist)
        self.facts["assets"][a]["sections"] = len(secs)
        self.facts["assets"][a]["charts"] = sum(x.count("class='chartbox") for x in secs)
        self.facts["assets"][a]["kind"] = kind
        return f"<div class='detail' id='d-{h(a)}' data-asset='{h(a)}' hidden>{''.join(secs)}</div>"

    # ------------------------------------------------------------------ page
    def pick(self, rows: list[dict[str, Any]]) -> str:
        """The asset for the screenshots: most sections; then both methods at ALERT or above; then D2 naming a
        bearing with a location (then a bearing); then the most charts; then queue order."""

        def key(r: dict[str, Any]) -> tuple:
            f = self.facts["assets"][r["asset"]]
            both = r["v"] is not None and r["ml"] is not None
            wr = r["wr"]
            located = (
                wr is not None
                and wr["suspected_condition"] == "bearing"
                and "unknown" not in str(wr["suspected_detail"])
            )
            bearing = wr is not None and wr["suspected_condition"] == "bearing"
            return (-f["sections"], not both, not located, not bearing, -f["charts"])

        return sorted(rows, key=key)[0]["asset"]

    def html(self) -> str:
        rows = self.rows()
        table = self.queue_table(rows)
        details = "".join(self.detail(r) for r in rows)
        ok = self.ok_list()
        c = self.q.counts_at(self.T)
        strip = (
            f"<div class='strip'><div class='sitem'><span class='sl'>Vendor Rules</span>{chip('ALARM')} <b>{c['vendor_rules ALARM']}</b> {chip('ALERT')} <b>{c['vendor_rules ALERT']}</b></div>"
            f"<div class='sitem'><span class='sl'>ML Model <span class='tag'>Shadow</span></span>{chip('ALARM')} <b>{c['ml_model ALARM']}</b> {chip('ALERT')} <b>{c['ml_model ALERT']}</b></div>"
            f"<div class='sitem'><span class='sl'>Assets in queue</span><b>{len(rows)}</b></div><div class='sitem'><span class='sl'>OK</span><b>{len(self.q.assets) - len(rows)}</b></div></div>"
        )
        default = self.pick(rows)
        self.facts.update(
            {
                "snapshot_utc": self.T.isoformat(),
                "snapshot_local": local(self.T, "%Y-%m-%d %H:%M %Z"),
                "choice": {
                    k: (int(v) if isinstance(v, (int, np.integer)) else str(v))
                    for k, v in self.q.snapshot_choice.items()
                },
                "counts": {k: int(v) for k, v in c.items()},
                "order": [r["asset"] for r in rows],
                "default": default,
            }
        )
        nav = "".join(
            f"<div class='nav{' active' if n == 'Maintenance Queue' else ''}'><span class='ico'>{i}</span>{n}</div>"
            for i, n in (
                ("▤", "Work Orders"),
                ("☰", "Maintenance Queue"),
                ("▣", "Assets"),
                ("◷", "PM Schedule"),
                ("▥", "Reports"),
            )
        )
        top = (
            f"<div class='top'><span class='prod'>{PRODUCT}</span><span class='site'>Plant 1</span><span class='grow'></span>"
            f"<span class='asof'>Data as of {local(self.T, '%Y-%m-%d %H:%M')} CT</span><span class='user'><span class='av'>MP</span>{USER}</span></div>"
        )
        bar = "<div class='bar'><div class='crumb'>Maintenance Queue</div><div class='search'><input type='text' placeholder='Search assets, work orders' readonly></div></div>"
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Maintenance Queue · {PRODUCT}</title><style>{CSS}</style></head><body>"
            "<div class='byline'>Created by Brian Davis</div>"
            f"<div class='frame'>{top}<div class='body'><div class='rail'>{nav}</div><div class='main'>{bar}{strip}{table}{ok}"
            f"<div id='details'>{details}</div></div></div></div><script>const DEFAULT={json.dumps(default)};{JS}</script></body></html>"
        )


CSS = """
:root{--ink:#2b3038;--muted:#69717c;--line:#d9dde2;--soft:#f4f5f7;--rail:#272d36;--amber:#c77700;--amberbg:#fdf1dc;--red:#c8322b;--redbg:#fbe4e2;--gray:#6d7681;--graybg:#eceef1;}
*{box-sizing:border-box}html,body{margin:0;background:#e9ebee;color:var(--ink);font:12px/1.35 "Segoe UI",-apple-system,Roboto,Helvetica,Arial,sans-serif}
.byline{max-width:1440px;margin:0 auto;padding:6px 12px 4px;font-size:11px;color:var(--muted)}
.frame{max-width:1440px;margin:0 auto 24px;background:#fff;border:1px solid var(--line)}
.top{display:flex;align-items:center;gap:14px;height:34px;padding:0 12px;border-bottom:1px solid var(--line);background:#fff}
.prod{font-weight:700;font-size:13px;letter-spacing:.2px}.site{color:var(--muted);border-left:1px solid var(--line);padding-left:12px}.grow{flex:1}
.asof{color:var(--muted)}.user{display:flex;align-items:center;gap:6px;color:var(--ink)}.av{display:inline-flex;width:20px;height:20px;border-radius:50%;background:#59616c;color:#fff;font-size:9px;align-items:center;justify-content:center}
.body{display:flex;min-height:900px}.rail{width:136px;flex:none;background:var(--rail);padding-top:6px}
.nav{display:flex;gap:8px;align-items:center;color:#c3c9d1;padding:7px 12px;font-size:12px;border-left:3px solid transparent;cursor:default}
.nav.active{background:#343b46;color:#fff;border-left-color:#c3c9d1;font-weight:600}.ico{width:14px;text-align:center;opacity:.8}
.main{flex:1;min-width:0;padding:8px 12px 16px}
.bar{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}.crumb{font-weight:600;font-size:13px}
.search input{width:260px;font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:2px;background:#fff;color:var(--muted)}
.strip{display:flex;gap:0;border:1px solid var(--line);margin-bottom:8px;background:var(--soft)}
.sitem{display:flex;align-items:center;gap:6px;padding:5px 12px;border-right:1px solid var(--line)}.sitem b{margin-right:6px}.sl{color:var(--muted);margin-right:4px}
.chip{display:inline-block;min-width:44px;text-align:center;padding:0 5px;border-radius:2px;font-size:10.5px;font-weight:700;letter-spacing:.3px;line-height:16px;border:1px solid}
.chip.red{color:var(--red);background:var(--redbg);border-color:#efb5b0}.chip.amber{color:var(--amber);background:var(--amberbg);border-color:#efcf95}.chip.gray{color:var(--gray);background:var(--graybg);border-color:#d6dae0}
.tag{display:inline-block;padding:0 4px;border:1px solid var(--line);border-radius:2px;font-size:10px;color:var(--muted);line-height:14px;font-weight:400}
.tag.only{color:var(--ink)}
table{border-collapse:collapse;width:100%}
.queue th{text-align:left;font-weight:600;font-size:11px;color:var(--muted);background:var(--soft);border:1px solid var(--line);padding:4px 6px;vertical-align:top}
.queue th .sub{font-weight:400}
.queue tr.filters th{background:#fff;padding:3px 6px}.fbox{border:1px solid var(--line);padding:1px 5px;color:var(--muted);font-weight:400;display:flex;justify-content:space-between}
.queue td{border:1px solid var(--line);padding:4px 6px;vertical-align:top}
.qrow{cursor:pointer}.qrow:hover td{background:#f8f9fa}.qrow.sel td{background:#eef2f7}.qrow.sel td:first-child{box-shadow:inset 3px 0 0 #59616c}
.aname{font-weight:600}.sub{color:var(--muted);font-size:11px}.muted{color:var(--muted)}.note{font-size:11px;color:var(--ink);margin-top:2px;padding-left:6px;border-left:2px solid #c9ced5}
a.wo{color:#2f5f9e;text-decoration:none}a.wo:hover{text-decoration:underline}
.oklist{margin:8px 0;border:1px solid var(--line)}.oklist summary{padding:5px 8px;cursor:pointer;background:var(--soft);font-weight:600}.count{color:var(--muted);font-weight:400;margin-left:6px}
table.plain td,table.plain th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}table.plain th{font-weight:600;color:var(--muted);font-size:11px;background:var(--soft)}
.detail{border:1px solid var(--line);margin-top:10px}
.dhead{padding:8px 10px;border-bottom:1px solid var(--line);background:#fbfbfc}.dtitle{font-size:14px;font-weight:700;margin-bottom:6px}
.hchips{display:flex;gap:18px;margin-bottom:6px}.hchip{display:flex;align-items:center;gap:6px}.hchip .sub{margin-left:4px}.hl{color:var(--muted)}
table.kv{width:auto}table.kv td{padding:2px 12px 2px 0;vertical-align:top}table.kv td:first-child{color:var(--muted);white-space:nowrap}
.sec{padding:6px 10px 8px;border-bottom:1px solid var(--line)}.sec:last-child{border-bottom:0}.sech{font-weight:600;margin-bottom:4px}.unit{font-weight:400;color:var(--muted);font-size:11px}
.chartbox{border:1px solid #eceef1;margin:4px 0 6px;padding:3px 4px 0}.ctitle{font-weight:600;font-size:11.5px;margin:2px 2px 0;display:flex;gap:6px;align-items:center}
.chartbox.narrow{max-width:780px}.row2{display:flex;gap:8px}.chartbox.half{flex:1;min-width:0}
svg.chart{display:block}svg .tick{font-size:9.5px;fill:#7a828d}svg .lbl{font-size:10px;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}
.mono{font-family:Consolas,"SFMono-Regular",Menlo,monospace;font-size:11px}.hist tr.flash td{background:#fff6dd}
"""

JS = """
const rows=[...document.querySelectorAll('.qrow')];
function openAsset(a,scroll){rows.forEach(r=>r.classList.toggle('sel',r.dataset.asset===a));
document.querySelectorAll('.detail').forEach(d=>d.hidden=d.dataset.asset!==a);
if(scroll){document.getElementById('d-'+a).scrollIntoView({block:'start'});}}
rows.forEach(r=>r.addEventListener('click',e=>{if(e.target.closest('a'))return;openAsset(r.dataset.asset,true);}));
document.querySelectorAll('a.wo').forEach(l=>l.addEventListener('click',e=>{e.preventDefault();e.stopPropagation();const a=l.dataset.asset;openAsset(a,false);
const t=l.dataset.wo?document.getElementById('wo-'+a+'-'+l.dataset.wo):document.getElementById('wr-'+a);
if(t){t.scrollIntoView({block:'center'});const tr=t.closest('tr');if(tr){tr.classList.add('flash');setTimeout(()=>tr.classList.remove('flash'),1500);}}}));
openAsset(DEFAULT,false);
"""


def build() -> dict[str, Any]:
    p = Page()
    page = p.html()
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (p.q.cache / "page.json").write_text(json.dumps(p.facts, indent=1, default=str), encoding="utf-8")
    LOG.info(
        "queue page built: snapshot %s, %d assets in queue, default %s",
        p.facts["snapshot_local"],
        len(p.facts["order"]),
        p.facts["default"],
    )
    return p.facts


if __name__ == "__main__":
    log.configure()
    build()

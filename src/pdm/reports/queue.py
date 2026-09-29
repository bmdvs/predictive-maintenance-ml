"""The maintenance queue page (docs/index.html): one row per monitored asset as of the record end and, below
the queue, one detail panel per asset, shown one at a time.

Inputs: the Data object (gold snapshot) and the page text, section A. Output: docs/index.html and its
figures under docs/figures.
"""

from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd

from pdm.advisory import requests as adv
from pdm.features import envelope, filters, kinematics
from pdm.reports import style
from pdm.reports.data import END, LAYER_LABEL, TZ, Data, local, naive_date, utc
from pdm.reports.text import PageText

CLASS = {"SP": "HMC spindle", "TSC": "TSC pump", "CP": "Flood coolant pump"}
MACHINE_CLASS = {"hmc": "HMC", "vmc": "VMC", "lathe": "Lathe", "mill_turn": "Mill-turn", "grinder": "Grinder"}
STATUS_ORDER = {"ALARM": 0, "WATCH": 1, "OK": 2}
TREND = [
    ("env_bpfi", "Envelope BPFI"),
    ("env_bpfo", "Envelope BPFO"),
    ("env_bsf", "Envelope BSF"),
    ("hf", "HF acceleration"),
    ("vel", "Velocity"),
    ("ord_01", "1x"),
]
DEFAULT = "CMP-01"


def _assets(d: Data) -> list[dict]:
    m = d.machines
    hmc = sorted(m.loc[m["class"] == "hmc", "machine_id"])
    out = []
    for mch in hmc:
        for suffix in ("SP", "TSC", "CP"):
            out.append(
                {"asset": f"{mch}-{suffix}", "class": CLASS[suffix], "machine": mch, "kind": "vibration"}
            )
        out.append({"asset": mch, "class": "HMC (controller)", "machine": mch, "kind": "controller"})
    for c in ("CMP-01", "CMP-02"):
        out.append({"asset": c, "class": "Compressor", "machine": c, "kind": "compressor"})
    other = m[m["class"] != "hmc"].sort_values("machine_id")
    for mid, cls in zip(other["machine_id"], other["class"], strict=True):
        out.append(
            {
                "asset": mid,
                "class": f"{MACHINE_CLASS[cls]} (controller)",
                "machine": mid,
                "kind": "controller",
            }
        )
    return out


def _points(d: Data, asset: str) -> list[str]:
    s = d.sensors
    return sorted(s.loc[s["asset_id"] == asset, "point_id"])


def _layers(s: str) -> str:
    names = [LAYER_LABEL.get(x.strip(), x.strip()) for x in str(s or "").split(",") if x.strip()]
    return ", ".join(dict.fromkeys(names))


class Queue:
    def __init__(self, d: Data, t: PageText) -> None:
        self.d, self.t = d, t
        self.assets = _assets(d)
        self.req_by_case = d.requests.set_index("case_id")
        stats = utc_cols(d.g("point_healthy_stats"), ["segment_start"])
        ind = d.silver_2026
        self.hz = adv.daily_point_z(adv.hourly_z(ind, stats))
        self.hz["day"] = pd.to_datetime(self.hz["day"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
        self.log = adv.log_indicators(ind).assign(
            point_id=ind["point_id"].to_numpy(),
            hour_utc=ind["hour_utc"].to_numpy(),
            regime=ind["regime"].to_numpy(),
        )
        self.stats = stats
        ep = d.episodes
        self.model_eps = ep[ep["method"] == "nbm"]
        ra = d.g("rule_floor_alerts")
        ra["alert_ts"] = utc(ra["alert_ts"])
        self.vendor_alerts = ra[ra["method"] == "vendor"]
        self.bursts = pd.read_parquet(d.st.file("bursts/panel_bursts.parquet"))
        self.bursts["hour_utc"] = utc(self.bursts["hour_utc"])
        self.ctrl = d.g("controller_health_daily")
        self.ctrl["local_date"] = naive_date(self.ctrl["local_date"])
        self.disc = d.g("compressor_discharge_daily")
        self.disc["local_date"] = naive_date(self.disc["local_date"])
        self.start = (END - pd.Timedelta(days=90)).tz_convert(TZ).tz_localize(None).normalize()
        self.stop = END.tz_convert(TZ).tz_localize(None)

    # ---------------------------------------------------------------- queue
    def rows(self) -> pd.DataFrame:
        d, rows = self.d, []
        status_label = self.t.field("A", "Request status label")
        for a in self.assets:
            st = d.status_at_end.get(a["asset"], "OK")
            c = d.open_cases.loc[a["asset"]] if a["asset"] in d.open_cases.index else None
            w = (
                self.req_by_case.loc[c["case_id"]]
                if c is not None and c["case_id"] in self.req_by_case.index
                else None
            )
            sus = ""
            if w is not None:
                sus = w["suspected_condition"] + (
                    f", {w['suspected_detail']}" if w["suspected_detail"] else ""
                )
            rows.append(
                {
                    "Asset": a["asset"],
                    "Class": a["class"],
                    "Status": st,
                    "Case opened": local(c["opened"]) if c is not None else "",
                    "Days open": round((END - c["opened"]) / pd.Timedelta(days=1), 1)
                    if c is not None
                    else np.nan,
                    "Suspected condition": sus,
                    "Points": ", ".join(p for p in str(c["trigger_points"] or "").split(",") if p)
                    if c is not None
                    else "",
                    "Triggering layer": _layers(c["layers"]) if c is not None else "",
                    "Request id": w["request_id"] if w is not None else "",
                    "Request status": status_label if w is not None else "",
                }
            )
        q = pd.DataFrame(rows)
        q["_o"] = q["Status"].map(STATUS_ORDER)
        q["_d"] = -q["Days open"].fillna(-1)
        return q.sort_values(["_o", "_d", "Asset"]).drop(columns=["_o", "_d"]).reset_index(drop=True)

    # ---------------------------------------------------------------- figures
    def trend(self, asset: str, points: list[str], case: pd.Series | None) -> str:
        f, axes = style.fig(h=1.55 * len(TREND) + 0.4, w=8.6, nrows=len(TREND), sharex=True)
        colors = [style.BRAND_BLUE, style.AMBER, style.GREEN]
        for ax, (k, lab) in zip(axes, TREND, strict=True):
            ax.axhspan(
                -self.d.cfg["evaluation"]["advisory_z"],
                self.d.cfg["evaluation"]["advisory_z"],
                color=style.LIGHT_BLUE,
                alpha=0.25,
                lw=0,
            )
            for p, colr in zip(points, colors, strict=False):
                g = self.hz[
                    (self.hz["point_id"] == p) & (self.hz["day"] >= self.start) & (self.hz["day"] < self.stop)
                ]
                ax.plot(g["day"], g[k], color=colr, lw=1.1, label=p)
            ax.set_ylabel(f"{lab}, z", fontsize=8)
        top = axes[0]
        for p in points:
            for t in self.vendor_alerts.loc[
                (self.vendor_alerts["point_id"] == p)
                & (self.vendor_alerts["alert_ts"] >= END - pd.Timedelta(days=90))
                & (self.vendor_alerts["alert_ts"] < END),
                "alert_ts",
            ]:
                top.axvline(t.tz_convert(TZ).tz_localize(None), color=style.GREY, lw=0.8, alpha=0.8)
            for t in self.model_eps.loc[
                (self.model_eps["point_id"] == p)
                & (self.model_eps["opened"] >= END - pd.Timedelta(days=90))
                & (self.model_eps["opened"] < END),
                "opened",
            ]:
                for ax in axes:
                    ax.axvline(t.tz_convert(TZ).tz_localize(None), color=style.GREEN, lw=1.0, ls=":")
        w = self.d.wo
        for t in w.loc[
            w["asset_id"].str.startswith(asset)
            & (w["wo_type"] != "pm")
            & (w["closed"] >= END - pd.Timedelta(days=90))
            & (w["closed"] < END),
            "closed",
        ]:
            for ax in axes:
                ax.axvline(t.tz_convert(TZ).tz_localize(None), color="#000000", lw=0.9, ls="--")
        if case is not None:
            for ax in axes:
                ax.axvline(case["opened"].tz_convert(TZ).tz_localize(None), color=style.RED, lw=1.2)
        handles = [plt_line(c, label=p) for p, c in zip(points, colors, strict=False)]
        handles += [
            plt_line(style.GREY, label="Vendor alert"),
            plt_line(style.GREEN, ":", label="Model alert opened"),
            plt_line("#000000", "--", label="Work order closed"),
        ]
        if case is not None:
            handles.append(plt_line(style.RED, label="Case opened"))
        f.autofmt_xdate()
        f.tight_layout(rect=(0, 0.06, 1, 1))
        f.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
        return style.save(f, f"queue_trend_{asset}", f"{asset} trend")

    def controller(self, a: dict) -> str | None:
        mch = a["machine"]
        if a["kind"] == "compressor":
            g = self.disc[
                (self.disc["compressor_id"] == mch)
                & (self.disc["local_date"] >= self.start)
                & (self.disc["local_date"] < self.stop)
            ]
            f, ax = style.fig(h=2.2, w=8.6)
            ax.plot(g["local_date"], g["discharge_residual_c"], color=style.AMBER, lw=1.1)
            ax.set_ylabel("Discharge residual, C", fontsize=8)
            f.autofmt_xdate()
            return style.save(f, f"queue_ctrl_{a['asset']}", f"{a['asset']} compressor controller")
        g = self.ctrl[
            (self.ctrl["machine_id"] == mch)
            & (self.ctrl["local_date"] >= self.start)
            & (self.ctrl["local_date"] < self.stop)
        ]
        hmc = mch.startswith("HMC-")
        n = 4 if hmc else 3
        f, axes = style.fig(h=1.5 * n + 0.3, w=8.6, nrows=n, sharex=True)
        axes[0].plot(g["local_date"], 100 * g["spindle_load_drift"], color=style.AMBER, lw=1.1)
        axes[0].set_ylabel("Spindle load\ndrift, %", fontsize=8)
        axes[1].plot(g["local_date"], 100 * g["axis_drift_max"], color=style.AMBER, lw=1.1)
        axes[1].set_ylabel("Axis load\ndrift, %", fontsize=8)
        axes[2].plot(g["local_date"], g["thermistor_residual_c"], color=style.AMBER, lw=1.1)
        axes[2].set_ylabel("Thermistor\nresidual, C", fontsize=8)
        if hmc:
            p = self.d.pressure
            p = p[(p["machine_id"] == mch) & (p["local_date"] >= self.start) & (p["local_date"] < self.stop)]
            axes[3].plot(p["local_date"], p["pump_on_bar"], color=style.BRAND_BLUE, lw=1.1)
            axes[3].axhline(69.0, color=style.DARK_GREY, lw=0.8, ls="--")
            axes[3].set_ylabel("TSC pressure,\nbar", fontsize=8)
        f.autofmt_xdate()
        return style.save(f, f"queue_ctrl_{a['asset']}", f"{a['asset']} controller")

    def spectra(self, asset: str, case: pd.Series) -> tuple[str | None, list[tuple[str, float]]]:
        b = self.bursts[self.bursts["case_id"] == case["case_id"]]
        if b.empty:
            return None, []
        pts = sorted(b["point_id"].unique())
        brg = self.d.st.read_pd("bronze", "reg_bearings").set_index("part")
        sens = self.d.sensors.set_index("point_id")
        f, axes = style.fig(h=5.2, w=8.6, nrows=2, ncols=len(pts), squeeze=False)
        colors = {
            "baseline reference": style.GREY,
            "first after first alarm": style.AMBER,
            "latest before the cut": style.RED,
        }
        offsets = []
        for j, p in enumerate(pts):
            geo = brg.loc[sens.loc[p, "bearing"]]
            orders = kinematics.fault_orders(
                int(geo["n"]), float(geo["d_mm"]), float(geo["pitch_mm"]), float(geo["contact_deg"])
            )
            axis = 1 if str(sens.loc[p, "radial_axis"]) == "y" else 0
            g = b[b["point_id"] == p]
            last_speed = None
            for r in g.itertuples():
                z = np.load(self.d.st.file(r.file))
                x = filters.remove_dc(np.asarray(z["accel_g"], dtype=float))
                spec, freqs = filters.rfft_with_freqs(x, float(z["fs_hz"]))
                n = x.shape[-1]
                amp = 2 * np.abs(spec[axis]) / n
                ef, ea = envelope.envelope_spectrum(
                    spec[axis], freqs, n, float(z["fs_hz"]), r.band_lo_hz, r.band_hi_hz
                )
                lab = f"{r.role}, {local(r.hour_utc, '%Y-%m-%d %H:%M')}"
                axes[0, j].semilogy(freqs, amp, color=colors[r.role], lw=0.5, label=lab)
                axes[1, j].plot(ef, ea, color=colors[r.role], lw=0.7, label=lab)
                if r.role == "first after first alarm":
                    offsets.append((p, float(r.offset_from_first_alarm_days)))
                last_speed = r.speed_rpm
            ff = kinematics.fault_frequencies_hz(orders, last_speed / 60.0)
            for k, ls in (("bpfo", ":"), ("bpfi", "--"), ("bsf", "-."), ("ftf", (0, (1, 3)))):
                for h in (1, 2, 3):
                    axes[1, j].axvline(ff[k] * h, color=style.BRAND_BLUE, ls=ls, lw=0.7, alpha=0.7)
            axes[0, j].set_xlim(0, 10000)
            axes[0, j].set_ylim(1e-5, 1)
            axes[0, j].set_title(f"{p}, spectrum", fontsize=9)
            axes[1, j].set_xlim(0, ff["bpfi"] * 3.4)
            axes[1, j].set_title(f"{p}, envelope spectrum", fontsize=9)
            axes[1, j].set_xlabel("Hz", fontsize=8)
            axes[0, j].set_ylabel("g", fontsize=8)
        h, lab = axes[0, 0].get_legend_handles_labels()
        h += [
            plt_line(style.BRAND_BLUE, ":", label="BPFO"),
            plt_line(style.BRAND_BLUE, "--", label="BPFI"),
            plt_line(style.BRAND_BLUE, "-.", label="BSF"),
        ]
        f.tight_layout(rect=(0, 0.1, 1, 1))
        f.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8)
        return style.save(f, f"queue_spectra_{asset}", f"{asset} spectral evidence"), offsets

    # ---------------------------------------------------------------- panel parts
    def why(self, asset: str, points: list[str], case: pd.Series, w: pd.Series | None) -> str:
        zthr = self.d.cfg["evaluation"]["advisory_z"]
        rows = []
        for p in points:
            g = self.hz[(self.hz["point_id"] == p) & (self.hz["day"] < self.stop)].sort_values("day")
            last = g[g["day"] >= self.stop - pd.Timedelta(days=7)]
            for k, lab in TREND:
                if last.empty or g[k].isna().all():
                    continue
                znow = float(last[k].median())
                if znow < zthr:
                    continue
                above = (g[k] >= zthr).to_numpy()
                streak = 0
                for v in above[::-1]:
                    if not v:
                        break
                    streak += 1
                ratio = self._ratio(p, k)
                rows.append(
                    {
                        "Point": p,
                        "Indicator": lab,
                        "Against baseline, last 7 days": f"{ratio:.1f} times" if ratio else "",
                        "z, last 7 days": znow,
                        "Days above z 3": streak,
                    }
                )
        head = f"<p><b>Triggering layer:</b> {html.escape(_layers(case['layers']))}</p>"
        if not rows:
            return head
        return head + style.table(pd.DataFrame(rows), {"z, last 7 days": 1})

    def _ratio(self, p: str, k: str) -> float | None:
        lg = self.log[
            (self.log["point_id"] == p)
            & (pd.to_datetime(self.log["hour_utc"], utc=True) >= END - pd.Timedelta(days=7))
        ]
        st = self.stats[(self.stats["point_id"] == p) & (self.stats["indicator"] == k)]
        if lg.empty or st.empty:
            return None
        seg = st[st["segment_start"] == st["segment_start"].max()]
        med = seg.set_index("regime")["median"]
        v = lg[k] - lg["regime"].map(med)
        v = v.dropna()
        return float(10 ** v.median()) if len(v) else None

    def history(self, asset: str) -> str:
        w = self.d.wo
        h = w[
            w["asset_id"].str.startswith(asset)
            & (w["wo_type"] != "pm")
            & (w["wo_type"] != "project")
            & (w["closed"] < END)
        ].sort_values("opened", ascending=False)
        wt = pd.DataFrame(
            {
                "Work order": h["wo_id"],
                "Type": h["wo_type"].str.replace("_", " "),
                "Finding code": h["failure_code"].fillna(""),
                "Opened": h["opened"].map(local),
                "Closed": h["closed"].map(local),
                "Downtime, h": h["downtime_hours"].astype(float),
            }
        )
        c = self.d.cases[self.d.cases["asset_id"] == asset].sort_values("opened", ascending=False)
        ct = (
            pd.DataFrame()
            if c.empty
            else pd.DataFrame(
                {
                    "Case": c["case_id"],
                    "Opened": c["opened"].map(local),
                    "Closed": c["closed"].map(lambda x: local(x) if x < END else ""),
                    "Closed by": c.apply(
                        lambda r: (r["closing_wo"] or "") if r["closed"] < END else "", axis=1
                    ),
                    "Triggering layer": c["layers"].map(_layers),
                }
            )
        )
        out = ""
        if len(ct):
            out += "<h4>Cases</h4>" + style.table(ct)
        if len(wt):
            out += "<h4>Work orders</h4>" + style.table(wt, {"Downtime, h": 1})
        return out

    def last_work(self, asset: str) -> str:
        w = self.d.wo
        h = w[
            w["asset_id"].str.startswith(asset) & (w["closed"] < END) & (w["wo_type"] != "project")
        ].sort_values("closed")
        if h.empty:
            return ""
        r = h.iloc[-1]
        kind = "PM" if r["wo_type"] == "pm" else r["wo_type"].replace("_", " ")
        return f"{r['wo_id']} ({kind}{', ' + r['failure_code'] if r['failure_code'] else ''}), {local(r['closed'])}"

    def panel(self, a: dict, q: pd.Series) -> str:
        d, t = self.d, self.t
        asset = a["asset"]
        labels = t.items("A", "Detail panel section labels")
        case = d.open_cases.loc[asset] if asset in d.open_cases.index else None
        w = (
            self.req_by_case.loc[case["case_id"]]
            if case is not None and case["case_id"] in self.req_by_case.index
            else None
        )
        points = _points(d, asset)
        summary = [("Asset", f"{asset}, {a['class']}"), ("Status", q["Status"])]
        if case is not None:
            summary += [
                ("Case opened", f"{local(case['opened'])}, {q['Days open']:.1f} days open"),
                ("Suspected condition", q["Suspected condition"]),
                ("Points", q["Points"]),
                ("Request", f"{q['Request id']}, {q['Request status']}"),
            ]
        summary.append(("Last repair or PM", self.last_work(asset)))
        dl = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in summary)
        parts = [
            f"<h3>{html.escape(asset)}</h3><h4>{html.escape(labels[0])}</h4><dl class='summary'>{dl}</dl>"
        ]
        fixed = fixed_lines(t)
        if asset == "HMC-03-SP":
            parts.append(f"<p class='fixed'>{html.escape(fixed['HMC-03-SP'])}</p>")
        if case is None:
            parts.append(f"<p class='fixed'>{html.escape(fixed['Asset with no open case'])}</p>")
        if case is not None and points:
            parts.append(f"<h4>{html.escape(labels[1])}</h4>" + self.why(asset, points, case, w))
        if points:
            parts.append(f"<h4>{html.escape(labels[2])}</h4>" + self.trend(asset, points, case))
            if case is not None:
                img, offsets = self.spectra(asset, case)
                if img:
                    parts.append(f"<h4>{html.escape(labels[3])}</h4>")
                    line = fixed["Spectral evidence, where no burst exists at the first alarm"]
                    for p, off in offsets:
                        if abs(off) > 1 / 24:
                            parts.append(
                                f"<p class='fixed'>{html.escape(p)}: {html.escape(line.replace('{offset}', f'{off:.1f}'))}</p>"
                            )
                    parts.append(img)
        img = self.controller(a) if (a["kind"] != "vibration" or asset.startswith("HMC-")) else None
        if img:
            title = labels[2] if a["kind"] == "controller" else labels[4]
            note = (
                f"<p class='fixed'>{html.escape(fixed['Coolant pressure'])}</p>"
                if a["machine"].startswith("HMC-")
                else ""
            )
            parts.append(f"<h4>{html.escape(title)}</h4>{note}{img}")
        parts.append(
            f"<h4>{html.escape(labels[5])}</h4><p class='fixed'>{html.escape(fixed['Remaining time, where no estimate exists'])}</p>"
        )
        parts.append(f"<h4>{html.escape(labels[6])}</h4>" + self.history(asset))
        return f"<section class='panel' id='panel-{html.escape(asset)}' data-asset='{html.escape(asset)}'>{''.join(parts)}</section>"

    # ---------------------------------------------------------------- page
    def build(self) -> str:
        t = self.t
        q = self.rows()
        attrs = [
            f'data-asset="{html.escape(a)}" class="row status-{s.lower()}"'
            for a, s in zip(q["Asset"], q["Status"], strict=True)
        ]
        table = style.table(q, {"Days open": 1}, cls="data queue", row_attrs=attrs)
        legend = "".join(f"<li>{html.escape(x)}</li>" for x in t.items("A", "Status legend"))
        qi = q.set_index("Asset")
        panels = "".join(self.panel(a, qi.loc[a["asset"]]) for a in self.assets)
        title = t.field("A", "Title")
        body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{style.CSS}{QUEUE_CSS}</style></head>
<body><div class="page-header"><h1>{html.escape(title)}</h1><div class="sub">{style.CREDIT}</div></div>
<main class="queue-page">
<p class="lead">{html.escape(t.field("A", "Line under the byline"))}</p>
<p class="qhead"><b>{html.escape(t.field("A", "Queue header"))}</b></p>
<div class="legend"><ul>{legend}</ul><p>{html.escape(t.field("A", "Legend note"))}</p></div>
<div id="queue">{table}</div>
<div id="detail">{panels}</div>
</main>
<script>
const DEFAULT = {json.dumps(DEFAULT)};
function show(a, scroll) {{
  document.querySelectorAll('.panel').forEach(p => p.style.display = p.dataset.asset === a ? 'block' : 'none');
  document.querySelectorAll('tr.row').forEach(r => r.classList.toggle('selected', r.dataset.asset === a));
  if (scroll) document.getElementById('panel-' + a).scrollIntoView();
}}
document.querySelectorAll('tr.row').forEach(r => r.addEventListener('click', () => show(r.dataset.asset, true)));
const h = decodeURIComponent(location.hash.replace('#', ''));
if (new URLSearchParams(location.search).get('capture') === 'detail') document.body.classList.add('capture-detail');
if (h.startsWith('panel-')) show(h.slice(6), true); else show(DEFAULT, false);
</script>
</body></html>"""
        style.write("index.html", body)
        return "docs/index.html"


QUEUE_CSS = """
body { background: white; }
.page-header { background: var(--brand); color: white; padding: 14px 40px; display: flex; justify-content: space-between; align-items: center; }
.page-header h1 { margin: 0; font-size: 21px; color: white; }
.page-header .sub { font-size: 12px; }
.queue-page { max-width: 1360px; margin: 0 auto; padding: 18px 40px 60px; }
.lead { color: var(--muted); margin: 4px 0 6px; }
.qhead { margin: 6px 0 4px; }
.legend { font-size: 13px; color: var(--muted); }
.legend ul { display: flex; gap: 22px; list-style: none; padding: 0; margin: 4px 0; }
.legend p { margin: 2px 0 10px; }
table.queue { font-size: 12.5px; }
table.queue tr.row { cursor: pointer; }
table.queue td:first-child, table.queue td:nth-child(9) { white-space: nowrap; }
body.capture-detail #queue, body.capture-detail .lead, body.capture-detail .qhead, body.capture-detail .legend { display: none; }
body.capture-detail .panel { border-top: 0; margin-top: 0; }
table.queue tr.row:hover td { background: #F3F6F9; }
table.queue tr.selected td { background: #E6EEF5; }
table.queue tr.status-alarm td:nth-child(3) { color: #CC0000; font-weight: 700; }
table.queue tr.status-watch td:nth-child(3) { color: #D4881E; font-weight: 700; }
.panel { display: none; border-top: 3px solid var(--brand); margin-top: 26px; padding-top: 4px; }
.panel h3 { font-size: 19px; margin: 10px 0 4px; }
.panel h4 { border-bottom: 1px solid var(--rule); padding-bottom: 3px; }
.panel img { max-width: 900px; }
dl.summary { display: grid; grid-template-columns: 170px 1fr; gap: 3px 14px; font-size: 13.5px; margin: 6px 0; }
dl.summary dt { color: var(--muted); }
dl.summary dd { margin: 0; }
p.fixed { color: var(--muted); font-size: 13.5px; }
"""


def fixed_lines(t: PageText) -> dict[str, str]:
    """The fixed panel lines by label: the quoted text after each bold label."""
    out = {}
    for x in t.items("A", "Fixed panel lines"):
        m = re.match(r"\*\*(.+?):\*\*\s*\"(.+?)\"", x)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def utc_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = utc(df[c])
    return df


def plt_line(color: str, ls: str = "-", label: str = ""):
    from matplotlib.lines import Line2D

    return Line2D([0], [0], color=color, ls=ls, lw=1.2, label=label)


def build(d: Data, t: PageText) -> str:
    return Queue(d, t).build()

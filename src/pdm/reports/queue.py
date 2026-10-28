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
TREND_UNITS = [
    ("env_bpfi", "env_bpfi_1_g2", "Envelope BPFI, g²", True),
    ("env_bpfo", "env_bpfo_1_g2", "Envelope BPFO, g²", True),
    ("hf", "hf_rms_g", "HF acceleration, g", False),
    ("vel", "vel_rms_mm_s", "Velocity, mm/s", False),
]
BASELINE_DAYS = 60
WHOLE_BAND = "Whole envelope band"
FAULTS = [("BPFO", "bpfo", ":"), ("BPFI", "bpfi", "--"), ("BSF", "bsf", "-."), ("FTF", "ftf", (0, (1, 3)))]
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


def request_id(s: str) -> str:
    """The advisory request id as shown: asset and creation hour joined by a hyphen."""
    return str(s).replace("|", "-")


def _days(days: float) -> str:
    if np.isnan(days):
        return ""
    return "<1" if days < 1 else str(int(days))


def _suspected(w: pd.Series) -> str:
    """Condition and location, capitalized, without the z-score."""
    cond = str(w["suspected_condition"])
    where = re.sub(r"\s*\(.*\)$", "", str(w["suspected_detail"] or "")).strip()
    loc = "" if (not where or " z " in f" {where} ") else where
    return cond[:1].upper() + cond[1:] + (f", {loc}" if loc else "")


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
        self.no_baseline: list[str] = []
        ref = d.st.read_pd("silver", "hourly_indicators_trend_reference")
        ref["hour_utc"] = utc(ref["hour_utc"])
        ref = ref[~ref["implausible"].fillna(False).astype(bool)]
        cols = ["point_id", "hour_utc"] + [c for _, c, _, _ in TREND_UNITS]
        raw = pd.concat([ref[cols], ind[cols]], ignore_index=True)
        raw["day"] = raw["hour_utc"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
        self.daily = raw.groupby(["point_id", "day"])[cols[2:]].median().reset_index()
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
            days = (END - c["opened"]) / pd.Timedelta(days=1) if c is not None else np.nan
            rows.append(
                {
                    "Asset": a["asset"],
                    "Class": a["class"],
                    "Status": st,
                    "Case opened": local(c["opened"]) if c is not None else "",
                    "Days open": _days(days),
                    "Suspected condition": _suspected(w) if w is not None else "",
                    "Points": ", ".join(p for p in str(c["trigger_points"] or "").split(",") if p)
                    if c is not None
                    else "",
                    "Triggering layer": _layers(c["layers"]) if c is not None else "",
                    "Request id": request_id(w["request_id"]) if w is not None else "",
                    "Request status": status_label.split(":")[0] if w is not None else "",
                    "_days": days,
                }
            )
        q = pd.DataFrame(rows)
        q["_o"] = q["Status"].map(STATUS_ORDER)
        q["_d"] = -q["_days"].fillna(-1)
        return q.sort_values(["_o", "_d", "Asset"]).drop(columns=["_o", "_d"]).reset_index(drop=True)

    # ---------------------------------------------------------------- figures
    def trend(self, asset: str, points: list[str], case: pd.Series | None) -> str:
        n = len(TREND_UNITS)
        f, axes = style.fig(h=1.9 * n + 0.6, w=12.5, nrows=n, sharex=True)
        colors = [style.BRAND_BLUE, style.AMBER, style.GREEN]
        ref_end = case["opened"].tz_convert(TZ).tz_localize(None) if case is not None else self.start
        ref_start = ref_end - pd.Timedelta(days=BASELINE_DAYS)
        for ax, (_, col, lab, logy) in zip(axes, TREND_UNITS, strict=True):
            for p, colr in zip(points, colors, strict=False):
                g = self.daily[self.daily["point_id"] == p]
                base = g.loc[(g["day"] >= ref_start) & (g["day"] < ref_end), col].dropna()
                if len(base):
                    ax.axhspan(base.quantile(0.05), base.quantile(0.95), color=colr, alpha=0.13, lw=0)
                w = g[(g["day"] >= self.start) & (g["day"] < self.stop)]
                ax.plot(w["day"], w[col], color=colr, lw=1.1, label=p)
            if logy:
                ax.set_yscale("log")
            ax.set_ylabel(lab, fontsize=8)
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
        handles.append(
            plt_band(
                style.GREY,
                "Baseline band, 60 days before the case"
                if case is not None
                else "Baseline band, 60 days before",
            )
        )
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
            f, ax = style.fig(h=2.2, w=12.5)
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
        f, axes = style.fig(h=1.5 * n + 0.3, w=12.5, nrows=n, sharex=True)
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

    def spectra(self, asset: str, case: pd.Series) -> str | None:
        b = self.bursts[self.bursts["case_id"] == case["case_id"]]
        if b.empty:
            return None
        pts = sorted(b["point_id"].unique())
        brg = self.d.st.read_pd("bronze", "reg_bearings").set_index("part")
        sens = self.d.sensors.set_index("point_id")
        hw = float(self.d.cfg["indicators"]["envelope"]["band_half_width_frac"])
        f, axes = style.fig(h=3.0 * 2 * len(pts) + 0.5, w=12.5, nrows=2 * len(pts), squeeze=False)
        colors = {"baseline reference": style.GREY, "latest before the cut": style.RED}
        for j, p in enumerate(pts):
            bar_ax, env_ax = axes[2 * j, 0], axes[2 * j + 1, 0]
            geo = brg.loc[sens.loc[p, "bearing"]]
            orders = kinematics.fault_orders(
                int(geo["n"]), float(geo["d_mm"]), float(geo["pitch_mm"]), float(geo["contact_deg"])
            )
            axis = 1 if str(sens.loc[p, "radial_axis"]) == "y" else 0
            g = b[(b["point_id"] == p) & b["role"].isin(list(colors))]
            energy: dict[str, dict[str, float]] = {}
            ff_latest, bandwidth = None, None
            for r in g.itertuples():
                z = np.load(self.d.st.file(r.file))
                x = filters.remove_dc(np.asarray(z["accel_g"], dtype=float))
                spec, freqs = filters.rfft_with_freqs(x, float(z["fs_hz"]))
                n = x.shape[-1]
                ef, ea = envelope.envelope_spectrum(
                    spec[axis], freqs, n, float(z["fs_hz"]), r.band_lo_hz, r.band_hi_hz
                )
                bandwidth = float(r.band_hi_hz - r.band_lo_hz)
                ff = kinematics.fault_frequencies_hz(orders, r.speed_rpm / 60.0)
                whole = (ef > 0) & (ef <= bandwidth)
                energy[r.role] = {WHOLE_BAND: float((ea[whole] ** 2).sum() / 2.0)} | {
                    f"{name} {h}x": (
                        envelope.band_values(ef, ea, ff[k] * h, hw)[0]
                        if ff[k] * h * (1 + hw) <= bandwidth
                        else np.nan
                    )
                    for name, k, _ in FAULTS
                    for h in (1, 2, 3)
                }
                lab = f"{r.role[:1].upper()}{r.role[1:]}, {local(r.hour_utc, '%Y-%m-%d %H:%M')}"
                env_ax.plot(ef, ea, color=colors[r.role], lw=0.8, label=lab)
                if r.role == "latest before the cut":
                    ff_latest = ff
            if "baseline reference" in energy:
                self._fault_bars(bar_ax, p, energy)
            else:
                bar_ax.axis("off")
                bar_ax.text(
                    0.0,
                    0.5,
                    f"{p}: {fixed_lines(self.t)['No comparable baseline burst']}",
                    transform=bar_ax.transAxes,
                    fontsize=10,
                    color=style.DARK_GREY,
                    va="center",
                )
                self.no_baseline.append(p)
            xmax = min(1.05 * 3 * max(ff_latest[k] for _, k, _ in FAULTS), bandwidth)
            marks = sorted(
                (ff_latest[k] * h, f"{name} {h}x", ls)
                for name, k, ls in FAULTS
                for h in (1, 2, 3)
                if ff_latest[k] * h <= xmax
            )
            levels: list[float] = []
            gap = 0.02 * xmax
            top = 0
            for fx, label, ls in marks:
                env_ax.axvline(fx, color=style.BRAND_BLUE, ls=ls, lw=0.7, alpha=0.75)
                level = 0
                while level < len(levels) and fx - levels[level] < gap:
                    level += 1
                if level == len(levels):
                    levels.append(fx)
                else:
                    levels[level] = fx
                top = max(top, level)
                env_ax.text(
                    fx,
                    1.01 + 0.27 * level,
                    label,
                    transform=env_ax.get_xaxis_transform(),
                    rotation=90,
                    fontsize=6.5,
                    color=style.BRAND_BLUE,
                    ha="center",
                    va="bottom",
                )
            env_ax.set_xlim(0, xmax)
            env_ax.set_title(f"{p}, envelope spectrum", fontsize=9, loc="left", pad=34 + 44 * top)
            env_ax.set_xlabel("Hz", fontsize=8)
            env_ax.set_ylabel("g", fontsize=8)
            env_ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        f.tight_layout(h_pad=1.6)
        return style.save(f, f"queue_spectra_{asset}", f"{asset} spectral evidence")

    @staticmethod
    def _fault_bars(ax, point: str, energy: dict[str, dict[str, float]]) -> None:
        base, last = energy.get("baseline reference", {}), energy.get("latest before the cut", {})
        names = list(last)
        ratio = [
            last[k] / base[k] if base.get(k) and np.isfinite(base[k]) and np.isfinite(last[k]) else np.nan
            for k in names
        ]
        xs = np.arange(len(names))
        colors = {
            "BPFO": style.BRAND_BLUE,
            "BPFI": style.AMBER,
            "BSF": style.GREEN,
            "FTF": style.GREY,
            WHOLE_BAND.split()[0]: style.DARK_GREY,
        }
        ax.bar(xs, np.nan_to_num(ratio, nan=0.0), color=[colors[k.split()[0]] for k in names], width=0.7)
        ax.axhline(1.0, color="#000000", lw=0.9, ls="--")
        finite = [v for v in ratio if np.isfinite(v)]
        ymax = max(finite + [1.0]) * 1.18
        for x, v in zip(xs, ratio, strict=True):
            ax.text(
                x,
                (v if np.isfinite(v) else 0.0) + ymax * 0.015,
                f"{v:.1f}" if np.isfinite(v) else "outside band",
                ha="center",
                va="bottom",
                fontsize=7,
                color=style.DARK_GREY,
            )
        ax.set_ylim(0, ymax)
        ax.set_xticks(xs, [k.replace("envelope ", "envelope\n") for k in names], fontsize=7.5)
        ax.set_ylabel("Latest / baseline", fontsize=8)
        ax.set_title(
            f"{point}, envelope energy at fault frequencies, latest burst against baseline",
            fontsize=9,
            loc="left",
        )

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
                        "Days above threshold": streak,
                    }
                )
        head = f"<p><b>Triggering layer:</b> {html.escape(_layers(case['layers']))}</p>"
        if not rows:
            return head
        return head + style.table(pd.DataFrame(rows))

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

    def last_work(self, asset: str) -> list[tuple[str, str]]:
        """The last repair (with its finding code) and the last PM before the record end."""
        w = self.d.wo
        h = w[w["asset_id"].str.startswith(asset) & (w["closed"] < END)].sort_values("closed")
        out = []
        rep = h[h["wo_type"].isin(["unplanned_repair", "condition_based", "corrective"])]
        if len(rep):
            r = rep.iloc[-1]
            code = f", {str(r['failure_code']).replace('_', ' ')}" if r["failure_code"] else ""
            out.append(("Last repair", f"{r['wo_id']}{code}, {local(r['closed'])}"))
        pm = h[h["wo_type"] == "pm"]
        if len(pm):
            r = pm.iloc[-1]
            out.append(("Last PM", f"{r['wo_id']}, {local(r['closed'])}"))
        return out

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
                (
                    "Case opened",
                    f"{local(case['opened'])}, {q['Days open']} {'day' if q['Days open'] in ('<1', '1') else 'days'} open",
                ),
                ("Suspected condition", q["Suspected condition"]),
                ("Points", q["Points"]),
                ("Request", f"{q['Request id']}, {q['Request status']}"),
            ]
        for label, text in self.last_work(asset):
            summary.append((label, text))
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
                img = self.spectra(asset, case)
                if img:
                    parts.append(f"<h4>{html.escape(labels[3])}</h4>{img}")
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
        fold = ((q["Status"] == "OK") & (q["Case opened"] == "")).to_list()
        attrs = [
            f'data-asset="{html.escape(a)}" class="row status-{s.lower()}{" okfold" if f else ""}"'
            for a, s, f in zip(q["Asset"], q["Status"], fold, strict=True)
        ]
        shown = q.drop(columns=["_days"])
        table = style.table(shown, cls="data queue", row_attrs=attrs)
        first = attrs[fold.index(True)]
        toggle = (
            f'<tr class="fold" id="okfold"><td colspan="{shown.shape[1]}">'
            f'<span class="caret">&#9656;</span> {sum(fold)} assets OK</td></tr>'
        )
        table = table.replace(f"<tr {first}>", toggle + f"<tr {first}>", 1)
        status = t.field("A", "Request status label")
        legend = "".join(f"<li>{html.escape(x)}</li>" for x in t.items("A", "Status legend"))
        legend += f"<li>{html.escape(status)}.</li>"
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
  const row = document.querySelector('tr.row[data-asset="' + a + '"]');
  if (row && row.classList.contains('okfold')) document.body.classList.add('ok-open');
  if (scroll) document.getElementById('panel-' + a).scrollIntoView();
}}
document.querySelectorAll('tr.row').forEach(r => r.addEventListener('click', () => show(r.dataset.asset, true)));
document.getElementById('okfold').addEventListener('click', () => document.body.classList.toggle('ok-open'));
const h = decodeURIComponent(location.hash.replace('#', ''));
if (new URLSearchParams(location.search).get('capture') === 'detail') document.body.classList.add('capture-detail');
if (h.startsWith('panel-')) show(h.slice(6), true); else show(DEFAULT, false);
</script>
</body></html>"""
        if not self.no_baseline:
            t.not_shown.add(fixed_lines(t)["No comparable baseline burst"])
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
table.queue tr.okfold { display: none; }
body.ok-open table.queue tr.okfold { display: table-row; }
table.queue tr.fold td { cursor: pointer; color: var(--muted); font-weight: 600; background: #F7F9FB; }
table.queue tr.fold .caret { display: inline-block; transition: transform 0.15s; }
body.ok-open table.queue tr.fold .caret { transform: rotate(90deg); }
table.queue tr.row:hover td { background: #F3F6F9; }
table.queue tr.selected td { background: #E6EEF5; }
table.queue tr.status-alarm td:nth-child(3) { color: #CC0000; font-weight: 700; }
table.queue tr.status-watch td:nth-child(3) { color: #D4881E; font-weight: 700; }
.panel { display: none; border-top: 3px solid var(--brand); margin-top: 26px; padding-top: 4px; }
.panel h3 { font-size: 19px; margin: 10px 0 4px; }
.panel h4 { border-bottom: 1px solid var(--rule); padding-bottom: 3px; }
.panel img { width: 100%; max-width: 100%; }
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


def plt_band(color: str, label: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=color, alpha=0.25, lw=0, label=label)


def plt_line(color: str, ls: str = "-", label: str = ""):
    from matplotlib.lines import Line2D

    return Line2D([0], [0], color=color, ls=ls, lw=1.2, label=label)


def build(d: Data, t: PageText) -> str:
    return Queue(d, t).build()

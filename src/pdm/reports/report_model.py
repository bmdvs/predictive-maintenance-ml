"""The model report (docs/reports/model_report.html): the monitoring setup, the ML Model against Vendor Rules over
the evaluation year, the model overview and the model evaluation, from the report data bundle.

Inputs: build/report_data (bundle.json, exhibit tables, bursts.json; written by the checkpoint (b) analysis) and the
queue screenshot. Outputs: the page and its figures under docs/figures.

python -m pdm.reports.report_model
"""

from __future__ import annotations

import html
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

from pdm import log, paths
from pdm.reports import style

LOG = log.get_logger("pdm.reports.report_model")
DATA = paths.REPO_ROOT / "build" / "report_data"
ML, VR = style.BRAND_BLUE, style.GREY
VR_LABEL = "#8C8C8C"  # thin grey text reads lighter than the grey line; one step darker matches it
AMB, REDC = style.AMBER, style.RED
TZ = "America/Chicago"
TITLE = "Report: Predictive Maintenance ML Model Overview & Performance"


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def n1(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}"


def esc(s: str) -> str:
    return html.escape(s, quote=False).replace("[[b]]", "<b>").replace("[[/b]]", "</b>")


def p(s: str) -> str:
    return f"<p>{esc(s)}</p>"


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def num(n: float, cap: bool = False) -> str:
    """A count in text: words below ten, numerals from ten up."""
    n = int(n)
    w = WORDS[n] if 0 <= n < 10 else f"{n:,}"
    return w.capitalize() if cap else w


def ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def uline(s: str) -> str:
    return r"$\underline{\mathbf{" + s.replace(" ", r"\ ") + "}}$"


def bold(s: str) -> str:
    return f"<h4>{html.escape(s)}</h4>"


def machine(asset: str) -> str:
    """The machine an asset sits on: HMC-08-SP is on HMC-08; a compressor is its own machine."""
    return "-".join(asset.split("-")[:2])


def loc_date(ts: Any, fmt: str = "%b %-d, %Y") -> str:
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    t = t.tz_convert(TZ)
    return t.strftime(fmt.replace("%-d", str(t.day)))


class Report:
    def __init__(self) -> None:
        self.B = json.loads((DATA / "bundle.json").read_text(encoding="utf-8"))
        self.t = {f.stem: pd.read_parquet(f) for f in DATA.glob("*.parquet")}
        self.bursts = json.loads((DATA / "bursts.json").read_text(encoding="utf-8"))

    def save(self, f: Any, name: str, alt: str) -> str:
        return style.save(f, f"report_{name}", alt, up=1)

    def exhibit(self, title: str, body: str, note: str = "") -> str:
        n = f"<p class='exnote'>{esc(note)}</p>" if note else ""
        return f"<div class='chart-title'>{html.escape(title)}</div>{body}{n}"

    # ------------------------------------------------------------------ section 1
    def fig_layers(self) -> str:
        f, ax = plt.subplots(figsize=(8.8, 2.9))
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(3.4, 5.85)
        existing, new, plain = "#E6E6E6", "#F3E3C6", "white"

        def box(x, y, w, h, t, s="", fc=plain, dashed=False, bold_=True):
            ax.add_patch(
                Rectangle((x, y - h / 2), w, h, fc=fc, ec=style.DARK_GREY, lw=0.9, ls="--" if dashed else "-")
            )
            ax.text(
                x + w / 2,
                y + (0.12 if s else 0),
                t,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if bold_ else "normal",
            )
            if s:
                ax.text(x + w / 2, y - 0.17, s, ha="center", va="center", fontsize=7, color=style.DARK_GREY)

        def arrow(a, b, dashed=False):
            ax.add_patch(
                FancyArrowPatch(
                    a,
                    b,
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color=style.DARK_GREY,
                    lw=0.8,
                    ls=(0, (5, 4)) if dashed else "-",
                )
            )

        box(0.0, 5.0, 1.4, 0.5, "HMCs (8)", "Bottleneck", plain)
        box(0.0, 4.3, 1.4, 0.5, "Compressors (4)", "", plain)
        box(1.7, 4.65, 2.3, 0.75, "Wireless sensor data", "Vibration and temperature, 40 points", plain)
        arrow((1.42, 5.0), (1.68, 4.8))
        arrow((1.42, 4.3), (1.68, 4.5))
        box(4.35, 5.3, 2.35, 0.7, "Vendor Rules", "Threshold alerts", existing)
        box(4.35, 4.0, 2.35, 0.7, "ML Model", "One ML model per sensor point", new, dashed=True)
        arrow((4.02, 4.85), (4.33, 5.2))
        arrow((4.02, 4.45), (4.33, 4.1), dashed=True)
        box(8.3, 4.65, 1.7, 0.7, "Maintenance queue", "", plain)
        arrow((6.72, 5.3), (8.28, 4.8))
        arrow((6.72, 4.0), (8.28, 4.5), dashed=True)
        # left-aligned, the same distance in from the top of Vendor Rules' box and the bottom of the ML Model's
        inset = 0.18
        for y, t in (
            (5.3 + 0.35 - inset, "Outputs drive inspections"),
            (4.0 - 0.35 + inset, "Outputs recorded but not acted on"),
        ):
            ax.text(6.85, y, t, ha="left", va="center", fontsize=7.5, color=style.TEXT)
        handles = [
            Patch(fc=existing, ec=style.DARK_GREY, lw=0.6),
            Patch(fc=new, ec=style.DARK_GREY, lw=0.8, ls="--"),
        ]
        f.legend(
            handles,
            ["Existing Monitoring Layer", "New Monitoring Layer"],
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=8,
        )
        f.tight_layout(rect=(0, 0.1, 1, 1))
        return self.save(f, "layers", "Condition Monitoring Pilot Setup")

    def fig_types(self) -> str:
        d = self.t["conds_by_type"]
        x = pd.crosstab(d["condition_type"], d["asset_class"])
        x = x.loc[x.sum(axis=1).sort_values().index]
        cols = ["HMC spindle", "TSC pump", "Flood coolant pump", "Compressor"]
        colors = [ML, style.ACCENT, style.LIGHT_BLUE, style.GREY]
        f, ax = style.fig(h=3.4, grid="x")
        left = np.zeros(len(x))
        for c, col in zip(cols, colors, strict=True):
            if c in x:
                ax.barh(x.index, x[c], left=left, color=col, label=c, height=0.62)
                left += x[c].to_numpy()
        multi = (x[[c for c in cols if c in x]] > 0).sum(axis=1).to_numpy() > 1
        lft = np.zeros(len(x))
        for c in cols:
            if c not in x:
                continue
            for i, v in enumerate(x[c].to_numpy()):
                if v and multi[i]:
                    ax.text(
                        lft[i] + v / 2, i, f"{int(v)}", ha="center", va="center", fontsize=8, color="white"
                    )
            lft += x[c].to_numpy()
        for i, v in enumerate(left):
            ax.text(v + 0.15, i, f"{int(v)}", va="center", fontsize=9)
        ax.set_xlabel("Conditions")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_xlim(0, left.max() + 1.5)
        return self.save(f, "types", "Sensor-detectable conditions by type")

    def section1(self) -> str:
        b, c = self.B["business"], self.B["conditions"]
        ou = sorted(b["other_util"])
        s = "<h2 id='s1'>1. Condition Monitoring Overview</h2>"
        s += p(
            f"The shop runs 40 CNC machines and four compressors that supply the plant's air. Its eight HMCs are the bottleneck: they carry {pct(b['hmc_share'], 1)} of the shop's cutting hours at {pct(b['hmc_util'])} utilization, against {round(100 * ou[0])}% to {round(100 * ou[-1])}% for the other machine classes, and an hour of HMC downtime costs ${b['margin']:.0f} in contribution margin, {b['ratio']:.2f}x the average for the rest of the fleet."
        )
        s += p(
            "In July 2023 the shop began a pilot of sensor-based condition monitoring on the HMC cell and the compressors: wireless sensors on 40 points, monitored as 28 assets (each HMC's spindle, through-spindle coolant (TSC) pump and flood coolant pump, and each compressor)."
        )
        ti = self.B["training_inspections"]
        s += p(
            f"Through the pilot, the sensor data has been monitored by the [[b]]Vendor Rules[[/b]] layer, the sensor vendor's threshold alerts based on limits over baseline. These alerts led to {ti['alert_driven']} machine inspections from July 2023 to June 2025, and {pct(ti['alert_driven_nff'])} of those inspections found nothing wrong. To improve these results, the pilot's second phase adds the [[b]]ML Model[[/b]] layer, replacing Vendor Rules' relatively simple thresholds with an anomaly detection machine learning model. Over the evaluation year (July 2025 to June 2026), the ML Model ran in shadow mode beside Vendor Rules, with its ALERTs and ALARMs recorded and not acted on, in order to evaluate its performance."
        )
        s += p(
            "This report evaluates the performance of the ML Model against Vendor Rules over the evaluation year."
        )
        s += self.exhibit("Condition Monitoring Pilot Setup", self.fig_layers())
        s += bold("Conditions in the Evaluation Year")
        hc = self.B["conditions_hmc_cmp"]
        ctrl = c["cooler"] + c["ru_leak"]
        s += p(
            f"Across the eight HMCs and four compressors, {hc['total']} conditions were confirmed from July 2025 to June 2026. {c['in_scope']} were on the sensor points and showed in the sensor data; the comparison in section 2 is on those. Of the other {hc['total'] - c['in_scope']}, {NUM.get(hc['no_sensor'], hc['no_sensor'])} was on an HMC axis which has no sensor, {NUM.get(ctrl, ctrl)} showed only in the machine's controller data ({NUM.get(c['cooler'], c['cooler'])} compressor oil cooler faults and {NUM.get(c['ru_leak'], c['ru_leak'])} rotary union leaks without coolant ingress), and {NUM.get(c['no_precursor'], c['no_precursor'])} had no precursor signature in any data (sudden trips, crashes and stoppages)."
        )
        s += self.exhibit("Sensor-detectable conditions by type, evaluation year", self.fig_types())
        return s

    # ------------------------------------------------------------------ section 2
    def fig_levels(self, level: str) -> str:
        a, al = self.B["alert"], self.B["alarm"]
        f, axs = plt.subplots(1, 3, figsize=(8.8, 2.9))
        for x in axs:
            style.chart_style(x)
        names = ["ML Model", "Vendor Rules"]
        cols = [ML, VR]
        pad = 12
        if level == "ALERT":
            x = axs[0]
            for i, m in enumerate(names):
                k = a["strict"][m]
                x.bar(i, k, color=cols[i])
                x.bar(i, 32 - k, bottom=k, color=cols[i], alpha=0.25)
                x.text(i, k / 2, f"{k}", ha="center", va="center", color="white", fontsize=9)
                x.text(i, 32 + 0.5, "of 32", ha="center", fontsize=9)
            x.set_xticks([0, 1])
            x.set_xticklabels(names)
            x.set_ylim(0, 36)
            x.set_title("Conditions Caught", fontsize=9.5, pad=pad)
            x = axs[1]
            vals = [a["warn_both"][m] for m in names]
            x.bar(names, vals, color=cols)
            [x.text(i, v + 0.5, n1(v), ha="center", fontsize=9) for i, v in enumerate(vals)]
            x.set_title(
                f"Median Warning Days\n(on {a['both']} Conditions Both Caught)", fontsize=9.5, pad=pad
            )
            x = axs[2]
            vals = [a["false"][m] for m in names]
            x.bar(names, vals, color=cols)
            [x.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9) for i, v in enumerate(vals)]
            x.set_title("False ALERTs per Asset-Month", fontsize=9.5, pad=pad)
        else:
            x = axs[0]
            vals = [al["share"][m] for m in names]
            x.bar(names, [100 * v for v in vals], color=cols)
            [x.text(i, 100 * v + 1, pct(v), ha="center", fontsize=9) for i, v in enumerate(vals)]
            x.set_title("Share with a Confirmed Condition", fontsize=9.5, pad=pad)
            x = axs[1]
            vals = [al["warn"][m] for m in names]
            x.bar(names, vals, color=cols)
            [x.text(i, v + 0.5, n1(v), ha="center", fontsize=9) for i, v in enumerate(vals)]
            x.set_title(
                f"Median Warning Days\n(on {al['warn']['n']} Conditions Both Escalated)",
                fontsize=9.5,
                pad=pad,
            )
            x = axs[2]
            vals = [al["false"][m] for m in names]
            x.bar(names, vals, color=cols)
            [x.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9) for i, v in enumerate(vals)]
            x.set_title("False ALARMs per Asset-Month", fontsize=9.5, pad=pad)
        for x in axs:
            x.tick_params(axis="x", labelsize=8)
            x.set_ylim(0, x.get_ylim()[1] * 1.08)
        warn = [a["warn_both"][m] for m in names] + [al["warn"][m] for m in names]
        false = [a["false"][m] for m in names] + [al["false"][m] for m in names]
        axs[1].set_ylim(0, max(warn) * 1.15)
        axs[2].set_ylim(0, max(false) * 1.15)
        f.tight_layout()
        return self.save(f, f"levels_{level.lower()}", f"ML Model and Vendor Rules, {level}")

    A0, A1 = pd.Timestamp("2025-07-01 05:00", tz="UTC"), pd.Timestamp("2026-07-01 05:00", tz="UTC")

    def _tl(self) -> pd.DataFrame:
        tl = self.t["timeline"]
        return tl.assign(start=pd.to_datetime(tl["start"], utc=True), end=pd.to_datetime(tl["end"], utc=True))

    def _shade(self, ax: Any, asset: str, y: float, label_size: float = 6.3) -> None:
        cd = self.t["conditions"]
        a0, a1 = self.A0, self.A1
        g = cd[cd["asset"] == asset].copy()
        g["start"] = [
            pd.Timestamp(o) if pd.notna(o) else pd.Timestamp(c) - pd.Timedelta(days=1)
            for o, c in zip(g["onset"], g["confirmed"], strict=True)
        ]
        last = -1e9
        for r in g.sort_values("start").itertuples():
            x0 = (max(r.start, a0) - a0).total_seconds() / 86400
            x1 = (min(pd.Timestamp(r.confirmed), a1) - a0).total_seconds() / 86400
            ax.add_patch(
                Rectangle(
                    (x0, y - 0.66), max(x1 - x0, 0.8), 0.9, fc="#BDBDBD", alpha=0.5, ec="none", zorder=0.8
                )
            )
            low = x0 - last < 45
            last = -1e9 if low else x0
            ax.text(
                x0 + 0.8,
                y - 0.68 if low else y + 0.2,
                TYPE_LABEL.get(r.condition_type, r.condition_type),
                fontsize=label_size,
                color=style.DARK_GREY,
                va="top" if low else "bottom",
                ha="left",
                zorder=1,
            )

    def _lanes(
        self, ax: Any, tl: pd.DataFrame, v: str, y: float, ticks: list, labels: list, label_size: float = 6.3
    ) -> None:
        a0 = self.A0
        self._shade(ax, v, y, label_size)
        for lane, dy in (("ML Model", 0.0), ("Vendor Rules", -0.42)):
            g = tl[(tl["asset"] == v) & (tl["lane"] == lane)]
            for r in g.itertuples():
                if r.kind in ("ALERT", "ALARM"):
                    ax.barh(
                        y + dy,
                        (r.end - r.start).total_seconds() / 86400,
                        left=(r.start - a0).total_seconds() / 86400,
                        height=0.34,
                        color=AMB if r.kind == "ALERT" else REDC,
                        lw=0,
                        zorder=2,
                    )
                elif lane == "Vendor Rules" and r.kind.startswith("inspection"):
                    filled = r.kind.endswith("confirmed")
                    ax.plot(
                        (r.start - a0).total_seconds() / 86400,
                        y + dy,
                        "o",
                        ms=4,
                        mfc=style.TEXT if filled else "white",
                        mec=style.TEXT,
                        mew=0.8,
                        zorder=3,
                    )
            ticks.append(y + dy)
            labels.append(f"{v}  {'ML' if lane == 'ML Model' else 'VR'}")
        for r in tl[(tl["asset"] == v) & (tl["lane"] == "both")].itertuples():
            xx = (r.start - a0).total_seconds() / 86400
            if r.kind == "failure":
                ax.plot(xx, y - 0.21, "x", ms=7, mew=1.6, color=style.TEXT, zorder=4)
            else:
                ax.plot(xx, y - 0.21, "D", ms=4, mfc=style.GREEN, mec=style.GREEN, zorder=4)

    def _xaxis(self, ax: Any, labels: bool = True, size: float = 8) -> None:
        a0 = self.A0
        months = pd.date_range("2025-07-01", "2026-06-01", freq="MS")
        ax.set_xticks(
            [(pd.Timestamp(m, tz=TZ).tz_convert("UTC") - a0).total_seconds() / 86400 for m in months]
        )
        ax.set_xticklabels([m.strftime("%b") for m in months] if labels else [], fontsize=size)
        ax.set_xlim(0, (self.A1 - a0).total_seconds() / 86400)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.xaxis.grid(True, color="#EEEEEE")
        ax.set_axisbelow(True)

    def _legend(self, f: Any, top: bool, size: float = 7.5, repair: bool = False) -> None:
        hd = [
            Patch(fc="#D9D9D9"),
            Patch(fc=AMB),
            Patch(fc=REDC),
            Line2D([], [], marker="o", ls="", mfc="white", mec=style.TEXT),
            Line2D([], [], marker="o", ls="", mfc=style.TEXT, mec=style.TEXT),
            Line2D([], [], marker="D", ls="", mfc=style.GREEN, mec=style.GREEN),
            Line2D([], [], marker="x", ls="", color=style.TEXT, mew=1.6),
        ]
        labs = [
            "Condition, onset to confirmation",
            "ALERT",
            "ALARM",
            "Inspection, nothing found",
            "Inspection, condition confirmed",
            "PM finding",
            "Failure",
        ]
        if repair:
            # the example panels: no PM finding among them; the planned repair marks the confirmation
            keep = [i for i, lab in enumerate(labs) if lab != "PM finding"]
            hd, labs = [hd[i] for i in keep], [labs[i] for i in keep]
            hd.append(Line2D([], [], marker="s", ls="", mfc=style.BRAND_BLUE, mec=style.BRAND_BLUE))
            labs.append("Planned repair")
        f.legend(
            hd, labs, loc="upper center" if top else "lower center", ncol=4, frameon=False, fontsize=size
        )

    def panel_titles(self) -> list[tuple[str, str, str]]:
        cd = self.t["conditions"].set_index("wo_id")
        neither = cd.loc["HMC-06-WO00036"]
        assert not neither["ml_strict"] and not neither["vr_strict"] and neither["unplanned"]
        ml_only = cd.loc["HMC-01-WO00027"]
        assert ml_only["ml_strict"] and not ml_only["vr_strict"] and ml_only["unplanned"]
        return [
            (
                "HMC-01-WO00027",
                "Condition 1: Flood Pump Bearing Failure",
                f"ML Model caught {ml_only['ml_warn']:.0f} days ahead, Vendor Rules raised no ALERT",
            ),
            (
                "HMC-06-WO00035",
                "Condition 2: Flood Pump Wear",
                f"Vendor Rules caught {cd.at['HMC-06-WO00035', 'vr_warn']:.0f} days ahead, ML Model raised no ALERT",
            ),
            (
                "HMC-01-WO00024",
                "Condition 3: TSC Pump Bearing Defect",
                f"ML Model caught {cd.at['HMC-01-WO00024', 'ml_warn']:.0f} days ahead, Vendor Rules {cd.at['HMC-01-WO00024', 'vr_warn']:.0f} days ahead",
            ),
            (
                "HMC-06-WO00036",
                "Condition 4: TSC Pump Seal Failure",
                "Neither raised an alert beforehand and it ended in a failure",
            ),
        ]

    def _cond_panel(self, ax: Any, wo: str, tl: pd.DataFrame) -> None:
        """One condition over the 120 days before its confirmation: only the episodes credited with it,
        Vendor Rules' inspections in the window and the confirming event at the right edge."""
        r = self.t["conditions"].set_index("wo_id").loc[wo]
        cf = pd.Timestamp(r["confirmed"])
        a0, a1 = cf - pd.Timedelta(days=PANEL_DAYS), cf + pd.Timedelta(days=2)

        def x(t: Any) -> float:
            return (pd.Timestamp(t) - a0).total_seconds() / 86400

        on = pd.Timestamp(r["onset"]) if pd.notna(r["onset"]) else cf - pd.Timedelta(days=1)
        ax.add_patch(
            Rectangle(
                (x(max(on, a0)), -0.66), x(cf) - x(max(on, a0)), 0.9, fc="#BDBDBD", alpha=0.5, ec="none"
            )
        )
        for k, y in (("ml", 0.0), ("vr", -0.42)):
            if not bool(r[f"{k}_strict"]) or pd.isna(r[f"{k}_opened"]):
                continue
            o, c = max(pd.Timestamp(r[f"{k}_opened"]), a0), min(pd.Timestamp(r[f"{k}_closed"]), cf)
            am = pd.Timestamp(r[f"{k}_alarm_ts"]) if pd.notna(r[f"{k}_alarm_ts"]) else None
            mid = min(am, c) if am is not None and am < c else c
            ax.barh(y, x(mid) - x(o), left=x(o), height=0.34, color=AMB, lw=0, zorder=2)
            if am is not None and am < c:
                ax.barh(
                    y, x(c) - x(max(am, a0)), left=x(max(am, a0)), height=0.34, color=REDC, lw=0, zorder=2
                )
        g = tl[(tl["asset"] == r["asset"]) & (tl["start"] >= a0) & (tl["start"] <= a1)]
        for e in g[(g["lane"] == "Vendor Rules") & g["kind"].str.startswith("inspection")].itertuples():
            filled = e.kind.endswith("confirmed")
            ax.plot(
                x(e.start), -0.42, "o", ms=5, mfc=style.TEXT if filled else "white", mec=style.TEXT, zorder=3
            )
        both = g[g["lane"] == "both"]
        for e in both.itertuples():
            if e.kind == "failure":
                ax.plot(x(e.start), -0.21, "x", ms=8, mew=1.8, color=style.TEXT, zorder=4)
            else:
                ax.plot(x(e.start), -0.21, "D", ms=5, mfc=style.GREEN, mec=style.GREEN, zorder=4)
        # the finding at a PM check, and the planned repair at the confirmation (the end of the warning)
        if r["found_by"] == "pm" and not (both["kind"] != "failure").any():
            ax.plot(x(r["found_ts"]), -0.21, "D", ms=5, mfc=style.GREEN, mec=style.GREEN, zorder=4)
        same_hour = pd.notna(r["found_ts"]) and abs(pd.Timestamp(r["found_ts"]) - cf) < pd.Timedelta(hours=12)
        if not bool(r["unplanned"]) and not same_hour:
            ax.plot(x(cf), -0.21, "s", ms=5.5, mfc=style.BRAND_BLUE, mec=style.BRAND_BLUE, zorder=5)
        ticks = pd.date_range(a0.tz_convert(TZ).normalize(), cf.tz_convert(TZ), freq="SMS")
        ax.set_xticks([x(t.tz_convert("UTC")) for t in ticks])
        ax.set_xticklabels([f"{t:%b} {t.day}" for t in ticks], fontsize=9.5)
        ax.set_xlim(0, x(a1))
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.xaxis.grid(True, color="#EEEEEE")
        ax.set_axisbelow(True)

    def panels(self, rows: list[tuple[str, str, str]] | None = None, name: str = "timeline") -> str:
        """One timeline per example condition, each under its own title; the legend under the last."""
        tl = self._tl()
        rows = rows or self.panel_titles()
        assets = self.t["conditions"].set_index("wo_id")["asset"]
        out = ""
        for i, (wo, head, sub) in enumerate(rows):
            head = f"{head} ({machine(assets[wo])})"
            last = i == len(rows) - 1
            f, ax = plt.subplots(figsize=(8.8, 1.55 + (0.55 if last else 0)))
            self._cond_panel(ax, wo, tl)
            ax.set_yticks([0.0, -0.42])
            ax.set_yticklabels(["ML Model", "Vendor Rules"], fontsize=9.5)
            ax.set_ylim(-0.95, 0.6)
            if last:
                self._legend(f, top=False, size=9, repair=True)
                f.tight_layout(rect=(0, 0.2, 1, 1))
            else:
                f.tight_layout()
            img = self.save(f, f"{name}_{i + 1}", f"{head}. {sub}")
            out += (
                f"<div class='chart-title panel-title'>{html.escape(head)}</div>"
                f"<div class='panel-sub'>{html.escape(sub)}</div>{img}"
            )
        return out

    def fig_timeline(self, assets: list[str], name: str, title: str) -> str:
        tl = self._tl()
        machines = sorted({a[:6] for a in assets}, key=lambda m: (m.startswith("CMP"), m))
        if any(a.startswith("CMP") for a in assets):
            machines = [m for m in machines if not m.startswith("CMP")] + ["Compressors"]
        rows = []
        for m in machines:
            rows.append(("head", m))
            for a in [x for x in assets if (x[:6] == m or (m == "Compressors" and x.startswith("CMP")))]:
                rows.append(("asset", a))
        h = 0.36 * sum(2 if k == "asset" else 1 for k, _ in rows) + 1.1
        f, ax = plt.subplots(figsize=(8.8, max(h, 2.5)))
        y = 0
        ticks, labels = [], []
        for kind, v in rows:
            if kind == "head":
                ax.text(
                    -0.01,
                    y,
                    v,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=style.BRAND_BLUE,
                )
                y -= 0.8
                continue
            self._lanes(ax, tl, v, y, ticks, labels)
            y -= 1.35
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)
        self._xaxis(ax)
        ax.set_ylim(y + 0.6, 0.6)
        self._legend(f, top=False)
        f.tight_layout(rect=(0, 0.035, 1, 1))
        return self.save(f, name, title)

    def fig_curves(self) -> str:
        cv = self.t["curves"]
        cv = cv[cv["setting"] != "operating threshold (prod)"]
        f, ax = style.fig(h=4.2)
        for m, c in (("ML Model", ML), ("Vendor Rules", VR)):
            g = cv[cv["method"] == m]
            ax.plot(
                g["false per asset-month (truth-based)"], 100 * g["strict"] / 32, "-o", color=c, ms=4, label=m
            )
            op = g[g["operating point"]]
            ax.plot(op["false per asset-month (truth-based)"], 100 * op["strict"] / 32, "o", color=c, ms=10)
            ends = [g.iloc[0], g.iloc[-1], op.iloc[0]]
            labs = (
                [
                    "Loosest: 84th percentile",
                    "Strictest: 99.99th percentile",
                    "Chosen setting: 99.5th percentile",
                ]
                if m == "ML Model"
                else [
                    "Loosest: 10% of zone B width",
                    "Strictest: 100% of zone B width",
                    "Current setting: 25%",
                ]
            )
            offs = [(18, -10), (14, -22), (22, 14)] if m == "ML Model" else [(-30, -18), (18, 10), (22, 16)]
            tc = c if m == "ML Model" else VR_LABEL
            for r, lab, o in zip(ends, labs, offs, strict=True):
                ax.annotate(
                    lab,
                    (r["false per asset-month (truth-based)"], 100 * r["strict"] / 32),
                    textcoords="offset points",
                    xytext=o,
                    fontsize=7.5,
                    ha="left" if o[0] > 0 else "right",
                    va="center",
                    color=tc,
                    arrowprops={"arrowstyle": "-", "color": tc, "lw": 0.7, "shrinkA": 1, "shrinkB": 4},
                )
        ax.set_xlabel("False alerts per asset-month")
        ax.set_ylabel("Conditions strictly caught, % of 32")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(50, 76)
        ax.legend(frameon=False, loc="upper right")
        return self.save(f, "curves", "Conditions caught against false episodes per asset-month")

    def fig_grid(self) -> str:
        g = self.t["diag_grid"]
        names = {
            "bearing": "Bearing",
            "lubrication": "Lubrication",
            "cavitation or wear": "Pump seal or impeller wear",
            "spindle cooling": "Spindle cooling",
            "rotary union": "Rotary union",
            "imbalance": "Imbalance",
            "looseness": "Looseness",
        }
        f, ax = style.fig(h=3.4, grid="x")
        ax.xaxis.grid(False)
        fams = list(names)
        for i, fam in enumerate(fams):
            r = g[g["family"] == fam]
            ok = int((r["d2"] == fam).sum())
            wrong = r[r["d2"] != fam]["d2"].value_counts()
            y = len(fams) - 1 - i
            ax.barh(y, ok, color=ML, height=0.6)
            ax.barh(y, len(r) - ok, left=ok, color=VR, alpha=0.6, height=0.6)
            if ok:
                ax.text(
                    ok / 2,
                    y,
                    str(ok),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
            note = ", ".join(
                f"{n} {'unclassified' if k == 'unclassified' else 'named ' + k}" for k, n in wrong.items()
            )
            if fam == "spindle cooling" and len(wrong):
                note += " (alert shared with a lubrication episode)"
            if note:
                ax.text(len(r) + 0.15, y, note, va="center", fontsize=7.5, color=style.DARK_GREY)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels(
            [f"{names[fm]} ({int((g['family'] == fm).sum())})" for fm in fams][::-1], fontsize=8
        )
        ax.set_xlim(0, 13)
        ax.set_xticks(range(0, 7))
        ax.set_xlabel("Conditions caught by the ML Model")
        ax.legend(
            [Patch(fc=ML), Patch(fc=VR, alpha=0.6)],
            ["Diagnosed correctly", "Diagnosed incorrectly"],
            frameon=False,
            fontsize=8,
            loc="lower right",
        )
        return self.save(f, "diagnosis", "Diagnosis by confirmed condition")

    def section2(self) -> str:
        a, al, ins, d = self.B["alert"], self.B["alarm"], self.B["inspection"], self.B["diag"]
        fl, fr = self.B["failures"], self.t["failures"]
        s = "<h2 id='s2'>2. Model Results</h2>"
        fewer = 1 - a["false"]["ML Model"] / a["false"]["Vendor Rules"]
        fewer_alarm = 1 - al["false"]["ML Model"] / al["false"]["Vendor Rules"]
        longer = a["warn_both"]["ML Model"] / a["warn_both"]["Vendor Rules"] - 1
        share_pts = round(100 * al["share"]["ML Model"]) - round(100 * al["share"]["Vendor Rules"])
        s += p(
            "The ML Model was trained on the first two years of sensor data and has been live in production since July 2025. The results below are presented for the entire evaluation year (July 2025 through June 2026)."
        )
        s += p(
            f"The ML Model achieved notable improvements over Vendor Rules. It caught {a['strict']['ML Model']} of 32 conditions compared to {a['strict']['Vendor Rules']} of 32; its ALERTs gave an {pct(longer)} longer warning time and produced a {pct(fewer)} lower false-alert rate. Had the shop acted on the ML Model's ALERTs instead of Vendor Rules', it would have spent {ins['saved']:.0f} fewer hours on inspections of false alerts over the year."
        )
        s += self.exhibit("ML Model vs. Vendor Rules: ALERTS", self.fig_levels("ALERT"))
        s += p(
            f"The ML Model's ALARMs were generally more accurate than Vendor Rules', with a {share_pts} percentage point higher share of ALARMs with a confirmed condition and a {pct(fewer_alarm)} lower false ALARM rate, but this came at the cost of less warning time."
        )
        s += self.exhibit("ML Model vs. Vendor Rules: ALARMS", self.fig_levels("ALARM"))
        s += bold("Detection at ALERT")
        s += p(
            "ALERT is the level at which each method calls for an inspection, so this is the detection comparison. Over the evaluation year, only Vendor Rules' ALERTs triggered inspections in the maintenance queue. The ML Model ran in the background, where its results were recorded and not acted on, so its effectiveness could be measured against Vendor Rules."
        )
        s += self.exhibit(
            "Alert threshold by method",
            style.table(
                pd.DataFrame(
                    [
                        {
                            "ML Model": "Anomaly score above the point's threshold, the 99.5th percentile of its scores on healthy training hours, for 6 consecutive hours; or above 5 times the threshold for 2 hours, except on pumps during a controller low-pressure alarm",
                            "Vendor Rules": "Velocity above the point's baseline plus 25% of the ISO 20816 zone B width, or high-frequency acceleration above its baseline plus 25% of its reference width; a bearing fault band 6 dB above its baseline for 3 hours; velocity in ISO zone C; or housing temperature at the asset class's warning limit. Baselines are set over 21 days after installation and after each repair.",
                        }
                    ]
                ),
                cls="data even2",
            ),
        )
        s += p(
            "A condition counts as caught when an ALERT was open on the asset in the 30 days before confirmation. An ALERT that opened more than seven days before the condition began, with no sign of degradation on the asset, is deemed coincidental and not counted."
        )
        s += p(
            f"[[b]]Of the 32 conditions, the ML Model's ALERTs caught {a['strict']['ML Model']} and Vendor Rules caught {a['strict']['Vendor Rules']}[[/b]]. {a['both']} conditions were caught by both models."
        )
        s += ul(
            [
                f"The ML Model caught {num(a['only_ml'])} conditions Vendor Rules missed: two spindle cooling faults, a lubrication episode, a bearing failure and a rotary union leak. On these, Vendor Rules either had no alert (2) or produced unrelated ALERTs that missed the condition (3).",
                f"Vendor Rules caught {num(a['only_vr'])} conditions the ML Model missed: two with under half a day's warning (a spindle cooling fault and a compressor lubrication episode), a pump wear cycle the ML Model had learned as normal in training, and a rotary union leak that fell inside a long ML Model alert on the same spindle.",
            ]
        )
        s += p(
            "The chart below shows four representative conditions across different outcomes: a condition only the ML Model caught, one only Vendor Rules caught, one that both caught, and one that neither caught. The full chart across all monitored assets is found in the appendix."
        )
        s += self.panels()
        s += p(
            f"[[b]]The ML Model produces {a['false']['ML Model']:.3f} false ALERTs per asset-month against the Vendor Rules' {a['false']['Vendor Rules']:.3f}[[/b]]. Vendor Rules raised {ins['vr_n']} false alerts, which took {ins['vr_hours']:.0f} inspection hours. The ML Model raised {ins['ml_n']}, which would have taken about {ins['ml_hours']:.0f} hours at the same labor rate. At their peak month, Vendor Rules' inspections took {pct(ins['vr_peak_share'], 1)} of the three technicians' available hours, against {pct(ins['ml_peak_share'], 1)} if the ML Model's ALERTs were used to drive inspections."
        )
        cv = self.t["curves"]
        mlc = cv[(cv["method"] == "ML Model") & (cv["setting"] != "operating threshold (prod)")]
        vop = float(
            cv[(cv["method"] == "Vendor Rules") & cv["operating point"]][
                "false per asset-month (truth-based)"
            ].iloc[0]
        )
        left = mlc[mlc["false per asset-month (truth-based)"] < vop]
        s += p(
            f"The operating curve below shows conditions caught against false alerts per asset-month for ML Model and Vendor Rules at different alert threshold levels. The ML Model's curve lies to the left of Vendor Rules' current setting from the {left['setting'].iloc[0].split(' ')[0]} percentile up: at those thresholds it raised fewer false alerts than Vendor Rules does today, while catching {int(left['strict'].min())} to {int(left['strict'].max())} conditions."
        )
        s += self.exhibit(
            "Conditions caught against false episodes per asset-month, evaluation year", self.fig_curves()
        )
        s += bold("Escalation to ALARM")
        s += p(
            "ALARM is each method's urgent level, so this tests whether an escalation pointed at a real problem."
        )
        s += self.exhibit(
            "ALARM threshold by method",
            style.table(
                pd.DataFrame(
                    [
                        {
                            "ML Model": "An open ALERT plus a severity criterion for the asset class: velocity in ISO 20816 zone D; a stage 3 or 4 bearing signature; the housing temperature residual above its set level; the spindle thermistor at 44 °C or above for 10 minutes, or compressor discharge temperature at 100 °C or above, from the controller; or sustained cavitation on a TSC pump",
                            "Vendor Rules": "Velocity above the point's baseline plus 50% of the ISO 20816 zone B width, or high-frequency acceleration above its baseline plus 50% of its reference width; velocity in ISO zone D; or housing temperature at the asset class's alarm limit",
                        }
                    ]
                ),
                cls="data even2",
            ),
        )
        nff = int((fr["vr_alarm_before"] & (fr["inspections"] > 0) & (fr["found"] == 0)).sum())
        fnd = int((fr["vr_alarm_before"] & (fr["found"] > 0)).sum())
        noi = int((fr["vr_alarm_before"] & (fr["inspections"] == 0)).sum())
        s += p(
            f"[[b]]The ML Model's ALARMs were more accurate than Vendor Rules[[/b]]: {pct(al['share']['ML Model'])} of its {al['n']['ML Model']} ALARMs had a confirmed condition, against {pct(al['share']['Vendor Rules'])} of Vendor Rules' {al['n']['Vendor Rules']} ALARMs, and it produced a {pct(1 - al['false']['ML Model'] / al['false']['Vendor Rules'])} lower false-alarm rate ({al['false']['ML Model']:.3f} against {al['false']['Vendor Rules']:.3f}). On the sensor data alone, which both models read, the ML Model's false-alarm rate was even lower, at {al['false_sensor_ml']:.3f}. This accuracy came at the expense of warning time: the ML Model's ALARMs gave a median of {n1(al['warn']['ML Model'])} days of warning against Vendor Rules' {n1(al['warn']['Vendor Rules'])} days on the {al['warn']['n']} conditions that both layers escalated."
        )
        s += p(
            f"Vendor Rules ALARMed before {num(fl['vr_alarm_before'])} of the {num(fl['n'])} unplanned failures, and the failures still happened: the inspections it triggered found no fault in {num(nff)}, found the problem without preventing the failure in {num(fnd)}, and no inspection was raised in the other {num(noi)}."
        )
        s += bold("Condition Diagnosis")
        ru = {r["wo_id"]: r for r in self.B["ru"]["catches"]}
        s += p(
            f"Work requests on the ML Model's ALERTs and ALARMs correctly diagnosed the confirmed condition on {d['strict_named']} of the {d['strict']} conditions it caught."
        )
        s += p(
            f"The ML Model's work requests were strongest on conditions with a distinct signature, naming all {num(d['bearing'])} bearing defects ({num(d['bearing_located'])} with the damaged component located), {num(d['lub_named'])} of {num(d['lub'])} lubrication episodes and both imbalance and looseness cases. It struggled where conditions overlapped or share a signal: three of four pump-wear cases were misnamed, mostly as imbalance; the three spindle cooling faults were named lubrication, the condition sharing their alert; and one of two rotary union leaks was named bearing despite a {100 * ru['HMC-03-WO00045']['drop share max']:.1f}% coolant pressure drop."
        )
        s += self.exhibit("Diagnosis by confirmed condition", self.fig_grid())
        return s

    # ------------------------------------------------------------------ section 3
    def fig_signal(self) -> str:
        b = next(x for x in self.bursts if x["role"] == "healthy")
        fs = 25600.0
        f, axs = plt.subplots(4, 1, figsize=(8.8, 7.6))
        for x in axs:
            style.chart_style(x)
        w = np.array(b["wave"])
        axs[0].plot(np.arange(len(w)) / fs * 1000, w, color=ML, lw=0.6)
        axs[0].set_title("Acceleration, 50 ms", fontsize=9.5)
        axs[0].set_xlabel("ms", fontsize=8)
        fr, am = np.array(b["freqs"]), np.array(b["amp"])
        m = (fr >= 10) & (fr <= 10000)
        for i in (1, 2):
            axs[i].semilogy(fr[m], am[m], color=ML, lw=0.5)
            axs[i].axvline(b["shaft_hz"], color=AMB, lw=0.9, ls="--")
            axs[i].text(b["shaft_hz"] * 1.15, am[m].max() * 0.5, "1x shaft speed", fontsize=7.5, color=AMB)
            axs[i].set_xlabel("Hz", fontsize=8)
        axs[1].set_title("Spectrum, 10 Hz to 10 kHz", fontsize=9.5)
        hf = (fr >= 2000) & (fr <= 10000)
        rms = float(np.sqrt(np.sum((am[hf] / np.sqrt(2)) ** 2 * 4)))
        axs[2].axvspan(2000, 10000, color=style.LIGHT_BLUE, alpha=0.35)
        axs[2].set_title(f"High-frequency band, 2 to 10 kHz (RMS {rms:.3f} g)", fontsize=9.5)
        ef, ea = np.array(b["env_f"]), np.array(b["env_a"])
        k = ef > 1
        axs[3].plot(ef[k], ea[k], color=ML, lw=0.5)
        for name, key in (("BPFO", "bpfo"), ("BPFI", "bpfi"), ("BSF", "bsf")):
            fz = b["fault_hz"][key]
            if fz <= 3000:
                axs[3].axvline(fz, color=style.DARK_GREY, lw=0.8, ls=":")
                axs[3].text(fz, ea[k].max() * 0.95, name, fontsize=7.5, ha="left", va="top")
        axs[3].set_title("Envelope spectrum, 0 to 3 kHz", fontsize=9.5)
        axs[3].set_xlabel("Hz", fontsize=8)
        f.tight_layout()
        return self.save(f, "signal", "One second of vibration, from raw signal to fault-frequency spectrum")

    def fig_inputs(self) -> str:
        inp = self.t["inputs20"]
        f, ax = style.fig(h=3.4)
        colors = [ML if k_ == "sensor" else style.ACCENT for k_ in inp["kind"]]
        ax.bar(range(len(inp)), inp["value"], color=colors)
        ax.axhline(0, color=style.DARK_GREY, lw=0.6)
        ax.set_xticks(range(len(inp)))
        ax.set_xticklabels([LABELS.get(c, c) for c in inp["input"]], rotation=60, ha="right", fontsize=8)
        ax.legend(
            [Patch(fc=ML), Patch(fc=style.ACCENT)],
            ["Sensor Measurements", "Controller Measurements"],
            frameon=False,
            fontsize=8.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.42),
            ncol=2,
        )
        return self.save(f, "inputs", "The Model's 20 Inputs: Representative Readings on a Healthy Hour")

    def fig_healthy_band(self) -> str:
        d = self.t["healthy_band"]
        meas = [
            ("vel_rms_mm_s", "Velocity, mm/s"),
            ("hf_rms_g", "HF acceleration, g"),
            ("env_bpfo_1_g2", "Envelope BPFO, g²"),
            ("env_bpfi_1_g2", "Envelope BPFI, g²"),
            ("env_bsf_1_g2", "Envelope BSF, g²"),
        ]
        f, axs = plt.subplots(1, 5, figsize=(8.8, 2.8))
        cols = {"low": style.LIGHT_BLUE, "mid": style.ACCENT, "high": ML}
        for ax, (k, lab) in zip(axs, meas, strict=True):
            style.chart_style(ax)
            for i, bnd in enumerate(("low", "mid", "high")):
                r = d[(d["measure"] == k) & (d["band"] == bnd)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([i, i], [r["p05"], r["p95"]], color=cols[bnd], lw=3, solid_capstyle="round")
                ax.plot(i, r["p50"], "o", color=cols[bnd], mec="white")
            ax.set_xticks(range(3))
            ax.set_xticklabels(["Low", "Mid", "High"], fontsize=7.5)
            ax.set_title(lab, fontsize=8.5)
            if k.startswith("env"):
                ax.set_yscale("log")
        f.tight_layout()
        return self.save(f, "healthy_band", "Healthy measurements by speed band, spindle front points")

    def fig_train_hours(self) -> str:
        d = self.t["train_hours"].copy()
        order = {
            "spindle_front": 0,
            "spindle_rear": 1,
            "tsc_pump": 2,
            "flood_pump": 3,
            "compressor_motor": 4,
            "compressor_airend": 5,
        }
        d["o"] = d["point_class"].map(order)
        d = d.sort_values(["o", "point_id"])
        f, ax = style.fig(h=3.6)
        colors = [
            ML if c.startswith("spindle") else style.ACCENT if "pump" in c else style.GREY
            for c in d["point_class"]
        ]
        rr = self.t["train_repairs"].set_index("point_id")["repairs_remounts"]
        ax.bar(range(len(d)), d["n_train"], color=colors)
        for i, (v, pid) in enumerate(zip(d["n_train"], d["point_id"], strict=True)):
            ax.text(i, v + 60, str(int(rr.get(pid, 0))), ha="center", fontsize=6.5, color=style.DARK_GREY)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d["point_id"], rotation=90, fontsize=6.5)
        ax.set_ylabel("Healthy training hours")
        ax.legend(
            [Patch(fc=ML), Patch(fc=style.ACCENT), Patch(fc=style.GREY)],
            ["Spindle points", "Pump points", "Compressor points"],
            frameon=False,
            fontsize=8,
            loc="upper right",
        )
        ax.text(
            0.0,
            1.02,
            "Number above each bar: repairs and sensor remounts in the training years",
            transform=ax.transAxes,
            fontsize=7.5,
            color=style.DARK_GREY,
        )
        return self.save(f, "train_hours", "Healthy training hours per sensor point")

    def fig_rebuild(self) -> str:
        d = self.t["rebuild"]
        f, axs = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=False)
        for ax, lab in zip(axs, ("healthy", "faulty"), strict=True):
            style.chart_style(ax)
            g = d[d["hour"] == lab].iloc[:13]
            x = np.arange(len(g))
            ax.bar(x, g["z"], color=style.LIGHT_BLUE, label="Input")
            ax.plot(x, g["rebuild"], "-o", color=AMB, ms=3, lw=1.2, label="Rebuild")
            for i, (z, r) in enumerate(zip(g["z"], g["rebuild"], strict=True)):
                ax.plot([i, i], [z, r], color=REDC, lw=2.5, alpha=0.35)
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS.get(c, c) for c in g["input"]], rotation=60, ha="right", fontsize=7)
            ts = loc_date(g["ts"].iloc[0])
            ax.set_title(
                f"{'Healthy hour' if lab == 'healthy' else 'Faulty hour'}, HMC-01-TSC, {ts}", fontsize=9
            )
            top = 10 if lab == "healthy" else 100
            ax.set_ylim(min(ax.get_ylim()[0], -top * 0.2), top)
            sc_, th_ = float(g["score"].iloc[0]), float(g["threshold"].iloc[0])
            ax.text(
                0.5,
                0.95,
                f"Anomaly score {sc_:.2f}, {sc_ / th_:.2f}× threshold"
                if lab == "healthy"
                else f"Anomaly score {sc_:.0f}, {sc_ / th_:.0f}× threshold",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "fc": "white",
                    "ec": style.DARK_GREY if lab == "healthy" else REDC,
                    "lw": 0.6 if lab == "healthy" else 0.9,
                },
            )
        hd, lb = axs[0].get_legend_handles_labels()
        f.legend(hd, lb, frameon=False, fontsize=8, loc="lower center", ncol=2)
        f.tight_layout(rect=(0, 0.06, 1, 1))
        return self.save(f, "rebuild", "Inputs against the model's rebuild, healthy hour and faulty hour")

    def fig_trace(self) -> str:
        d = self.t["score_trace"].copy()
        d["hour_utc"] = pd.to_datetime(d["hour_utc"], utc=True)
        d = d[d["state"] == "scored"]
        tr = self.B["trace"]
        f, ax = style.fig(h=3.2)
        ax.semilogy(d["hour_utc"], d["score"], color=ML, lw=0.7, label="Score")
        ax.semilogy(d["hour_utc"], d["threshold"], color=style.DARK_GREY, lw=1.0, ls="--", label="Threshold")
        top = d["score"].max()
        marks = [
            *((x, "ALERT", AMB) for x in tr["alerts"]),
            *((x, "ALARM", REDC) for x in tr["alarms"]),
            (tr["confirmed"], "Confirmed", style.TEXT),
        ]
        for i, (t, lab, c) in enumerate(sorted(marks, key=lambda m: pd.Timestamp(m[0]))):
            tt = pd.Timestamp(t)
            if d["hour_utc"].min() <= tt <= d["hour_utc"].max():
                ax.axvline(tt, color=c, lw=1.0)
                ax.text(
                    tt,
                    top * (3.0 if i % 2 else 1.3),
                    f" {lab}",
                    color=c,
                    fontsize=7.5,
                    va="bottom",
                    ha="left" if lab != "Confirmed" else "left",
                )
        ax.set_ylim(top=top * 8)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2)
        ax.set_title(
            "HMC-08-SP-F, spindle front point, around a ball bearing defect (HMC-08-WO00046)", fontsize=9
        )
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %-d".replace("%-d", "%d")))
        return self.save(f, "trace", "From score to level, one spindle point")

    def fig_score_dist(self) -> str:
        d = self.t["score_dist"]
        f, ax = style.fig(h=3.0)
        bins = np.logspace(-2, 4, 60)
        for g, c in (("no condition", VR), ("120 days before a confirmed condition", ML)):
            x = d.loc[d["group"] == g, "ratio"].clip(1e-2, 1e4)
            ax.hist(
                x,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.45,
                color=c,
                label=f"Hours with {g}" if g == "no condition" else f"Hours {g}",
            )
        ax.set_xscale("log")
        ax.axvline(1.0, color=REDC, lw=1.2)
        ax.text(1.1, ax.get_ylim()[1] * 0.9, "Threshold", color=REDC, fontsize=8)
        ax.set_xlabel("Score relative to the point's threshold")
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8)
        return self.save(f, "score_dist", "Score distribution, evaluation year")

    def fig_level_month(self) -> str:
        d = self.t["level_month"]
        f, ax = style.fig(h=2.8)
        x = np.arange(len(d))
        al_, am_ = 100 * d["alert"].to_numpy(), 100 * d["alarm"].to_numpy()
        ax.bar(x, al_, color=AMB, label="Inside an ALERT")
        ax.bar(x, am_, bottom=al_, color=REDC, label="Inside an ALARM")
        for i, (u, v) in enumerate(zip(al_, am_, strict=True)):
            ax.text(i, u / 2, f"{u:.0f}%", ha="center", va="center", fontsize=7, color=style.TEXT)
            if v >= 2.5:
                ax.text(i, u + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white")
            ax.text(i, u + v + 1, f"{u + v:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([pd.Timestamp(m + "-01").strftime("%b") for m in d["month"]], fontsize=8)
        ax.set_ylim(0, 60)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
        ax.set_ylabel("Share of measured hours")
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
        return self.save(f, "level_month", "Hours by level, by month")

    def fig_signatures(self) -> str:
        s = self.t["signature_series"].copy()
        s["day"] = pd.to_datetime(s["day"], utc=True)
        cases = self.B["signature_cases"]
        f, axs = plt.subplots(1, 5, figsize=(8.8, 2.6))
        for x in axs:
            style.chart_style(x)
            x.tick_params(labelsize=6.5)
        g = s[s["wo_id"] == "PM-002016"]
        axs[0].plot(g["day"], g["value"], color=ML, lw=1)
        axs[0].set_title(
            f"{uline('Lubrication')}\nHF acceleration, g\n{cases['PM-002016']['asset']}", fontsize=7.5
        )
        b = next(x for x in self.bursts if x["role"] == "bearing")
        r = next(x for x in self.bursts if x["role"] == "bearing reference")
        for bb, c in ((r, VR), (b, ML)):
            ef, ea = np.array(bb["env_f"]), np.array(bb["env_a"])
            k = (ef > 1) & (ef <= 1200)
            axs[1].plot(ef[k], ea[k], color=c, lw=0.5)
        axs[1].axvline(b["fault_hz"]["bpfo"], color=AMB, ls=":", lw=0.9)
        axs[1].set_title(
            f"{uline('Bearing defect')}\nenvelope spectrum, Hz\nHMC-01-TSC",
            fontsize=7.5,
        )
        g = s[s["wo_id"] == "HMC-03-WO00042"]
        for k_, c in (
            ("ord_01_g2", ML),
            ("env_bpfo_1_g2", style.ACCENT),
            ("env_bpfi_1_g2", style.LIGHT_BLUE),
            ("env_bsf_1_g2", style.GREY),
        ):
            gg = g[g["measure"] == k_]
            axs[2].semilogy(gg["day"], gg["value"], color=c, lw=0.9)
        axs[2].set_title(
            f"{uline('Pump seal or impeller wear')}\n1x and fault bands, g²\n{cases['HMC-03-WO00042']['asset']}",
            fontsize=7.5,
        )
        g = s[s["wo_id"] == "HMC-04-WO00024"]
        for k_, c, ls in (("reference_median_bar", VR, "--"), ("pressure_median_bar", ML, "-")):
            gg = g[g["measure"] == k_]
            axs[3].plot(gg["day"], gg["value"], color=c, lw=1, ls=ls)
        axs[3].set_title(
            f"{uline('Rotary union leak')}\nTSC pump-on pressure, bar\n{cases['HMC-04-WO00024']['asset']}",
            fontsize=7.5,
        )
        g = s[s["wo_id"] == "HMC-01-WO00036"]
        axs[4].plot(g["day"], g["value"], color=ML, lw=1)
        axs[4].set_title(
            f"{uline('Spindle cooling fault')}\ntemperature residual, C\n{cases['HMC-01-WO00036']['asset']}",
            fontsize=7.5,
        )
        for x, wo in (
            (axs[0], "PM-002016"),
            (axs[2], "HMC-03-WO00042"),
            (axs[3], "HMC-04-WO00024"),
            (axs[4], "HMC-01-WO00036"),
        ):
            on, cf = pd.Timestamp(cases[wo]["onset"]), pd.Timestamp(cases[wo]["confirmed"])
            x.set_xlim(on - pd.Timedelta(days=30), cf + pd.Timedelta(days=1))
            x.axvspan(on, cf, color=AMB, alpha=0.15, lw=0)
            x.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            x.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
            x.relim()
            x.autoscale_view(scalex=False)
        f.tight_layout()
        return self.save(f, "signatures", "Signatures of the most common conditions")

    def section3(self) -> str:
        tr, sc, sg = self.B["train"], self.B["scoring"], self.B["signatures"]
        s = "<h2 id='s3'>3. Model Overview</h2>"
        s += bold("Sensors")
        s += p(
            "The ML Model and Vendor Rules read from 40 sensor points: four on each of the eight HMCs and two on each of the four compressors. Once an hour, each point records one second of vibration on three axes, sampled 25,600 times a second, and its housing temperature. A measurement is taken only in a steady state (spindles above 3,000 rpm and steady for 10 seconds; pumps and compressors running), so hours are comparable."
        )
        pts = pd.DataFrame(
            [
                ("HMC spindle", 2, "Front and rear bearing housings", "Triaxial", 16),
                ("TSC pump", 1, "Pump motor, drive end", "Triaxial", 8),
                ("Flood coolant pump", 1, "Pump motor, drive end", "Triaxial", 8),
                ("Compressor", 2, "Motor drive end; airend inlet", "Triaxial", 8),
            ],
            columns=["Asset type", "Points per unit", "Location", "Axes", "Total points"],
        )
        s += self.exhibit("Sensor points by asset type", style.table(pts, cls="data ctr2"))
        s += p(
            "The sensor node computes and reports 13 measurements (used by both layers): vibration velocity, high-frequency acceleration, bearing fault-frequency energies (six: the outer race, inner race and rolling elements, each at 1× and 2× its fault frequency), shaft-order energies (two: 1× and 2× running speed), crest factor, kurtosis and housing temperature. The ML Model also reads seven measurements (so 20 total inputs) of the machine's operating state from its controller, such as the spindle speed band and the cutting load, so it compares each hour with healthy hours under the same conditions."
        )
        s += bold("ML Model at a glance")
        s += style.table(
            pd.DataFrame(
                [
                    {
                        "What it reads": "Every hourly measurement on 40 sensor points: velocity, high-frequency acceleration, the bearing fault bands, shaft-order energies, crest factor, kurtosis and the temperature residual, with the operating measurements from the controller",
                        "What it learns": "Each point's healthy pattern from up to two years of its own history, across operating regimes, re-centered after each repair and sensor remount",
                        "What it produces": "An hourly anomaly score per point, the inputs that moved most, and for those scores beyond threshold, a level per asset (ALERT, ALARM) with a suspected condition",
                    }
                ]
            ),
            cls="data boxes",
        )
        s += p(
            "The ML Model is an autoencoder, a neural network trained separately for each sensor point on that point's healthy hours. Each hour, it compresses the 20 inputs through a narrow bottleneck layer then rebuilds them. Having learned only healthy machine readings, it rebuilds a healthy hour closely. A developing fault moves the measurements in ways the healthy hours never did, so the rebuild falls short. The size of that gap is the model's anomaly score, and the measurements with the largest gaps are reported as the leading inputs."
        )
        s += bold("Model outputs")
        s += p(
            "Each hour, the ML Model's anomaly score level for each asset (the highest of its points) appears in the maintenance queue as OK, ALERT, or ALARM, beside Vendor Rules' level. An ALERT opens a work request naming the suspected condition and a recommended inspection. During the evaluation year these ALERTs and ALARMs were recorded and not acted on."
        )
        s += style.screenshot("screenshots/queue_report.png", "Maintenance queue at the snapshot", up=1)
        s += "<h3 id='s32'>3.1 Training Data</h3>"
        s += bold("From signal to inputs")
        s += p(
            "Each hour's measurement starts as one second of acceleration, shown here from a healthy hour on an HMC spindle's front bearing. Its frequency spectrum shows the spindle's rotation as narrow peaks, which give the velocity and shaft-order measurements. The 2 to 10 kHz band, where bearing impacts first ring the housing, gives high-frequency acceleration, and demodulating it produces the envelope spectrum, checked for peaks at each bearing's defect rates (outer race, BPFO; inner race, BPFI; rolling elements, BSF). On this healthy hour, the envelope spectrum shows no peak at any defect rate."
        )
        s += self.exhibit(
            "One second of vibration, from raw signal to fault-frequency spectrum", self.fig_signal()
        )
        inp = self.t["inputs20"].set_index("input")["value"]
        load = next(LABELS[k].split()[0].lower() for k in REGIME_KEYS if inp.get(k) == 1)
        speed = next(LABELS[k].split()[0].lower() for k in BAND_KEYS if inp.get(k) == 1)
        speed = {"mid": "medium"}.get(speed, speed)
        s += p(
            f"Each of the 13 sensor measurements is calculated from these signals as one number for the hour, which is then compared to the point's normal level, so a healthy hour sits near zero. These measurements join the seven controller measurements which mark the hour's speed and load with an on/off flag (1 and 0; here, the machine was running with a {load} load at {speed} speed). Together they are the ML Model's 20 inputs, as shown below, which is calculated from the vibration data on the healthy hour as detailed above."
        )
        s += self.exhibit(
            "The Model's 20 Inputs: Representative Readings on a Healthy Hour", self.fig_inputs()
        )
        s += bold("Healthy hours")
        s += p(
            "The ML Model is trained on healthy hours which form the basis of its anomaly detection. An hour counts as healthy when the point was measured and no repair was under way, no work order was open, and, after any repair or sensor remount, the point's readings had settled at a new level. The 120 days before any confirmed condition are excluded, so that a fault developing before it was found does not enter training as a healthy hour."
        )
        s += p(
            "On a healthy hour, vibration velocity sits in ISO 20816 zone A or B (1.2 to 2.3 mm/s on spindles, rising with load), kurtosis is near 3, crest factor is 4.1 to 5.0, and the envelope spectrum has no peak at a fault frequency. Healthy levels shift with spindle speed and rise with load, and the controller measurements let the model learn each point's healthy pattern across them."
        )
        s += self.exhibit("Healthy measurements by speed band, spindle front points", self.fig_healthy_band())
        s += (
            "<p class='gap'>"
            + esc(
                f"The ML Model was trained on {tr['total']:,} healthy hours from July 2023 to June 2025 ({tr['min']:,} to {tr['max']:,} per point). The chart below shows the number of healthy training hours for each sensor point. The number above each bar counts the point's repairs and sensor remounts in the training years. Each one removes hours from training, mostly the 120 days before the condition that led to it, so points with higher repairs and remounts had lower healthy training hours."
            )
            + "</p>"
        )
        s += self.exhibit("Healthy training hours per sensor point", self.fig_train_hours())
        s += "<h3 id='s33'>3.2 Scoring Summary</h3>"
        s += p(
            f"The ML model's ALERT threshold is the 99.5th percentile of each point's healthy training scores. Six consecutive hours above it raise ALERT, or two hours above five times it, except on a pump during a low-pressure alarm. ALARM follows on an open ALERT when a severity criterion for the asset class holds (section 2). If a point stays above its threshold for 30 days with no open work order, no confirmed condition and no significant upward trend, its threshold is reset on those 30 days; {num(sc['rearms'])} points were reset in this way over the evaluation year."
        )
        rb = self.t["rebuild"]
        fg = (
            rb[rb["hour"] == "faulty"]
            .iloc[:13]
            .assign(g=lambda d: (d["z"] - d["rebuild"]) ** 2)
            .sort_values("g", ascending=False)
        )
        ratio = float(fg["score"].iloc[0] / fg["threshold"].iloc[0])
        s += p(
            f"On a healthy hour the model rebuilds every input to within about one unit, so the anomaly score stays low. A faulty hour is shown below, five weeks before HMC-01's TSC pump bearing was confirmed, where kurtosis, outer-race (BPFO) envelope energy and crest factor all peaked and the anomaly score rose to {ratio:.0f}x the threshold; those inputs led the work request to name an outer-race bearing defect."
        )
        s += self.exhibit(
            "Inputs Against the model's rebuild on Healthy and Faulty Hours", self.fig_rebuild()
        )
        s += self.exhibit("From score to level, one spindle point", self.fig_trace())
        s += p(
            f"Over the evaluation year the ML Model scored {sc['hours']:,} measured hours on 40 points. {pct(sc['outside'])} were outside any episode, {pct(sc['alert'])} inside an ALERT and {pct(sc['alarm'])} inside an ALARM, {self.B['alert']['alert_hours']['ML Model']:.0f} alert-hours per asset-month against Vendor Rules' {self.B['alert']['alert_hours']['Vendor Rules']:.0f}. It opened {sc['episodes']} ALERT episodes across {sc['assets']} assets and escalated {sc['alarms']} to ALARM."
        )
        s += self.exhibit("Score distribution, evaluation year", self.fig_score_dist())
        s += self.exhibit("Hours by level, by month", self.fig_level_month())
        s += p(
            "31% of the ML Model's ALERT hours led with shaft-order energies (imbalance, looseness, pump wear), 25% with the temperature residual, 28% with the fault bands, crest factor or kurtosis, 10% with velocity or high-frequency acceleration, and 6% with the controller measurements."
        )
        cs = self.t["class_summary"]
        s += self.exhibit("Scoring summary table", style.table(cs, {"Median episode length, days": 1}))
        s += "<h3 id='s34'>3.3 Condition Signatures</h3>"
        cn = sg["counts"]
        s += p(
            "The five most common conditions of the evaluation year each carry their own vibration and temperature signature."
        )
        lo, hi = sg["lub_hf_pct"]
        tlo, thi = sg["lub_temp_c"]
        s += ul(
            [
                f"[[b]]Lubrication ({cn.get('lubrication', 0)}):[[/b]] high-frequency energy rises {lo:.0f} to {hi:.0f}% over days and temperature rises {tlo:.0f} to {thi:.0f} °C, with no fault-frequency peaks.",
                f"[[b]]Bearing defect ({cn.get('bearing (unplanned)', 0) + cn.get('bearing (condition-based)', 0)}):[[/b]] envelope peaks at the fault frequency and its harmonics, then rising kurtosis; velocity rises only late.",
                f"[[b]]Pump seal or impeller wear ({cn.get('pump seal or impeller', 0)}):[[/b]] energy at 1× running speed and broadband energy rise together.",
                f"[[b]]Rotary union leak with coolant ingress ({cn.get('rotary union', 0)}):[[/b]] through-spindle coolant pressure falls 9 to 10.5% below its reference; the spindle points show little change.",
                f"[[b]]Spindle cooling fault ({cn.get('spindle cooling', 0)}):[[/b]] the temperature residual climbs.",
            ]
        )
        s += self.exhibit("Signatures of the most common conditions", self.fig_signatures())
        return s

    # ------------------------------------------------------------------ section 4
    def fig_type_bars(self) -> str:
        d = self.t["by_type"]
        g = (
            d.groupby("condition_type")
            .agg(n=("wo_id", "size"), ml=("ml", "mean"), vr=("vr", "mean"))
            .sort_values("n")
        )
        f, ax = style.fig(h=3.8, grid="x")
        y = np.arange(len(g))
        for dy, col, c, lab in ((0.2, "ml", ML, "ML Model"), (-0.2, "vr", VR, "Vendor Rules")):
            v = 100 * g[col].to_numpy()
            ax.barh(y + dy, v, height=0.38, color=c, label=lab)
            for yy, vv in zip(y + dy, v, strict=True):
                ax.text(vv + 1, yy, f"{vv:.0f}%", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t} ({n})" for t, n in zip(g.index, g["n"], strict=True)], fontsize=8)
        ax.set_xlim(0, 110)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
        ax.set_xlabel("Conditions caught")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        return self.save(f, "by_type", "Catch rate by condition type, evaluation year")

    def fig_own_signal(self) -> str:
        d = self.t["own_signal"]
        types = list(dict.fromkeys(d["type"]))
        f, ax = style.fig(h=3.2, grid="x")
        for i, t in enumerate(types):
            for m, c, dy in (("ML Model", ML, 0.15), ("Vendor Rules", VR, -0.15)):
                r = d[(d["type"] == t) & (d["method"] == m)].iloc[0]
                ax.plot(100 * r["alerted"], i + dy, "o", color=c, ms=7)
                ax.plot(100 * r["healthy"], i + dy, "o", mfc="white", mec=c, ms=7)
                ax.plot([100 * r["healthy"], 100 * r["alerted"]], [i + dy] * 2, color=c, lw=0.8)
        ax.set_yticks(range(len(types)))
        ax.set_yticklabels(types, fontsize=8)
        ax.set_xlabel("% alerted")
        hd = [
            Line2D([], [], marker="o", ls="", color=ML),
            Line2D([], [], marker="o", ls="", color=VR),
            Line2D([], [], marker="o", ls="", mfc="white", mec=style.DARK_GREY),
        ]
        ax.legend(
            hd,
            [
                "ML Model, while the signal was present",
                "Vendor Rules, while the signal was present",
                "Hollow: matched healthy windows",
            ],
            frameon=False,
            fontsize=7.5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
        )
        return self.save(f, "own_signal", "Alerts on each condition type against matched healthy windows")

    def fig_context(self) -> str:
        d = self.t["context"]
        ctx = ["new program", "known program", "after re-centering", "benign event", "other"]
        f, ax = style.fig(h=3.0)
        x = np.arange(len(ctx))

        def lab_(xs: Any, vs: Any) -> None:
            for xx, vv in zip(xs, vs, strict=True):
                ax.text(xx, vv + 1, f"{vv:.0f}%", ha="center", va="bottom", fontsize=6.8)

        for j, (m, c) in enumerate((("ML Model", ML), ("Vendor Rules", VR))):
            g = d[d["method"] == m].set_index("context").reindex(ctx)
            ax.bar(
                x + (j - 1) * 0.27,
                100 * g["share"],
                width=0.27,
                color=c,
                label=f"{m}: share of its false alerts",
            )
            lab_(x + (j - 1) * 0.27, 100 * g["share"])
        h = d[d["method"] == "ML Model"].set_index("context").reindex(ctx)
        ax.bar(
            x + 0.27,
            100 * h["healthy"],
            width=0.27,
            color="white",
            edgecolor=style.DARK_GREY,
            hatch="//",
            label="Share of healthy hours",
        )
        lab_(x + 0.27, 100 * h["healthy"])
        ax.set_xticks(x)
        ax.set_xticklabels(["Compressors" if c == "other" else c.capitalize() for c in ctx], fontsize=8)
        ax.set_ylim(0, 88)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
        ax.set_ylabel("Share")
        ax.legend(frameon=False, fontsize=7.5)
        return self.save(f, "context", "False alerts by context")

    def fig_false_causes(self) -> str:
        d = self.t["false_causes"]
        order = [
            (LEVEL_OUT, ML, "white"),
            (REGIME_MISSING, ML, "white"),
            ("Operating state outside training", style.LIGHT_BLUE, style.TEXT),
            ("Post-repair re-centering", style.ACCENT, "white"),
            ("Transient", "#C9C9C9", style.TEXT),
            ("Unexplained", "#8F8F8F", "white"),
        ]
        f, ax = style.fig(h=2.6, grid="x")
        methods = ["Vendor Rules", "ML Model"]
        top = d.groupby("method")["alert_hours"].sum().max()
        for y, m in enumerate(methods):
            g = d[d["method"] == m].groupby("cause")["alert_hours"].sum()
            tot = float(g.sum())
            left = 0.0
            for cause, col, tc in order:
                v = float(g.get(cause, 0.0))
                if not v:
                    continue
                ax.barh(y, v, left=left, color=col, height=0.55)
                if v >= 0.035 * top:
                    ax.text(left + v / 2, y, pct(v / tot), ha="center", va="center", fontsize=8, color=tc)
                left += v
            ax.text(left + 0.01 * top, y, f"{left:,.0f}", va="center", fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlim(0, top * 1.12)
        ax.set_xlabel("False alert-hours")
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        hd = [Patch(fc=c) for _, c, _ in order[1:]]
        labs = [
            "New normal level outside training reference",
            *[c for c, _, _ in order[2:]],
        ]
        ax.legend(
            hd, labs, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3
        )
        return self.save(f, "false_causes", "False alert-hours by cause, evaluation year")

    def section4(self) -> str:
        a, fo = self.B["alert"], self.B["folds"]
        s = "<h2 id='s4'>4. Model Evaluation</h2>"
        s += "<h3 id='s41'>4.1 Model Selection</h3>"
        cand = (
            pd.read_parquet(DATA / "candidates.parquet") if (DATA / "candidates.parquet").exists() else None
        )
        skill = (
            float(cand.loc[cand["Selected"] == "✔", "Skill above chance"].iloc[0])
            if cand is not None
            else np.nan
        )
        s += p(
            "Four model types (PCA, Mahalanobis distance, isolation forest and an autoencoder) were trained per sensor point on five input sets (20 models trained), from the 13 sensor measurements to a full set of 52 inputs with the entire spectrum and envelope bands. Each model was trained with one threshold per point or one per spindle speed band. Each model was then scored and selected by a fixed rule: take the highest catch rate above chance (the share an alert would catch if it fell at random, given how much of the time it was open) at Vendor Rules' false-alert rate; and, among those within two caught conditions of the best, choose the simplest model."
        )
        s += p(
            f"The autoencoder with the 13 measurements was selected (skill of {skill:.3f}); the wider input sets did not improve it. A full detail of what each design choice added in the training round is in the appendix."
        )
        if cand is not None:
            cand.loc[cand["Selected"] == "✔", "Model"] = (
                "Autoencoder [16-6-16], 200 epochs (retuned to 16-4-16)"
            )
            s += self.exhibit(
                "Top Model Candidates After Training",
                style.table(
                    cand,
                    {
                        "Catch at budget": 2,
                        "Chance": 2,
                        "Skill above chance": 3,
                        "Caught at operating point": 1,
                        "False episodes per asset-month": 3,
                    },
                    cls="data ctrlast",
                ),
            )
        s += "<h3 id='s42'>4.2 Accuracy and Validation</h3>"
        s += p("The evaluation year scores the ML Model and compares it to Vendor Rules' performance.")
        s += p(
            f"On the evaluation year, the ML Model caught {pct(a['strict']['ML Model'] / 32)} of conditions with {n1(fo['eval_warn'])} days of warning on its catches at {a['false']['ML Model']:.3f} false episodes per asset-month, against {pct(fo['strict'] / fo['conditions'])} with {n1(fo['warn'])} days of warning on its catches at {fo['false']:.3f} false episodes per asset-month on the training folds. It caught fewer conditions in evaluation than training, mostly on lubrication episodes, discussed in 4.3."
        )
        s += (
            "<p>"
            + esc(
                "The ML Model raised fewer false alerts than Vendor Rules in every period, on both training folds and in all four quarters of the evaluation year. It caught a higher share of conditions on both folds and in Q2 and Q3, and a lower share in Q1 (44% against 78%, the quarter the "
            )
            + "<a href='monitoring_report.html'>monitoring report</a>"
            + esc(" flags) and Q4 (67% against 83%).")
            + "</p>"
        )
        per = self.t["periods"]
        per = per[per["definition"] == "E7"]
        rows = []
        for m in ("ML Model", "Vendor Rules"):
            r1 = {"Method": m, "Measure": "Catch rate"}
            r2_ = {"Method": m, "Measure": "False alerts per asset-month"}
            r3 = {"Method": m, "Measure": "Conditions"}
            ev = {"strict": a["strict"][m], "false": a["false"][m], "n": 32}
            for col, lab in (
                ("fold 1", "Fold 1"),
                ("fold 2", "Fold 2"),
                ("year", "Evaluation year"),
                ("evaluation year Q1", "Q1"),
                ("evaluation year Q2", "Q2"),
                ("evaluation year Q3", "Q3"),
                ("evaluation year Q4", "Q4"),
            ):
                if col == "year":
                    r1[lab], r2_[lab], r3[lab] = pct(ev["strict"] / 32), f"{ev['false']:.3f}", "32"
                    continue
                g = per[(per["period"] == col) & (per["method"] == m)].iloc[0]
                r1[lab], r2_[lab], r3[lab] = (
                    pct(g["strict"] / g["conditions"]),
                    f"{g['false (truth-based) per asset-month']:.3f}",
                    str(int(g["conditions"])),
                )
            rows += [r1, r2_, r3]
        s += self.exhibit("Results by period", style.table(pd.DataFrame(rows)))
        s += p(
            "Over the evaluation year, the ML Model matched or beat Vendor Rules on every condition type except pump seal or impeller wear (four of six against five of six), and caught all six bearing defects, one unplanned bearing defect and one spindle cooling fault more than Vendor Rules. Lubrication, the most common condition, was caught at 60% by both methods."
        )
        s += self.exhibit("Catch rate by condition type, evaluation year", self.fig_type_bars())
        os_ = self.t["own_signal"].set_index(["type", "method"])
        lv, sc_ = os_.loc[("lubrication", "Vendor Rules")], os_.loc[("spindle cooling", "ML Model")]
        s += p(
            f"A method can be credited with a catch simply because its alerts are open much of the time, so each condition type was checked against how often the same method alerted when nothing was wrong. In the chart below, the filled marker shows how often a method had an alert open while the condition was developing, and the hollow marker how often it had one open over matched healthy windows: stretches of the same length on the same asset with no condition present. The ML Model's alerts tracked bearing defects, lubrication and pump wear far above its healthy-window rate, while Vendor Rules' lubrication alerts barely exceeded theirs ({pct(lv['alerted'])} against {pct(lv['healthy'])}); on spindle cooling, the ML Model did no better than its healthy-window rate ({pct(sc_['alerted'])} against {pct(sc_['healthy'])})."
        )
        s += self.exhibit(
            "Alerts on each condition type against matched healthy windows", self.fig_own_signal()
        )
        s += "<h3 id='s43'>4.3 False Alerts & Misses</h3>"
        cx = self.t["context"].set_index(["method", "context"])

        def cs(m: str, k: str, col: str = "share") -> str:
            return pct(cx.at[(m, k), col])

        s += p(
            f"Measured against the share of healthy hours in each context, the ML Model's false alerts fell less often on new programs than Vendor Rules' ({cs('ML Model', 'new program')} of its false alerts against {cs('Vendor Rules', 'new program')}, with new programs making up {cs('ML Model', 'new program', 'healthy')} of healthy hours), where unfamiliar cutting loads push vibration past Vendor Rules' fixed limits. A higher share of the ML Model's false alerts fell on known programs ({cs('ML Model', 'known program')} against {cs('Vendor Rules', 'known program')}), in line with their {cs('ML Model', 'known program', 'healthy')} of healthy hours, and on the compressors ({cs('ML Model', 'other')}, where Vendor Rules raised none), also in line with their hours. Both methods' raised false alerts at elevated rates in the two weeks after a repair re-centered a point, at about twice that period's {cs('ML Model', 'after re-centering', 'healthy')} share of healthy hours."
        )
        s += self.exhibit("False alerts by context", self.fig_context())
        fc = self.t["false_causes"]
        fm, fv = fc[fc["method"] == "ML Model"], fc[fc["method"] == "Vendor Rules"]
        lvl = fm[fm["cause"] == LEVEL_OUT]
        n_ = fm["cause"].value_counts()
        s += p(
            f"The ML Model raised half the false-alert hours as Vendor Rules ({fm['alert_hours'].sum():,.0f} vs. {fv['alert_hours'].sum():,.0f}). {pct(lvl['alert_hours'].sum() / fm['alert_hours'].sum())} of the ML Model's false alert-hours came from sensor points whose healthy level moved outside the range it was trained on. {len(lvl)} of its {len(fm)} false alerts, across {num(lvl['lead_point'].nunique())} sensor points, opened this way. Of the rest, {n_.get('Transient', 0)} were short crossings above threshold and {n_.get('Unexplained', 0)} had no identifiable cause."
        )
        s += self.exhibit("False alert-hours by cause, evaluation year", self.fig_false_causes())
        mi = self.t["misses"]
        cnt = mi["Reason"].value_counts()
        s += p(f"[[b]]The ML Model missed {num(len(mi))} of the 32 conditions:[[/b]]")
        s += ul(
            [
                f"{num(cnt.get('removed as a coincidence', 0) + cnt.get('inside an episode opened more than 120 days earlier', 0), cap=True)} fell under an ALERT that had opened more than a week before the condition showed any sign on the machine, so the ALERT could not have been prompted by it and they are not counted.",
                f"{num(cnt.get('precursor under the threshold', 0), cap=True)} had a score that crossed the threshold without staying above it: a lubrication episode, a wear cycle on HMC-06's flood pump that the model had learned as healthy in training, and a wear cycle on HMC-06's TSC pump.",
                f"{num(cnt.get('point re-centering', 0), cap=True)} came while the point was re-centering after a repair, with no level to score against: a compressor lubrication episode.",
            ]
        )
        mt = mi[["Asset", "Condition", "Confirmed", "Reason"]].copy()
        mt["Confirmed"] = [loc_date(x, "%Y-%m-%d") for x in mt["Confirmed"]]
        mt["Reason"] = mt["Reason"].map(REASON).fillna(mt["Reason"])
        s += self.exhibit("Conditions the ML Model missed", style.table(mt))
        s += "<h3 id='s44'>4.4 Limitations</h3>"
        c = self.B["conditions"]
        s += ul(
            [
                f"[[b]]Small counts.[[/b]] 32 conditions and {num(self.B['failures']['n'])} failures; a difference of two or three conditions could reverse in another year.",
                f"[[b]]Points stuck in ALERT.[[/b]] The HMC-02 spindle points' temperature residual ran outside its training range after {self.B['train']['min']} training hours and held both points in ALERT for most of the year, hiding two conditions. A clamp on the residual fixed HMC-02 but raised stuck points elsewhere on the folds and was set aside.",
                f"[[b]]Conditions the sensors cannot see.[[/b]] {c['other']} of the year's {c['total']} conditions are outside this comparison: {num(c['no_sensors'])} on machines without sensors, and {c['other'] - c['no_sensors']} that left no trace in the sensor data (trips, crashes, stoppages, compressor oil cooler faults and leaks visible only in controller pressure).",
                "[[b]]Diagnosis limits.[[/b]] One alert names one condition, so a condition sharing an alert with another on the same asset is named only if it leads. On pumps, a rise at running speed alone cannot separate impeller wear from imbalance. Rotary-union leaks dropped through-spindle coolant pressure 9 to 10.5%; the drop names a leak only when no earlier rule fires, so one of the two caught was named bearing.",
                f"[[b]]Resets during degradation.[[/b]] {num(self.B['scoring']['rearms_degrading'], cap=True)} of the {num(self.B['scoring']['rearms'])} threshold resets in the year came while the point was degrading. None cost a catch, but each raised a threshold over a developing fault.",
                "[[b]]Wear learned as healthy.[[/b]] A wear cycle on HMC-06's flood pump was in the training years' healthy hours, so the model did not alert on its return.",
                "[[b]]Label noise.[[/b]] CMMS failure codes are wrong on about 15% of unplanned repairs, and the same noise limits the diagnosis match, which is scored against those codes.",
                "[[b]]Chiller faults[[/b]] have no controller fault contact, so abrupt faults show only in temperature.",
            ]
        )
        return s

    def appendix(self) -> str:
        s = "<h2 id='app'>Appendix</h2><h3 id='app1'>Indicators</h3>"
        ind = pd.DataFrame(
            [
                (LABELS[c], DESC[c], "Sensor" if not c.startswith(("regime", "band")) else "Controller")
                for c in LABELS
                if c in DESC
            ],
            columns=["Input", "What it measures", "Source"],
        )
        s += style.table(ind)
        s += "<h3 id='app2'>Platform</h3>"
        s += style.table(pd.DataFrame(PLATFORM, columns=["Component", "Role"]))
        dc = pd.read_parquet(DATA / "design.parquet") if (DATA / "design.parquet").exists() else None
        if dc is not None:
            s += "<h3 id='app4'>Model training</h3>"
            s += self.exhibit(
                "What each design choice added to model training performance",
                style.table(dc, {"Caught": 1, "False alerts per asset-month": 3, "Stuck points": 1}),
            )
        s += "<h3 id='app3'>Full timeline</h3>"
        allassets = sorted(
            self.t["train_hours"]["point_id"]
            .map(lambda p_: p_.rsplit("-", 1)[0] if p_.split("-")[-1] in ("F", "R", "MDE", "AE") else p_)
            .unique()
        )
        s += self.exhibit(
            "Alerts, inspections and failures over the evaluation year, all 28 assets",
            self.fig_timeline(allassets, "timeline_all", "Alerts, inspections and failures, all assets"),
            "Spindles and compressors carry two sensor points; each lane shows the higher level of the two at each hour.",
        )
        return s

    def build(self) -> str:
        body = self.section1() + self.section2() + self.section3() + self.section4() + self.appendix()
        toc = [
            ("s1", "1. Condition Monitoring Overview"),
            ("s2", "2. Model Results"),
            ("s3", "3. Model Overview"),
            ("s32", "3.1 Training Data", "sub"),
            ("s33", "3.2 Scoring Summary", "sub"),
            ("s34", "3.3 Condition Signatures", "sub"),
            ("s4", "4. Model Evaluation"),
            ("s41", "4.1 Model Selection", "sub"),
            ("s42", "4.2 Accuracy and Validation", "sub"),
            ("s43", "4.3 False Alerts & Misses", "sub"),
            ("s44", "4.4 Limitations", "sub"),
            ("app", "Appendix"),
        ]
        page = style.report_shell(TITLE, "", body, toc).replace("</style>", EXTRA_CSS + "</style>")
        style.write("reports/model_report.html", page)
        return page


LABELS = {
    "vel_rms_mm_s": "Velocity",
    "hf_rms_g": "HF acceleration",
    "env_bpfo_1_g2": "BPFO 1x",
    "env_bpfo_2_g2": "BPFO 2x",
    "env_bpfi_1_g2": "BPFI 1x",
    "env_bpfi_2_g2": "BPFI 2x",
    "env_bsf_1_g2": "BSF 1x",
    "env_bsf_2_g2": "BSF 2x",
    "ord_01_g2": "1x order",
    "ord_02_g2": "2x order",
    "crest_factor": "Crest factor",
    "kurtosis": "Kurtosis",
    "temp_resid_c": "Temperature residual",
    "regime_light": "Light load",
    "regime_medium": "Medium load",
    "regime_heavy": "Heavy load",
    "regime_running": "Running",
    "band_low": "Low speed",
    "band_mid": "Mid speed",
    "band_high": "High speed",
}
DESC = {
    "vel_rms_mm_s": "Vibration velocity RMS, 10 to 1,000 Hz (mm/s)",
    "hf_rms_g": "High-frequency acceleration RMS, 2 to 10 kHz (g)",
    "env_bpfo_1_g2": "Envelope energy at the outer-race fault frequency",
    "env_bpfo_2_g2": "Envelope energy at twice the outer-race fault frequency",
    "env_bpfi_1_g2": "Envelope energy at the inner-race fault frequency",
    "env_bpfi_2_g2": "Envelope energy at twice the inner-race fault frequency",
    "env_bsf_1_g2": "Envelope energy at the rolling-element fault frequency",
    "env_bsf_2_g2": "Envelope energy at twice the rolling-element fault frequency",
    "ord_01_g2": "Spectral energy at 1x running speed",
    "ord_02_g2": "Spectral energy at 2x running speed",
    "crest_factor": "Peak over RMS of the acceleration",
    "kurtosis": "Impulsiveness of the acceleration",
    "temp_resid_c": "Housing temperature less its expected value for the operating state (C)",
    "regime_light": "Cutting load regime: light",
    "regime_medium": "Cutting load regime: medium",
    "regime_heavy": "Cutting load regime: heavy",
    "regime_running": "Running without a cutting regime (pumps, compressors)",
    "band_low": "Spindle speed band: low",
    "band_mid": "Spindle speed band: mid",
    "band_high": "Spindle speed band: high",
}
PANEL_DAYS = 120
LEVEL_OUT = "Healthy level outside training range"
REGIME_MISSING = "Load regime missing from baseline"
REGIME_KEYS = ("regime_light", "regime_medium", "regime_heavy", "regime_running")
BAND_KEYS = ("band_low", "band_mid", "band_high")
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
TYPE_LABEL = {
    "lubrication": "Lubrication",
    "bearing (unplanned)": "Bearing failure",
    "bearing (condition-based)": "Bearing defect",
    "pump seal or impeller": "Pump wear",
    "rotary union": "Rotary union leak",
    "spindle cooling": "Spindle cooling",
    "imbalance": "Imbalance",
    "looseness": "Looseness",
}
PLATFORM = [
    ("Azure Data Lake Storage Gen2", "Raw sensor, controller, CMMS and ERP exports"),
    ("Unity Catalog", "Catalog, schemas, external location and access control"),
    ("Lakeflow Declarative Pipelines", "Bronze, silver and gold tables with expectations"),
    ("Lakeflow Jobs", "Training, scoring, levels and work requests, evaluation, monitoring"),
    (
        "MLflow with the Unity Catalog model registry",
        "The 40 point models as one registered model, alias champion",
    ),
    ("Declarative Automation Bundles deployed through GitHub Actions", "Development and production targets"),
]
REASON = {
    "removed as a coincidence": "ALERT opened more than a week before the condition with no sign of it",
    "inside an episode opened more than 120 days earlier": "Inside an ALERT opened more than 120 days earlier",
    "precursor under the threshold": "Score crossed the threshold without staying above it",
    "point re-centering": "Point re-centering after a repair",
}
EXTRA_CSS = """
.exnote { font-size: 12px; color: var(--muted); margin: 2px 0 14px; }
table.boxes td { width: 33%; background: #F7F9FB; }
table.even2 { table-layout: fixed; font-size: 15px; }
table.even2 th, table.even2 td { width: 50%; }
p.gap { margin-top: 28px; }
.chart-title.panel-title { color: var(--text); margin-top: 26px; }
.panel-sub { color: var(--text); font-size: 14px; text-align: center; margin: 2px 0 0; }
table.data.ctr2 th:nth-child(2), table.data.ctr2 td:nth-child(2) { text-align: center; }
table.data.ctr2, table.data.boxes { font-size: 15px; }
table.data.even2 th { text-align: center; }
table.data.ctrlast th:last-child, table.data.ctrlast td:last-child { text-align: center; }
"""


def build() -> str:
    page = Report().build()
    LOG.info("model report built: %s", style.DOCS / "reports" / "model_report.html")
    return page


if __name__ == "__main__":
    log.configure()
    build()

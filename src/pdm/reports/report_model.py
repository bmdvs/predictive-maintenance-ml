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
AMB, REDC = style.AMBER, style.RED
TZ = "America/Chicago"
TITLE = "Predictive Maintenance: ML Model Report"


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def n1(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}"


def esc(s: str) -> str:
    return html.escape(s, quote=False).replace("[[b]]", "<b>").replace("[[/b]]", "</b>")


def p(s: str) -> str:
    return f"<p>{esc(s)}</p>"


def ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def bold(s: str) -> str:
    return f"<h4>{html.escape(s)}</h4>"


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
        f, ax = plt.subplots(figsize=(8.8, 4.4))
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6.2)
        pilot, existing, new, plain = "#D6E2EC", "#E6E6E6", "#F3E3C6", "white"

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
                    ls="--" if dashed else "-",
                )
            )

        ax.add_patch(Rectangle((0.15, 3.6), 2.1, 2.45, fc="none", ec=style.ACCENT, lw=1.0, ls=":"))
        ax.text(1.2, 5.92, "Sensor pilot", ha="center", fontsize=7.5, color=style.ACCENT, fontweight="bold")
        box(0.3, 5.15, 1.8, 0.62, "HMCs (8)", "Bottleneck", pilot)
        box(0.3, 4.15, 1.8, 0.62, "Compressors (4)", "", pilot)
        for i, (t, n) in enumerate((("VMCs", 16), ("Lathes", 10), ("Mill-turns", 4), ("Grinders", 2))):
            box(0.3, 3.0 - i * 0.72, 1.8, 0.55, f"{t} ({n})", "", plain, bold_=False)
        box(2.75, 4.65, 2.55, 0.8, "Wireless sensor data", "Vibration and temperature on 40 points", plain)
        arrow((2.12, 5.15), (2.73, 4.8))
        arrow((2.12, 4.15), (2.73, 4.5))
        box(5.85, 5.35, 2.0, 0.75, "ML Model", "One ML model per sensor point", new, dashed=True)
        box(5.85, 3.95, 2.0, 0.75, "Vendor Rules", "Threshold alerts", existing)
        arrow((5.32, 4.85), (5.83, 5.3), dashed=True)
        arrow((5.32, 4.45), (5.83, 4.0))
        box(8.2, 3.95, 1.75, 0.75, "Maintenance queue", "", plain)
        arrow((7.87, 3.95), (8.18, 3.95))
        ax.text(
            6.85,
            5.88,
            "Recorded, not acted on, July 2025 to June 2026",
            ha="center",
            fontsize=7,
            color=style.DARK_GREY,
            style="italic",
        )
        handles = [Patch(fc=c, ec=style.DARK_GREY, lw=0.6) for c in (pilot, existing, new)]
        f.legend(
            handles,
            ["Sensor pilot", "Existing Layer", "New Layer"],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        f.tight_layout(rect=(0, 0.06, 1, 1))
        return self.save(f, "layers", "Machines and Equipment, Data Sources & Monitoring Layers")

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
            f"The shop runs 40 CNC machines and four compressors that supply the plant's air. Its eight HMCs are the bottleneck: they carry {pct(b['hmc_share'], 1)} of the shop's cutting hours at {pct(b['hmc_util'])} utilization, against {round(100 * ou[0])} to {round(100 * ou[-1])}% for the other machine classes, and an hour of HMC downtime costs {b['margin']:.0f} USD in contribution margin, {b['ratio']:.2f} times the average for the rest of the fleet."
        )
        s += p(
            "In July 2023 the shop began a pilot of sensor-based condition monitoring on the HMC cell and the compressors: wireless sensors on 40 points, monitored as 28 assets (each HMC's spindle, through-spindle coolant (TSC) pump and flood coolant pump, and each compressor)."
        )
        s += p(
            "Through the pilot, the sensor data has been monitored by the [[b]]Vendor Rules[[/b]] layer, the sensor vendor's threshold alerts. An alert is raised at the point's baseline plus 25 percent of the ISO 20816 zone B width on vibration velocity or high-frequency acceleration, and an alarm at plus 50 percent; a bearing fault band 6 dB above baseline for three hours raises an alert; sensor temperatures raise an alert and an alarm at fixed limits by asset class. Baselines are set over 21 days after installation and after each repair."
        )
        s += p(
            "The pilot's second phase adds the [[b]]ML Model[[/b]] layer: one machine learning model per sensor point, trained on that point's healthy hours, that scores each hour by how far its vibration and temperature depart from that history. It was trained on July 2023 to June 2025, frozen, and run beside Vendor Rules from July 2025 to June 2026. [[b]]This report evaluates the ML Model against Vendor Rules over that evaluation year.[[/b]]"
        )
        s += self.exhibit("Machines and Equipment, Data Sources & Monitoring Layers", self.fig_layers())
        s += bold("Conditions in the Evaluation Year")
        s += p(
            f"{c['total']} conditions were confirmed from July 2025 to June 2026. {c['in_scope']} were on the instrumented assets and showed in the sensor data; the comparison in section 2 is on those. The other {c['other']} were on machines without sensors ({c['no_sensors']}), were compressor oil cooler faults ({c['cooler']}), showed only in controller data ({c['ru_leak']} rotary union leaks without coolant ingress), or had no precursor in any data ({c['no_precursor']} trips, crashes and stoppages)."
        )
        s += self.exhibit("Sensor-detectable conditions by type, evaluation year", self.fig_types())
        return s

    # ------------------------------------------------------------------ section 2
    def fig_levels(self) -> str:
        a, al = self.B["alert"], self.B["alarm"]
        f, axs = plt.subplots(2, 3, figsize=(8.8, 5.0))
        for x in axs.ravel():
            style.chart_style(x)
        names = ["ML Model", "Vendor Rules"]
        cols = [ML, VR]
        x = axs[0, 0]
        for i, m in enumerate(names):
            k = a["strict"][m]
            x.bar(i, k, color=cols[i])
            x.bar(i, 32 - k, bottom=k, color=cols[i], alpha=0.25)
            x.text(i, k / 2, f"{k}", ha="center", va="center", color="white", fontweight="bold")
            x.text(i, 32 + 0.5, "of 32", ha="center", fontsize=8)
        x.set_xticks([0, 1])
        x.set_xticklabels(names)
        x.set_title("ALERT: strict catches", fontsize=9.5)
        x = axs[0, 1]
        vals = [a["warn_both"][m] for m in names]
        x.bar(names, vals, color=cols)
        [x.text(i, v + 0.5, n1(v), ha="center", fontsize=9) for i, v in enumerate(vals)]
        x.set_title(f"Median attributable warning, days\n({a['both']} conditions both caught)", fontsize=9.5)
        x = axs[0, 2]
        vals = [a["false"][m] for m in names]
        x.bar(names, vals, color=cols)
        [x.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9) for i, v in enumerate(vals)]
        x.set_title("False alerts per asset-month", fontsize=9.5)
        x = axs[1, 0]
        vals = [al["share"][m] for m in names]
        x.bar(names, [100 * v for v in vals], color=cols)
        [x.text(i, 100 * v + 1, pct(v), ha="center", fontsize=9) for i, v in enumerate(vals)]
        x.set_title("ALARM: share with a confirmed condition", fontsize=9.5)
        x = axs[1, 1]
        vals = [al["warn"][m] for m in names]
        x.bar(names, vals, color=cols)
        [x.text(i, v + 0.5, n1(v), ha="center", fontsize=9) for i, v in enumerate(vals)]
        x.set_title(
            f"Median days from ALARM to confirmation\n({al['warn']['n']} conditions both escalated)",
            fontsize=9.5,
        )
        x = axs[1, 2]
        vals = [al["false"][m] for m in names]
        x.bar(names, vals, color=cols)
        [x.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9) for i, v in enumerate(vals)]
        x.set_title("False ALARMs per asset-month", fontsize=9.5)
        for x in axs.ravel():
            x.tick_params(axis="x", labelsize=8)
        f.tight_layout()
        return self.save(f, "levels", "Results by level, ML Model and Vendor Rules")

    def fig_timeline(self, assets: list[str], name: str, title: str) -> str:
        tl = self.t["timeline"]
        tl = tl.assign(start=pd.to_datetime(tl["start"], utc=True), end=pd.to_datetime(tl["end"], utc=True))
        a0, a1 = pd.Timestamp("2025-07-01 05:00", tz="UTC"), pd.Timestamp("2026-07-01 05:00", tz="UTC")
        machines = sorted({a[:6] for a in assets}, key=lambda m: (m.startswith("CMP"), m))
        if any(a.startswith("CMP") for a in assets):
            machines = [m for m in machines if not m.startswith("CMP")] + ["Compressors"]
        rows = []
        for m in machines:
            rows.append(("head", m))
            for a in [x for x in assets if (x[:6] == m or (m == "Compressors" and x.startswith("CMP")))]:
                rows.append(("asset", a))
        h = 0.32 * sum(2 if k == "asset" else 1 for k, _ in rows) + 0.9
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
            for lane, dy in (("ML Model", 0.0), ("Vendor Rules", -0.42)):
                g = tl[(tl["asset"] == v) & (tl["lane"] == lane)]
                for r in g.itertuples():
                    if r.kind in ("ALERT", "ALARM"):
                        c = AMB if r.kind == "ALERT" else REDC
                        ax.barh(
                            y + dy,
                            (r.end - r.start).total_seconds() / 86400,
                            left=(r.start - a0).total_seconds() / 86400,
                            height=0.34,
                            color=c,
                            lw=0,
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
                        )
                ticks.append(y + dy)
                labels.append(f"{v}  {'ML' if lane == 'ML Model' else 'VR'}")
            ev = tl[(tl["asset"] == v) & (tl["lane"] == "both")]
            for r in ev.itertuples():
                xx = (r.start - a0).total_seconds() / 86400
                if r.kind == "failure":
                    ax.plot(xx, y - 0.21, "x", ms=7, mew=1.6, color=style.TEXT)
                else:
                    ax.plot(xx, y - 0.21, "D", ms=4, mfc=style.GREEN, mec=style.GREEN)
            y -= 1.15
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=7)
        months = pd.date_range("2025-07-01", "2026-06-01", freq="MS")
        ax.set_xticks(
            [(pd.Timestamp(m, tz=TZ).tz_convert("UTC") - a0).total_seconds() / 86400 for m in months]
        )
        ax.set_xticklabels([m.strftime("%b") for m in months], fontsize=8)
        ax.set_xlim(0, (a1 - a0).total_seconds() / 86400)
        ax.set_ylim(y + 0.6, 0.6)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.xaxis.grid(True, color="#EEEEEE")
        ax.set_axisbelow(True)
        hd = [
            Patch(fc=AMB),
            Patch(fc=REDC),
            Line2D([], [], marker="o", ls="", mfc="white", mec=style.TEXT),
            Line2D([], [], marker="o", ls="", mfc=style.TEXT, mec=style.TEXT),
            Line2D([], [], marker="D", ls="", mfc=style.GREEN, mec=style.GREEN),
            Line2D([], [], marker="x", ls="", color=style.TEXT, mew=1.6),
        ]
        f.legend(
            hd,
            [
                "ALERT",
                "ALARM",
                "Inspection, nothing found",
                "Inspection, condition confirmed",
                "PM finding",
                "Failure",
            ],
            loc="lower center",
            ncol=6,
            frameon=False,
            fontsize=7.5,
        )
        f.tight_layout(rect=(0, 0.04, 1, 1))
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
            offs = [(10, -4), (6, -16), (14, 8)] if m == "ML Model" else [(-10, 8), (10, 0), (10, 4)]
            for r, lab, o in zip(ends, labs, offs, strict=True):
                ax.annotate(
                    lab,
                    (r["false per asset-month (truth-based)"], 100 * r["strict"] / 32),
                    textcoords="offset points",
                    xytext=o,
                    fontsize=7.5,
                    ha="left" if o[0] > 0 else "right",
                    color=c,
                )
        ax.set_xlabel("False alerts per asset-month")
        ax.set_ylabel("Conditions strictly caught, % of 32")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(50, 76)
        ax.legend(frameon=False, loc="upper right")
        return self.save(f, "curves", "Conditions caught against false episodes per asset-month")

    def fig_grid(self) -> str:
        g = self.t["diag_grid"]
        x = pd.crosstab(g["d2"], g["family"])
        rows = [r for r in x.index if r != "unclassified"] + (
            ["unclassified"] if "unclassified" in x.index else []
        )
        x = x.loc[rows]
        f, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.imshow(x.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(1, x.to_numpy().max()))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = x.iat[i, j]
                if v:
                    ax.text(
                        j,
                        i,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if v >= x.to_numpy().max() * 0.6 else style.TEXT,
                        fontweight="bold" if x.index[i] == x.columns[j] else "normal",
                    )
        ax.set_xticks(range(x.shape[1]))
        ax.set_xticklabels([c.capitalize() for c in x.columns], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(x.shape[0]))
        ax.set_yticklabels([c.capitalize() for c in x.index], fontsize=8)
        ax.set_xlabel("Confirmed condition", fontsize=9)
        ax.set_ylabel("Suspected condition", fontsize=9)
        f.tight_layout()
        return self.save(f, "diagnosis", "Suspected condition against confirmed finding")

    def section2(self) -> str:
        a, al, ins, q, d = (
            self.B["alert"],
            self.B["alarm"],
            self.B["inspection"],
            self.B["quiet"],
            self.B["diag"],
        )
        fl, fr = self.B["failures"], self.t["failures"]
        s = "<h2 id='s2'>2. ML Model Results Against Vendor Rules</h2>"
        fewer = 1 - a["false"]["ML Model"] / a["false"]["Vendor Rules"]
        fewer_alarm = 1 - al["false"]["ML Model"] / al["false"]["Vendor Rules"]
        s += p(
            f"The ML Model improved on Vendor Rules at ALERT and on false ALARMs. At ALERT, it caught {a['strict']['ML Model']} of 32 conditions against {a['strict']['Vendor Rules']}, gave a median {n1(a['warn_both']['ML Model'])} days of warning against {n1(a['warn_both']['Vendor Rules'])} on the {a['both']} conditions both caught, and raised {pct(fewer)} fewer false alerts (ALERTs open while the asset showed no sign of a developing condition), at {a['false']['ML Model']:.3f} against {a['false']['Vendor Rules']:.3f} per asset-month. At ALARM, {pct(al['share']['ML Model'])} of its ALARMs had a confirmed condition, against {pct(al['share']['Vendor Rules'])} of Vendor Rules', and it raised {pct(fewer_alarm)} fewer false ALARMs ({al['false']['ML Model']:.3f} against {al['false']['Vendor Rules']:.3f} per asset-month), but its ALARMs gave less warning: a median {n1(al['warn']['ML Model'])} days against {n1(al['warn']['Vendor Rules'])} on the {al['warn']['n']} conditions both escalated."
        )
        s += p(
            f"Had the shop acted on the ML Model's ALERTs instead of Vendor Rules', it would have spent {ins['saved']:.0f} fewer hours on inspections of false alerts."
        )
        s += self.exhibit("Results by level, ML Model and Vendor Rules", self.fig_levels())
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
                            "ML Model": "Score above threshold for 6 hours, or above 5 times the threshold for 2 hours (not on pumps during a low-pressure alarm)",
                            "Vendor Rules": "Baseline + 25% of zone B; fault band +6 dB; zone C; temperature warning",
                        }
                    ]
                )
            ),
        )
        s += p(
            "A condition counts as caught when an ALERT was open on the asset in the 30 days before confirmation and had opened within the 120 days before it. An ALERT that opened more than 7 days before the condition began, with no sign of degradation on the asset, is not counted."
        )
        s += p(
            f"Of the 32 conditions, the ML Model's ALERTs caught {a['strict']['ML Model']} and Vendor Rules' {a['strict']['Vendor Rules']}; both caught {a['both']}."
        )
        s += ul(
            [
                f"The ML Model caught {a['only_ml']} conditions Vendor Rules missed: two spindle cooling faults, a lubrication episode, a bearing failure and a rotary union leak. On these, Vendor Rules either had no alert (2) or had one open that began weeks before the condition with no sign of it (3).",
                f"Vendor Rules caught {a['only_vr']} conditions the ML Model missed: two with under half a day's warning (a spindle cooling fault and a compressor lubrication episode), a pump wear cycle the ML Model had learned as normal in training, and a rotary union leak that fell inside a long ML Model alert on the same spindle.",
                f"Neither caught {a['neither']}: in 2 (a lubrication episode and a rotary union leak), both methods had an alert open that began more than a week before the condition with no sign of it; in 2 lubrication episodes, Vendor Rules' alert was of that kind and the ML Model had none that counted, one of them falling inside the HMC-02 spindle points' long alert; in 1, a pump wear cycle on HMC-06's TSC pump, the ML Model's score crossed its threshold without staying above it and Vendor Rules showed nothing.",
            ]
        )
        assets = self.B["timeline_assets"]
        s += self.exhibit(
            "Alerts, inspections and failures over the evaluation year, assets with a confirmed condition",
            self.fig_timeline(
                assets, "timeline", "Alerts, inspections and failures, assets with a confirmed condition"
            ),
            "Spindles and compressors carry two sensor points; each lane shows the higher level of the two at each hour.",
        )
        s += p(
            f"Alerts on assets with no confirmed condition: ML Model, {q['ML Model']['episodes']} episodes on {q['ML Model']['assets']} assets; Vendor Rules, {q['Vendor Rules']['episodes']} episodes, {q['inspections']} inspection{'s' if q['inspections'] != 1 else ''}, nothing found."
        )
        s += p(
            f"Vendor Rules raised {ins['vr_n']} false alerts, which took {ins['vr_hours']:.0f} inspection hours at the actual CMMS labor. The ML Model raised {ins['ml_n']}, which would have taken about {ins['ml_hours']:.0f} hours at the same labor per false alert. At their peak month, inspections took {pct(ins['vr_peak_share'], 1)} (Vendor Rules) and {pct(ins['ml_peak_share'], 1)} (ML Model) of the three technicians' available hours."
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
            f"Each point is the method at a different alert threshold. The ML Model's curve lies to the left of Vendor Rules' current setting from the {left['setting'].iloc[0].split(' ')[0]} percentile up: at those thresholds it raised fewer false alerts than Vendor Rules does today, while catching {int(left['strict'].min())} to {int(left['strict'].max())} conditions."
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
                            "ML Model": "Open ALERT plus a severity criterion for the asset class: velocity in the ISO 20816 danger zone, a late-stage bearing signature, a temperature residual above its set level, the spindle thermistor at its limit or compressor discharge temperature at its limit, or sustained pump cavitation",
                            "Vendor Rules": "Baseline + 50% of zone B on velocity or of the HF reference width; zone D; temperature alarm",
                        }
                    ]
                )
            ),
        )
        nff = int((fr["vr_alarm_before"] & (fr["inspections"] > 0) & (fr["found"] == 0)).sum())
        fnd = int((fr["vr_alarm_before"] & (fr["found"] > 0)).sum())
        noi = int((fr["vr_alarm_before"] & (fr["inspections"] == 0)).sum())
        s += p(
            f"Of the ML Model's {al['n']['ML Model']} ALARMs, {pct(al['share']['ML Model'])} had a confirmed condition, against {pct(al['share']['Vendor Rules'])} of Vendor Rules' {al['n']['Vendor Rules']}, at {al['false']['ML Model']:.3f} and {al['false']['Vendor Rules']:.3f} false ALARMs per asset-month. Vendor Rules escalated {al['escalated']['Vendor Rules']} of the conditions to ALARM and the ML Model {al['escalated']['ML Model']}, {'all' if al['ml_in_vr'] == al['escalated']['ML Model'] else al['ml_in_vr']} of them among Vendor Rules'. On the sensor criteria alone, which both methods read, the ML Model's rate was {al['false_sensor_ml']:.3f}. The ML Model's ALARMs gave a median {n1(al['warn']['ML Model'])} days of warning against Vendor Rules' {n1(al['warn']['Vendor Rules'])} on the {al['warn']['n']} conditions both escalated."
        )
        s += p(
            f"Vendor Rules ALARMed before {fl['vr_alarm_before']} of the {fl['n']} unplanned failures, and the failures still happened: the inspections it triggered found no fault in {nff}, found the problem without preventing the failure in {fnd}, and no inspection was raised in the other {noi}."
        )
        s += bold("Condition Diagnosis")
        ru = {r["wo_id"]: r for r in self.B["ru"]["catches"]}
        s += p(
            f"Work requests on the ML Model's ALERTs and ALARMs named the confirmed condition on {d['strict_named']} of the {d['strict']} conditions it caught. Bearing defects were named correctly on {d['bearing_named']} of {d['bearing']}, {d['bearing_located']} of them with the damaged component located from the fault frequency that stood out. Pump wear was separated from pump bearings by fault bands rising together rather than one standing out, though {d['pump_imbalance']} pump-wear cases were named imbalance. Lubrication was separated from spindle cooling by the rise in high-frequency vibration and named correctly on {d['lub_named']} of {d['lub']}. Of the 2 rotary union leaks it caught, one was named from a {100 * ru['HMC-04-WO00024']['drop share max']:.1f}% drop in through-spindle coolant pressure; the other was named bearing, its kurtosis rise read first, though its pressure had dropped {100 * ru['HMC-03-WO00045']['drop share max']:.1f}%."
        )
        s += p(
            f"5 alerts covered two conditions on the same asset at once, such as a lubrication episode during a spindle cooling fault, and named one of them; counted by alert, the work request named a confirmed condition on {d['episodes_named']} of {d['episodes']}."
        )
        s += self.exhibit("Suspected condition against confirmed finding", self.fig_grid())
        return s

    # ------------------------------------------------------------------ section 3
    def fig_points(self) -> str:
        f, axs = plt.subplots(1, 2, figsize=(8.8, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
        for ax in axs:
            ax.set_axis_off()
        ax = axs[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.add_patch(Rectangle((0.6, 2.6), 5.6, 1.5, fc="#E8ECEF", ec=style.DARK_GREY))
        ax.text(3.4, 4.3, "Spindle (integral motor)", ha="center", fontsize=8)
        for x, lab, brg in ((1.3, "SP-F", "7014C-P4"), (5.2, "SP-R", "7012C-P4")):
            ax.add_patch(Rectangle((x, 2.6), 0.8, 1.5, fc="#B0BEC9", ec=style.DARK_GREY))
            ax.add_patch(Rectangle((x + 0.15, 4.1), 0.5, 0.3, fc=AMB, ec=style.DARK_GREY))
            ax.text(
                x + 0.4,
                2.25,
                f"{lab}\n{brg}",
                ha="center",
                va="top",
                fontsize=7.5,
                color=style.BRAND_BLUE,
                fontweight="bold",
            )
        for x, lab, brg, nm in ((6.9, "TSC", "6309", "TSC pump"), (8.5, "CP", "6205", "Flood pump")):
            ax.add_patch(Rectangle((x, 2.6), 1.2, 1.2, fc="#E8ECEF", ec=style.DARK_GREY))
            ax.add_patch(Rectangle((x + 0.35, 3.8), 0.5, 0.3, fc=AMB, ec=style.DARK_GREY))
            ax.text(
                x + 0.6,
                2.25,
                f"{lab}\n{brg}",
                ha="center",
                va="top",
                fontsize=7.5,
                color=style.BRAND_BLUE,
                fontweight="bold",
            )
            ax.text(x + 0.6, 4.3, nm, ha="center", fontsize=7.5)
        ax.text(5, 0.4, "One HMC: four sensor points", ha="center", fontsize=8.5, fontweight="bold")
        ax = axs[1]
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 5)
        for x, lab, brg, nm in ((0.6, "MDE", "6316", "Motor"), (3.2, "AE", "NU2210", "Airend")):
            ax.add_patch(Rectangle((x, 2.6), 2.0, 1.4, fc="#E8ECEF", ec=style.DARK_GREY))
            ax.add_patch(Rectangle((x + 0.75, 4.0), 0.5, 0.3, fc=AMB, ec=style.DARK_GREY))
            ax.text(
                x + 1.0,
                2.25,
                f"{lab}\n{brg}",
                ha="center",
                va="top",
                fontsize=7.5,
                color=style.BRAND_BLUE,
                fontweight="bold",
            )
            ax.text(x + 1.0, 4.45, nm, ha="center", fontsize=7.5)
        ax.text(3, 0.4, "One compressor: two sensor points", ha="center", fontsize=8.5, fontweight="bold")
        f.tight_layout()
        return self.save(f, "points", "Sensor points on an HMC and a compressor")

    def fig_signal(self) -> str:
        b = next(x for x in self.bursts if x["role"] == "healthy")
        fs = 25600.0
        f, axs = plt.subplots(5, 1, figsize=(8.8, 9.6))
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
        inp = self.t.get("inputs20")
        if inp is not None:
            colors = [ML if k_ == "sensor" else style.ACCENT for k_ in inp["kind"]]
            axs[4].bar(range(len(inp)), inp["value"], color=colors)
            axs[4].set_xticks(range(len(inp)))
            axs[4].set_xticklabels(
                [LABELS.get(c, c) for c in inp["input"]], rotation=60, ha="right", fontsize=7
            )
            axs[4].set_title(
                "The 20 inputs: 13 sensor measurements (relative to the point's level) and 7 controller measurements",
                fontsize=9.5,
            )
        f.tight_layout()
        return self.save(f, "signal", "From one second of vibration to the model's inputs")

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
        ax.bar(range(len(d)), d["n_train"], color=colors)
        for i, (v, s_) in enumerate(zip(d["n_train"], d["segments"], strict=True)):
            ax.text(i, v + 60, str(int(s_)), ha="center", fontsize=6.5, color=style.DARK_GREY)
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
            "Number above each bar: segments",
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
            ax.bar(x, g["z"], color=style.LIGHT_BLUE if lab == "healthy" else ML, label="Input")
            ax.plot(x, g["rebuild"], "-o", color=AMB, ms=3, lw=1.2, label="Rebuild")
            for i, (z, r) in enumerate(zip(g["z"], g["rebuild"], strict=True)):
                ax.plot([i, i], [z, r], color=REDC, lw=2.5, alpha=0.35)
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS.get(c, c) for c in g["input"]], rotation=60, ha="right", fontsize=7)
            ts = loc_date(g["ts"].iloc[0])
            ax.set_title(
                f"{'Healthy hour' if lab == 'healthy' else 'Faulty hour'}, HMC-01-TSC, {ts}", fontsize=9
            )
        axs[0].legend(frameon=False, fontsize=8)
        f.tight_layout()
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
        ax.bar(x, 100 * d["alert"], color=AMB, label="Inside an ALERT")
        ax.bar(x, 100 * d["alarm"], bottom=100 * d["alert"], color=REDC, label="Inside an ALARM")
        ax.set_xticks(x)
        ax.set_xticklabels([pd.Timestamp(m + "-01").strftime("%b") for m in d["month"]], fontsize=8)
        ax.set_ylabel("% of measured hours")
        ax.legend(frameon=False, fontsize=8)
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
        axs[0].set_title(f"Lubrication\nHF acceleration, g\n{cases['PM-002016']['asset']}", fontsize=7.5)
        b = next(x for x in self.bursts if x["role"] == "bearing")
        r = next(x for x in self.bursts if x["role"] == "bearing reference")
        for bb, c in ((r, VR), (b, ML)):
            ef, ea = np.array(bb["env_f"]), np.array(bb["env_a"])
            k = (ef > 1) & (ef <= 1200)
            axs[1].plot(ef[k], ea[k], color=c, lw=0.5)
        axs[1].axvline(b["fault_hz"]["bpfo"], color=AMB, ls=":", lw=0.9)
        axs[1].set_title(
            f"Bearing defect\nenvelope spectrum, Hz\nHMC-01-TSC, {loc_date(b['hour'], '%b %-d')}",
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
            f"Pump seal or impeller wear\n1x and fault bands, g²\n{cases['HMC-03-WO00042']['asset']}",
            fontsize=7.5,
        )
        g = s[s["wo_id"] == "HMC-04-WO00024"]
        for k_, c, ls in (("reference_median_bar", VR, "--"), ("pressure_median_bar", ML, "-")):
            gg = g[g["measure"] == k_]
            axs[3].plot(gg["day"], gg["value"], color=c, lw=1, ls=ls)
        axs[3].set_title(
            f"Rotary union leak\nTSC pump-on pressure, bar\n{cases['HMC-04-WO00024']['asset']}", fontsize=7.5
        )
        g = s[s["wo_id"] == "HMC-01-WO00036"]
        axs[4].plot(g["day"], g["value"], color=ML, lw=1)
        axs[4].set_title(
            f"Spindle cooling fault\ntemperature residual, C\n{cases['HMC-01-WO00036']['asset']}",
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
        s = "<h2 id='s3'>3. Model Overview</h2><h3 id='s31'>3.1 What This Model Does</h3>"
        s += bold("Sensors")
        s += p(
            "This model reads from 40 sensor points: four on each of the eight HMCs and two on each of the four compressors. Once an hour, each point records one second of vibration on three axes, sampled 25,600 times a second, and its housing temperature. A measurement is taken only in a steady state (spindles above 3,000 rpm and steady for 10 seconds; pumps and compressors running), so hours are comparable."
        )
        s += p(
            "The sensor node computes and reports 13 measurements: vibration velocity, high-frequency acceleration, bearing fault-frequency energies (six: the outer race, inner race and rolling elements, each at 1× and 2× its fault frequency), shaft-order energies (two: 1× and 2× running speed), crest factor, kurtosis and housing temperature. The ML Model also reads 7 measurements (so 20 total inputs) of the machine's operating state from its controller, such as the spindle speed band and the cutting load, so it compares each hour with healthy hours under the same conditions."
        )
        s += self.exhibit("Sensor points on an HMC and a compressor", self.fig_points())
        pts = pd.DataFrame(
            [
                (
                    "HMC spindle",
                    2,
                    "Front and rear bearing housings",
                    "7014C-P4 (front), 7012C-P4 (rear)",
                    "Triaxial",
                    16,
                ),
                ("TSC pump", 1, "Pump motor, drive end", "6309", "Triaxial", 8),
                ("Flood coolant pump", 1, "Pump motor, drive end", "6205", "Triaxial", 8),
                (
                    "Compressor",
                    2,
                    "Motor drive end; airend inlet",
                    "6316 (motor), NU2210 (airend)",
                    "Triaxial",
                    8,
                ),
            ],
            columns=["Asset type", "Points per unit", "Location", "Bearing", "Axes", "Points in the fleet"],
        )
        s += self.exhibit("Sensor points by asset type", style.table(pts))
        s += bold("What the model does.")
        s += style.table(
            pd.DataFrame(
                [
                    {
                        "What it watches": "Every hourly measurement on 40 sensor points: velocity, high-frequency acceleration, the bearing fault bands, shaft-order energies, crest factor, kurtosis and the temperature residual, with the operating regime from the controller",
                        "What it learns": "Each point's healthy pattern from up to two years of its own history, across operating regimes, re-centered after each repair and sensor remount",
                        "What it produces": "An hourly score per point, the inputs that moved most, and a level per asset (ALERT, ALARM) with a suspected condition",
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
        s += style.screenshot("screenshots/queue.png", "Maintenance queue at the snapshot", up=1)
        s += "<h3 id='s32'>3.2 Training Data</h3>"
        s += bold("From signal to inputs")
        s += p(
            "An hour's measurement starts as one second of acceleration. Its frequency spectrum shows the spindle's rotation as tall, narrow peaks."
        )
        s += p(
            "Early bearing damage shows first in the high-frequency band, 2 to 10 kHz, where each small impact rings the bearing housing. The ring is too brief to see in the spectrum, so the band is demodulated, leaving only the rhythm of its impacts. Each type of bearing damage repeats at its own rate, set by the bearing's geometry and speed: the outer race (BPFO), inner race (BPFI) and rolling element (BSF). A peak at one of those rates locates the damage."
        )
        s += self.exhibit(
            "From one second of vibration to the model's inputs, healthy HMC spindle front bearing",
            self.fig_signal(),
        )
        s += bold("Healthy hours")
        s += p(
            "The model is trained on healthy hours which form the basis of its anomaly detection. An hour counts as healthy when the point was measured and no repair was under way, no work order was open, and, after any repair or sensor remount, the point's readings had settled at a new, plausible level. The 120 days before any confirmed condition, failures included, are excluded, so that a fault developing before it was found does not enter training as normal."
        )
        s += p(
            "On a healthy hour, vibration velocity sits in ISO 20816 zone A or B (1.2 to 2.3 mm/s on spindles, rising with load), kurtosis is near 3, crest factor is 4.1 to 5.0, and the envelope spectrum has no peak at a fault frequency. Healthy levels shift with spindle speed and rise with load, and the controller measurements let the model learn each point's healthy pattern across them."
        )
        s += self.exhibit("Healthy measurements by speed band, spindle front points", self.fig_healthy_band())
        s += p(
            f"The models were trained on {tr['total']:,} healthy hours from July 2023 to June 2025 ({tr['min']:,} to {tr['max']:,} per point)."
        )
        s += self.exhibit("Healthy training hours per sensor point", self.fig_train_hours())
        s += "<h3 id='s33'>3.3 Scoring Summary</h3>"
        s += p(
            f"The ML model's ALERT threshold is the 99.5th percentile of each point's healthy training scores, one per point. Six consecutive hours above it raise ALERT, or two hours above five times it, except on a pump during a low-pressure alarm. ALARM follows on an open ALERT when a severity criterion for the asset class holds (section 2). If a point stays above its threshold for 30 days with no open work order, no confirmed condition and no significant upward trend, its threshold is re-set on those 30 days; {sc['rearms']} points were re-armed over the evaluation year."
        )
        rb = self.t["rebuild"]
        fg = (
            rb[rb["hour"] == "faulty"]
            .iloc[:13]
            .assign(g=lambda d: (d["z"] - d["rebuild"]) ** 2)
            .sort_values("g", ascending=False)
        )
        lead = ", ".join(LABELS.get(c, c).lower() for c in fg["input"].iloc[:3])
        s += self.exhibit(
            f"Inputs against the model's rebuild, healthy hour and faulty hour (bearing defect, HMC-01-TSC, {loc_date(fg['ts'].iloc[0])}: the gap concentrates in {lead})",
            self.fig_rebuild(),
        )
        s += self.exhibit("From score to level, one spindle point", self.fig_trace())
        s += p(
            f"Over the evaluation year the ML Model scored {sc['hours']:,} measured hours on 40 points. {pct(sc['outside'])} were outside any episode, {pct(sc['alert'])} inside an ALERT and {pct(sc['alarm'])} inside an ALARM, {self.B['alert']['alert_hours']['ML Model']:.0f} alert-hours per asset-month against Vendor Rules' {self.B['alert']['alert_hours']['Vendor Rules']:.0f}. It opened {sc['episodes']} ALERT episodes across {sc['assets']} assets and escalated {sc['alarms']} to ALARM. 31% of ALERT hours led with shaft-order energies (imbalance, looseness, pump wear), 25% with the temperature residual, 28% with the fault bands, crest factor or kurtosis, 10% with velocity or high-frequency acceleration, and 6% with the controller measurements."
        )
        s += self.exhibit("Score distribution, evaluation year", self.fig_score_dist())
        s += self.exhibit("Hours by level, by month", self.fig_level_month())
        cs = self.t["class_summary"]
        s += self.exhibit("Scoring summary table", style.table(cs, {"Median episode length, days": 1}))
        s += "<h3 id='s34'>3.4 Condition Signatures</h3>"
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
        f, ax = style.fig(h=3.4, grid="x")
        y = np.arange(len(g))
        ax.barh(y + 0.2, 100 * g["ml"], height=0.38, color=ML, label="ML Model")
        ax.barh(y - 0.2, 100 * g["vr"], height=0.38, color=VR, label="Vendor Rules")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t} ({n})" for t, n in zip(g.index, g["n"], strict=True)], fontsize=8)
        ax.set_xlabel("Conditions strictly caught, %")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        return self.save(f, "by_type", "Catch rate by condition type")

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
            loc="lower right",
        )
        return self.save(f, "own_signal", "Alerts on each condition type against matched healthy windows")

    def fig_diff(self) -> str:
        a, w = self.B["against"], self.t["windows"]
        rows = [
            ("Catch difference, evaluation year", a["catch_diff"], a["catch_lo"], a["catch_hi"], True),
            ("Fold 1", a["fold1"], None, None, False),
            ("Fold 2", a["fold2"], None, None, False),
        ]
        for wd in (60, 90):
            g = w[w["catch window (days)"] == wd].set_index("method")
            rows.append(
                (
                    f"Catch window {wd} days",
                    int(g.at["ML Model", "strict"] - g.at["Vendor Rules", "strict"]),
                    None,
                    None,
                    False,
                )
            )
        f, axs = plt.subplots(1, 2, figsize=(8.8, 2.9), gridspec_kw={"width_ratios": [1.4, 1]})
        ax = axs[0]
        style.chart_style(ax, "x")
        for i, (_lab, v, lo, hi, hl) in enumerate(rows[::-1]):
            if lo is not None:
                ax.plot([lo, hi], [i, i], color=ML, lw=2)
            ax.plot(v, i, "o", color=ML if hl else style.DARK_GREY, ms=8 if hl else 6)
        ax.axvline(0, color=REDC, lw=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
        ax.set_xlabel("Conditions, ML Model less Vendor Rules", fontsize=8.5)
        ax = axs[1]
        style.chart_style(ax, "x")
        fr = [
            (
                "False-alert rate difference, evaluation year",
                a["false_diff"],
                -(a["false_hi"]) * 0.2353,
                -(a["false_lo"]) * 0.2353,
                True,
            ),
            ("Fold 1", a["fold1_false"], None, None, False),
            ("Fold 2", a["fold2_false"], None, None, False),
        ]
        for i, (_lab, v, lo, hi, hl) in enumerate(fr[::-1]):
            if lo is not None:
                ax.plot([lo, hi], [i, i], color=ML, lw=2)
            ax.plot(v, i, "o", color=ML if hl else style.DARK_GREY, ms=8 if hl else 6)
        ax.axvline(0, color=REDC, lw=1)
        ax.set_yticks(range(len(fr)))
        ax.set_yticklabels([r[0] for r in fr[::-1]], fontsize=8)
        ax.set_xlabel("False alerts per asset-month, ML Model less Vendor Rules", fontsize=8.5)
        f.tight_layout()
        return self.save(f, "diff", "Difference from Vendor Rules, by period and check")

    def fig_context(self) -> str:
        d = self.t["context"]
        ctx = ["new program", "known program", "after re-centering", "benign event", "other"]
        f, ax = style.fig(h=3.0)
        x = np.arange(len(ctx))
        for j, (m, c) in enumerate((("ML Model", ML), ("Vendor Rules", VR))):
            g = d[d["method"] == m].set_index("context").reindex(ctx)
            ax.bar(
                x + (j - 1) * 0.27,
                100 * g["share"],
                width=0.27,
                color=c,
                label=f"{m}: share of its false alerts",
            )
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
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in ctx], fontsize=8)
        ax.set_ylabel("%")
        ax.legend(frameon=False, fontsize=7.5)
        return self.save(f, "context", "False alerts by context")

    def fig_healthy_above(self) -> str:
        d = self.t["healthy_above"]
        pt = self.t["healthy_above_points"]
        f, ax = style.fig(h=3.2)
        months = sorted(d["month"].unique())
        top = TOP_POINTS
        shades = ["#1F2D3D", "#3D5166", "#56708A", "#7591AC", "#93AFC8"]
        for pid, c in zip(top, shades, strict=True):
            g = pt[(pt["point_id"] == pid) & (pt["hours"] >= 24)].set_index("month").reindex(months)
            ax.plot(range(len(months)), 100 * g["share"], "-o", color=c, ms=3, lw=1.1, label=pid)
        rest = pt[~pt["point_id"].isin(top)].groupby("month")[["above", "hours"]].sum().reindex(months)
        ax.plot(
            range(len(months)),
            100 * rest["above"] / rest["hours"],
            "--",
            color=ML,
            lw=1.1,
            label="Other 11 spindle points",
        )
        for k, c, lab in (
            ("tsc_pump", style.ACCENT, "TSC pump"),
            ("flood_pump", style.LIGHT_BLUE, "Flood coolant pump"),
            ("compressor", style.GREY, "Compressor"),
        ):
            g = d[d["cls"] == k].set_index("month").reindex(months)
            ax.plot(range(len(months)), 100 * g["above"], "-s", color=c, ms=3, lw=1.1, label=lab)
        ax.axhline(0.5, color=REDC, ls="--", lw=1)
        ax.text(len(months) - 1, 0.5, " Design line 0.5%", color=REDC, fontsize=7.5, va="bottom", ha="right")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([pd.Timestamp(m + "-01").strftime("%b") for m in months], fontsize=8)
        ax.set_ylabel("% of healthy hours above threshold")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        return self.save(f, "healthy_above", "Healthy evaluation hours above the threshold, by month")

    def section4(self) -> str:
        a, ag, fo = self.B["alert"], self.B["against"], self.B["folds"]
        s = "<h2 id='s4'>4. Model Evaluation</h2>"
        s += bold("Model Card")
        card = pd.DataFrame(
            [
                ("Name", "ML Model"),
                ("Type", "Autoencoder (a neural network), 16-4-16, one per sensor point (40 models)"),
                ("Inputs", "The 13 sensor point measurements and 7 controller measurements"),
                (
                    "Training data",
                    f"Healthy hours, July 2023 to June 2025; {self.B['train']['total']:,} hours",
                ),
                (
                    "Selection",
                    "Four model types on five input sets, two expanding-window folds, selection rule fixed in advance; fixes after the first evaluation adopted by a fixed fold rule",
                ),
                (
                    "Threshold",
                    "99.5th percentile of each point's healthy scores, one per point; re-set after 30 flat days above it",
                ),
                (
                    "Levels",
                    "ALERT at 6 hours above threshold or 2 hours above 5 times it; ALARM on a severity criterion",
                ),
                ("Output", "Hourly score, leading inputs and level per asset; work request at ALERT"),
                (
                    "Evaluation",
                    "Scored over July 2025 to June 2026 against Vendor Rules; refined after the first scoring, then scored once in production",
                ),
                (
                    "Headline result",
                    f"Catches {a['strict']['ML Model']} of 32 conditions at {a['false']['ML Model']:.3f} false alerts per asset-month; Vendor Rules {a['strict']['Vendor Rules']} at {a['false']['Vendor Rules']:.3f}",
                ),
            ],
            columns=["Field", "Value"],
        )
        s += style.table(card, cls="data nowrap1")
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
            "Four model types (PCA, Mahalanobis distance, isolation forest and an autoencoder) were trained per point on five input sets, from the 13 sensor measurements to 52 inputs with the full spectrum and envelope bands, each with one threshold per point or one per spindle speed band, and scored on both folds. The selection rule, fixed before training, takes the highest catch rate above chance (the share an alert would catch if it fell at random, given how much of the time it was open) at Vendor Rules' false-alert rate, and the simplest setting within two conditions of the best."
        )
        s += p(
            f"The autoencoder on the 13 measurements with one threshold per point was selected ({skill:.3f}); the wider input sets did not improve it."
        )
        if cand is not None:
            cand.loc[cand["Selected"] == "✔", "Model"] = (
                "Autoencoder [16-6-16], 200 epochs (retuned to 16-4-16)"
            )
            s += self.exhibit(
                "Candidates on the two folds",
                style.table(
                    cand,
                    {
                        "Catch at budget": 2,
                        "Chance": 2,
                        "Skill above chance": 3,
                        "Caught at operating point": 1,
                        "False episodes per asset-month": 3,
                    },
                ),
            )
        dc = pd.read_parquet(DATA / "design.parquet") if (DATA / "design.parquet").exists() else None
        if dc is not None:
            s += self.exhibit(
                "What each design choice added",
                style.table(dc, {"Caught": 1, "False alerts per asset-month": 3, "Stuck points": 1}),
            )
        s += "<h3 id='s42'>4.2 Accuracy and Validation</h3>"
        s += p("The evaluation year scores the selected model and compares it to Vendor Rules' performance.")
        s += p(
            f"On the evaluation year, the ML Model caught {pct(a['strict']['ML Model'] / 32)} of conditions with {n1(fo['eval_warn'])} days of warning on its catches at {a['false']['ML Model']:.3f} false episodes per asset-month, against {pct(fo['strict'] / fo['conditions'])} with {n1(fo['warn'])} days of warning on its catches at {fo['false']:.3f} false episodes per asset-month on the training folds. It caught fewer than in selection, mostly on lubrication episodes, discussed in 4.3."
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
        s += self.exhibit("Catch rate by condition type", self.fig_type_bars())
        s += p(
            "Each condition type was also tested against its own signal: whether the alert came while that signal was present, against how often the same method alerted in healthy windows on the same asset."
        )
        os_ = self.t["own_signal"]
        lines = []
        for t in dict.fromkeys(os_["type"]):
            g = os_[os_["type"] == t].set_index("method")
            parts = []
            for m in ("ML Model", "Vendor Rules"):
                r = g.loc[m]
                verdict = "no better than chance" if r["alerted"] <= r["healthy"] else "above chance"
                parts.append(
                    f"{m} {pct(r['alerted'])} against {pct(r['healthy'])} of healthy windows ({verdict})"
                )
            lines.append(f"{t.capitalize()}: " + "; ".join(parts) + ".")
        s += ul(lines)
        s += self.exhibit(
            "Alerts on each condition type against matched healthy windows", self.fig_own_signal()
        )
        s += bold("Against Vendor Rules")
        s += p(
            f"The ML Model caught {ag['catch_diff']} more condition than Vendor Rules (90% interval: {ag['catch_lo']:.0f} to {ag['catch_hi']:.0f}), from resampling whole conditions, at a false-alert rate {pct(ag['false_lower'])} lower ({pct(ag['false_lo'])} to {pct(ag['false_hi'])}). It ran the same way on fold 1 ({ag['fold1']:+d}) and fold 2 ({ag['fold2']:+d}). Its false ALARM rate was {ag['alarm_ratio']:.2f} of Vendor Rules' ({ag['alarm_lo']:.2f} to {ag['alarm_hi']:.2f}). The difference in conditions caught is within what chance could produce on this many conditions."
        )
        dfs = self.t["definitions"]
        piv = dfs.pivot_table(
            index="Method",
            columns="Definition",
            values=[
                "Conditions caught",
                "False alerts per asset-month",
                "False ALARMs per asset-month",
                "Median warning, days",
            ],
            aggfunc="first",
        )
        order = ["Headline", "Lookback credit", "Original"]
        rows = []
        for m in ("ML Model", "Vendor Rules"):
            for meas, d_ in (
                ("Conditions caught", 0),
                ("False alerts per asset-month", 3),
                ("False ALARMs per asset-month", 3),
                ("Median warning, days", 1),
            ):
                rows.append(
                    {
                        "Method": m,
                        "Measure": "Median warning on its own catches, days"
                        if meas == "Median warning, days"
                        else meas,
                        **{
                            o: (f"{piv.loc[m, (meas, o)]:.{d_}f}" if pd.notna(piv.loc[m, (meas, o)]) else "")
                            for o in order
                        },
                    }
                )
        s += self.exhibit("Results under each definition", style.table(pd.DataFrame(rows)))
        w = self.t["windows"]
        wv = ", ".join(
            f"{wd} days: {int(w[(w['catch window (days)'] == wd) & (w['method'] == 'ML Model')]['strict'].iloc[0])} against {int(w[(w['catch window (days)'] == wd) & (w['method'] == 'Vendor Rules')]['strict'].iloc[0])}"
            for wd in (60, 90)
        )
        s += p(f"The comparison holds when the catch window is 60 or 90 days instead of 120 ({wv}).")
        s += self.exhibit("Difference from Vendor Rules, by period and check", self.fig_diff())
        s += "<h3 id='s43'>4.3 False Alerts & Misses</h3>"
        cx = self.t["context"]
        np_ = self.B["new_programs"]
        r = cx[(cx["method"] == "ML Model") & (cx["context"] == "new program")].iloc[0]
        s += p("The model's false alerts and misses have identifiable causes.")
        s += p(
            f"{pct(r['share'])} of the ML Model's false alerts fell on programs first run in the evaluation year ({np_['programs']} programs, {np_['cutting_hours']:,.0f} cutting hours), against {pct(r['healthy'])} of its healthy hours."
        )
        rc = self.B["recenter"]
        s += p(
            f"{rc['false_after']} false alerts followed a post-repair re-centering; {rc['flags']} post-repair inspection flags were raised, and {rc['found']} found a real problem: {rc['what']}."
        )
        s += self.exhibit("False alerts by context", self.fig_context())
        s += p(
            "Spindle points were above the threshold on 29% of their healthy hours, against the 0.5% design line, and five points (HMC-02-SP-F and -R, HMC-05-SP-R, HMC-06-SP-F and -R) carry 70% of those hours. Pumps and compressors ran at 0.7% to 14.4% a month."
        )
        s += self.exhibit("Healthy evaluation hours above the threshold, by month", self.fig_healthy_above())
        st = {x["point"]: x for x in self.B["stuck"]}
        s += p(
            f"4 points stayed in ALERT for most of the year outside any condition. Both HMC-02 spindle points were in ALERT from July 7, 2025 for {st['HMC-02-SP-F']['days']} days on the temperature residual, which ran below its training range after only {self.B['train']['min']} training hours; HMC-06-SP-F from {loc_date(st['HMC-06-SP-F']['stuck period start'])} for {st['HMC-06-SP-F']['days']} days and HMC-08-SP-R from {loc_date(st['HMC-08-SP-R']['stuck period start'])} for {st['HMC-08-SP-R']['days']} days."
        )
        mi = self.t["misses"]
        cnt = mi["Reason"].value_counts()
        s += p(f"The ML Model missed {len(mi)} of the 32 conditions:")
        s += ul(
            [
                f"{cnt.get('removed as a coincidence', 0)} had an ALERT open that began more than a week before the condition with no sign of it, so they are not counted.",
                f"{cnt.get('inside an episode opened more than 120 days earlier', 0)} fell inside the HMC-02 spindle points' long ALERT, which had opened months earlier.",
                f"{cnt.get('precursor under the threshold', 0)} had a score that crossed the threshold without staying above it: a lubrication episode, a wear cycle on HMC-06's flood pump that the model had learned as healthy in training, and a wear cycle on HMC-06's TSC pump.",
                f"{cnt.get('point re-centering', 0)} came while the point was re-centering after a repair, with no level to score against: a compressor lubrication episode.",
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
                f"[[b]]Small counts.[[/b]] 32 conditions and {self.B['failures']['n']} failures; a difference of two or three conditions could reverse in another year.",
                f"[[b]]Points stuck in ALERT.[[/b]] The HMC-02 spindle points' temperature residual ran outside its training range after {self.B['train']['min']} training hours and held both points in ALERT for most of the year, hiding two conditions. A clamp on the residual fixed HMC-02 but raised stuck points elsewhere on the folds and was set aside.",
                f"[[b]]Conditions the sensors cannot see.[[/b]] {c['other']} of the year's {c['total']} conditions are outside this comparison: {c['no_sensors']} on machines without sensors, and {c['other'] - c['no_sensors']} that left no trace in the sensor data (trips, crashes, stoppages, compressor oil cooler faults and leaks visible only in controller pressure).",
                "[[b]]Diagnosis limits.[[/b]] One alert names one condition, so a condition sharing an alert with another on the same asset is named only if it leads. On pumps, a rise at running speed alone cannot separate impeller wear from imbalance. Rotary-union leaks dropped through-spindle coolant pressure 9 to 10.5%; the drop names a leak only when no earlier rule fires, so one of the two caught was named bearing.",
                f"[[b]]Re-arms during degradation.[[/b]] {self.B['scoring']['rearms_degrading']} of the {self.B['scoring']['rearms']} re-arms in the year came while the point was degrading. None cost a catch, but each raised a threshold over a developing fault.",
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
            ("s2", "2. ML Model Results Against Vendor Rules"),
            ("s3", "3. Model Overview"),
            ("s31", "3.1 What This Model Does", "sub"),
            ("s32", "3.2 Training Data", "sub"),
            ("s33", "3.3 Scoring Summary", "sub"),
            ("s34", "3.4 Condition Signatures", "sub"),
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
TOP_POINTS = ["HMC-02-SP-F", "HMC-02-SP-R", "HMC-05-SP-R", "HMC-06-SP-F", "HMC-06-SP-R"]
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
"""


def build() -> str:
    page = Report().build()
    LOG.info("model report built: %s", style.DOCS / "reports" / "model_report.html")
    return page


if __name__ == "__main__":
    log.configure()
    build()

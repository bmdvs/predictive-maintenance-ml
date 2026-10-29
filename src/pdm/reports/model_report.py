"""The model report (docs/reports/model_report.html): the page text of section B with its exhibits.

Inputs: the Data object (gold snapshot) and the page text. Outputs: the report and its figures.
"""

from __future__ import annotations

import json
import re

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from pdm.evaluation import events as ev
from pdm.reports import appendix, style
from pdm.reports.data import END, LABEL, METHODS, Q2, QUARTER, TZ, YEAR, Data, local, utc
from pdm.reports.text import PageText

S = "B"
COL = style.METHOD_COLOR


def exhibit_title(t: PageText, section: str, start: str) -> str:
    line = t.lines(section, start)[0]
    m = re.match(r"\*\*Exhibit(?: \(NEW\))?: (.+?)\*\*", line.strip())
    return m.group(1)


def _op(d: Data, df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        np.isclose(float(lv), float(d.op.get(m, -1))) for m, lv in zip(df["method"], df["level"], strict=True)
    ]
    return df[keep]


class ModelReport:
    def __init__(self, d: Data, t: PageText) -> None:
        self.d, self.t = d, t
        self.toc: list[tuple[str, ...]] = []
        self.parts: list[str] = []
        self.n = 0

    # ---------------------------------------------------------------- page assembly
    def text(self, start: str) -> None:
        self.parts.append(self.t.html(S, start))

    def h(self, level: int, start: str, anchor: str, sub: bool = False) -> None:
        title = self.t.heading(S, start).strip("*")
        self.parts.append(f"<h{level} id='{anchor}'>{title}</h{level}>")
        self.toc.append((anchor, title, "sub") if sub else (anchor, title))

    def exhibit(self, start: str, html_: str) -> None:
        self.parts.append(style.chart(exhibit_title(self.t, S, start), html_))

    def img(self, f, name: str) -> str:
        return style.save(f, f"model_{name}", name, up=1)

    # ---------------------------------------------------------------- exhibits
    def summary_panels(self, note: str) -> str:
        """Year results for the vendor rules, the vibration model and the fusion layer: conditions caught,
        median warning on the conditions all three caught, and false episodes per asset-month with outcome known."""
        d = self.d
        methods = (
            ("rule_floor", "Vendor rules"),
            ("nbm", "Vibration model"),
            ("model_layer_instrumented", "Fusion layer"),
        )
        e = d.events("evaluation_events_shadow", YEAR)
        e = e[e["method"].isin([m for m, _ in methods])]
        e["caught"] = e["caught"].astype(bool)
        shared = set.intersection(*(set(e[(e["method"] == m) & e["caught"]]["wo_id"]) for m, _ in methods))
        pend = d.pending.set_index(["method", "period"])
        labels = [lab for _, lab in methods]
        cols = [COL[m] for m, _ in methods]
        caught = [int(e[(e["method"] == m)]["caught"].sum()) for m, _ in methods]
        total = [int((e["method"] == m).sum()) for m, _ in methods]
        warn = [
            float(e[(e["method"] == m) & e["wo_id"].isin(shared)]["warning_days"].median())
            for m, _ in methods
        ]
        false = [float(pend.loc[(m, "year"), "false_closed_rate"]) for m, _ in methods]
        f, axs = style.fig(h=3.4, w=8.6, ncols=3)
        x = np.arange(len(methods))
        ax = axs[0]
        ax.bar(x, caught, color=cols, width=0.6)
        ax.bar(x, np.subtract(total, caught), bottom=caught, color="#E6E6E6", width=0.6)
        for i, (c, n) in enumerate(zip(caught, total, strict=True)):
            ax.text(i, c / 2, str(c), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            ax.text(
                i, c + (n - c) / 2, str(n - c), ha="center", va="center", color=style.DARK_GREY, fontsize=8
            )
        ax.set_title(f"Conditions caught, of {total[0]}", fontsize=9)
        ax.legend(
            handles=[Rectangle((0, 0), 1, 1, color=c) for c in (style.DARK_GREY, "#E6E6E6")],
            labels=["Caught", "Missed"],
            frameon=False,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(0, -0.22),
            ncol=2,
        )
        for ax, vals, title, fmt in (
            (axs[1], warn, "Median warning, days", "{:.1f}"),
            (axs[2], false, "False episodes per asset-month,\noutcome known", "{:.2f}"),
        ):
            ax.bar(x, vals, color=cols, width=0.6)
            for i, v in enumerate(vals):
                ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.set_ylim(0, max(vals) * 1.18)
        axs[1].text(
            0.5, -0.24, note, transform=axs[1].transAxes, ha="center", va="top", fontsize=7, wrap=True
        )
        for ax in axs:
            ax.set_xticks(x)
            ax.set_xticklabels([lab.replace(" ", "\n") for lab in labels], fontsize=8)
        f.tight_layout()
        return self.img(f, "summary_year")

    def timeline(self) -> str:
        d = self.d
        cq = d.conds[(d.conds["credit_ts"] >= Q2) & (d.conds["credit_ts"] < END)].sort_values("credit_ts")
        ep = d.episodes
        f, ax = style.fig(h=0.9 + 0.85 * len(cq), w=8.6, grid="x")
        for i, c in enumerate(cq.itertuples()):
            y0 = i * (len(METHODS) + 1.5)
            ax.add_patch(
                Rectangle(
                    (mdates(c.credit_ts - pd.Timedelta(days=60)), y0 - 0.6),
                    60,
                    len(METHODS) + 0.2,
                    color="#F2E6C9",
                    lw=0,
                )
            )
            for k, m in enumerate(METHODS):
                g = ep[ep["method"] == m]
                g = g[
                    ev._scope_mask(g, c.asset_key, c.point_scope)
                    & (g["closed"] > c.credit_ts - pd.Timedelta(days=90))
                    & (g["opened"] < c.credit_ts + pd.Timedelta(days=10))
                ]
                for e in g.itertuples():
                    ax.plot(
                        [mdates(e.opened), mdates(min(e.closed, END))],
                        [y0 + k, y0 + k],
                        color=COL[m],
                        lw=5,
                        solid_capstyle="butt",
                    )
            ax.plot(mdates(c.credit_ts), y0 + 1.5, marker="D", color="#000000", ms=6)
            ax.text(
                mdates(END + pd.Timedelta(days=4)),
                y0 + 1.5,
                f"{c.asset_id}, {c.condition_type}",
                va="center",
                fontsize=8,
            )
        ax.set_yticks([])
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(md.DateFormatter("%b %Y"))
        ax.set_xlim(mdates(Q2 - pd.Timedelta(days=75)), mdates(END + pd.Timedelta(days=62)))
        ax.set_xticks(
            [
                mdates(pd.Timestamp(x))
                for x in ("2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01")
            ]
        )
        for m in METHODS:
            ax.plot([], [], color=COL[m], lw=5, label=LABEL[m])
        ax.plot([], [], marker="D", color="#000000", ls="none", label="Confirmed")
        ax.add_patch(Rectangle((0, 0), 0, 0, color="#F2E6C9", label="60-day catch window"))
        f.tight_layout(rect=(0, 0.08, 1, 1))
        f.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        return self.img(f, "timeline")

    def compact(self) -> str:
        d = self.d
        e_all = d.g("evaluation_events_shadow")
        e_all = e_all[e_all["period"] == QUARTER]
        e = _op(d, e_all)
        dq = d.m("data_quality_monthly")
        cq = d.conds[(d.conds["credit_ts"] >= Q2) & (d.conds["credit_ts"] < END)].sort_values("credit_ts")
        methods = ["rule_floor", "nbm", "controller", "model_layer_instrumented"]
        rows = []
        for c in cq.itertuples():
            row = {"Condition": f"{c.asset_id}, {c.condition_type}", "Confirmed": local(c.credit_ts)}
            reasons = []
            for m in methods:
                x = e[(e["wo_id"] == c.wo_id) & (e["method"] == m)]
                if not len(x) and m == "controller":
                    x = e[(e["wo_id"] == c.wo_id) & (e["method"] == "controller_uninstrumented")]
                if not len(x):
                    row[LABEL[m]] = ""
                    continue
                x = x.iloc[0]
                if bool(x.caught):
                    row[LABEL[m]] = round(float(x.warning_days), 1)
                else:
                    row[LABEL[m]] = ""
                    reasons.append(f"{LABEL[m]}: {self._miss(c, x.method, e_all, dq)}")
            row["Reason for a miss"] = "; ".join(reasons)
            rows.append(row)
        return style.table(pd.DataFrame(rows), {LABEL[m]: 1 for m in methods}, cls="data nowrap2")

    def _miss(self, c, m: str, e_all: pd.DataFrame, dq: pd.DataFrame) -> str:
        if c.condition_type == "crash":
            return "crash"
        loose = e_all[(e_all["method"] == m) & (e_all["wo_id"] == c.wo_id) & e_all["caught"].astype(bool)]
        if len(loose):
            return "precursor present but under threshold"
        if m in ("rule_floor", "nbm", "model_layer_instrumented"):
            pts = self.d.sensors.loc[self.d.sensors["asset_id"].str.startswith(c.asset_key), "point_id"]
            mo = pd.Timestamp(c.credit_ts).tz_convert(TZ).strftime("%Y-%m")
            x = dq[dq["point_id"].isin(pts) & (dq["month"] == mo)]
            if len(x) and x["no_measurement_share"].max() > 0.30:
                return "dropout"
        return "no precursor"

    def curves(self) -> str:
        d = self.d
        cs = d.g("evaluation_curves_shadow")
        f, axs = style.fig(h=3.6, w=8.6, ncols=2, sharey=True)
        for ax, (lab, cv) in zip(
            axs,
            (
                ("Validation, July 2025 to March 2026", d.g("evaluation_curves_validation")),
                ("Shadow quarter, April to June 2026", cs[cs["period"] == QUARTER]),
            ),
            strict=True,
        ):
            for m in METHODS:
                g = cv[cv["method"] == m].sort_values("false_per_asset_month")
                if g.empty:
                    continue
                ax.plot(
                    g["false_per_asset_month"],
                    g["caught_share"] * 100,
                    marker="o",
                    ms=3,
                    color=COL[m],
                    label=LABEL[m],
                )
                if m == "model_layer_instrumented":
                    ax.plot(
                        g["false_per_asset_month"],
                        g["chance_share"] * 100,
                        ls="--",
                        color=style.DARK_GREY,
                        lw=1,
                        label="Chance, model layer",
                    )
                op = _op(d, g)
                if m == "rule_floor" and len(op):
                    ax.plot(
                        op["false_per_asset_month"],
                        op["caught_share"] * 100,
                        marker="*",
                        ms=13,
                        color="#000000",
                        ls="none",
                        label="Vendor operating point",
                    )
            ax.set_title(lab, fontsize=9)
            ax.set_xlabel("False episodes per asset-month", fontsize=8)
        axs[0].set_ylabel("Conditions caught, %", fontsize=8)
        f.tight_layout(rect=(0, 0.1, 1, 1))
        f.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        return self.img(f, "operating_curves")

    def warning(self) -> str:
        d = self.d
        val = _op(d, d.g("evaluation_events_validation"))
        sh = d.g("evaluation_events_shadow")
        sh = _op(d, sh[sh["period"] == QUARTER])
        f, ax = style.fig(h=2.9, w=8.6, grid="x")
        for k, m in enumerate(METHODS):
            for dy, (src, mk, lab) in zip(
                (-0.12, 0.12), ((val, "o", "Validation"), (sh, "s", "Shadow quarter")), strict=True
            ):
                v = src[(src["method"] == m) & src["caught"].astype(bool)]["warning_days"]
                ax.scatter(
                    v, np.full(len(v), k) + dy, marker=mk, color=COL[m], s=26, label=lab if k == 0 else None
                )
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([LABEL[m] for m in METHODS])
        ax.set_xlabel("Warning days", fontsize=8)
        f.tight_layout(rect=(0, 0.08, 1, 1))
        f.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
        return self.img(f, "warning_days")

    def coverage(self) -> str:
        d = self.d
        e = d.g("evaluation_events_shadow")
        e = _op(d, e[e["period"] == YEAR])
        hm = d.conds[
            (d.conds["machine_class"] == "hmc") & d.conds["instrumented"] & d.conds["wo_id"].isin(e["wo_id"])
        ]
        rows = []
        for c in hm.itertuples():
            g = e[e["wo_id"] == c.wo_id]
            sens = bool(g[g["method"].isin(["rule_floor", "nbm"])]["caught"].astype(bool).any())
            ctl = bool(g[g["method"] == "controller"]["caught"].astype(bool).any())
            rows.append(
                {
                    "type": c.condition_type,
                    "cov": "Both"
                    if sens and ctl
                    else "Sensor layers"
                    if sens
                    else "Controller health"
                    if ctl
                    else "Neither",
                }
            )
        m = pd.DataFrame(rows).pivot_table(index="type", columns="cov", aggfunc="size", fill_value=0)
        m = (
            m.reindex(
                columns=[
                    c for c in ("Sensor layers", "Controller health", "Both", "Neither") if c in m.columns
                ]
            )
            .reset_index()
            .rename(columns={"type": "HMC cell condition type"})
        )
        un = e[e["method"] == "controller_uninstrumented"]
        u = pd.DataFrame(
            {
                "Uninstrumented condition": un["asset_id"] + ", " + un["condition_type"],
                "Caught by controller health": un["caught"].astype(bool),
                "Warning days": un["warning_days"].where(un["caught"].astype(bool)),
                "Downtime, h": un["downtime_hours"].astype(float),
            }
        )
        return style.table(m) + style.table(u, {"Warning days": 1, "Downtime, h": 1})

    def leaks(self) -> str:
        d = self.d
        lk, p = d.leaks, d.pressure
        f, axs = style.fig(h=3.0, w=8.6, ncols=3, sharey=True)
        for ax, r in zip(axs, lk.itertuples(), strict=True):
            g = p[
                (p["machine_id"] == r.machine_id)
                & (p["local_date"] >= r.opened - pd.Timedelta(days=60))
                & (p["local_date"] <= r.opened)
            ]
            ax.axvspan(r.span_start, min(r.span_end, r.opened), color="#F2E6C9", lw=0)
            ax.plot(
                g["local_date"],
                g["advisory_bar"],
                color=style.GREY,
                lw=1,
                label="Advisory series (pump-off time included)",
            )
            ax.plot(g["local_date"], g["pump_on_bar"], color=style.BRAND_BLUE, lw=1.4, label="Pump running")
            ax.axvline(r.opened, color=style.RED, lw=1.2, label="Work order opened")
            ax.axhline(69.0, color=style.DARK_GREY, lw=0.7, ls="--")
            ax.set_title(f"{r.machine_id}, {r.opened:%B %Y}", fontsize=9)
            ax.xaxis.set_major_locator(md.DayLocator(bymonthday=[1, 15]))
            ax.xaxis.set_major_formatter(md.DateFormatter("%d %b"))
            ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        axs[0].set_ylabel("Bar, daily median", fontsize=8)
        axs[0].add_patch(Rectangle((0, 0), 0, 0, color="#F2E6C9", label="Low span"))
        f.tight_layout(rect=(0, 0.12, 1, 1))
        f.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
        return self.img(f, "leak_pressure")

    def outcomes(self) -> str:
        d = self.d
        cats = ["Confirmed finding", "Still open at the record end", "Closed, no finding recorded"]
        dv = d.g("diagnosis_agreement_validation")
        rows = [
            {
                "Period": "Validation",
                "Confirmed finding": int((dv["outcome"] == "confirmed").sum()),
                "Still open at the record end": 0,
                "Closed, no finding recorded": int((dv["outcome"] != "confirmed").sum()),
            }
        ]
        for mo, g in d.outcomes.groupby("month"):
            vc = g["outcome_end"].value_counts()
            rows.append(
                {"Period": pd.Timestamp(mo + "-01").strftime("%B %Y")} | {c: int(vc.get(c, 0)) for c in cats}
            )
        df = pd.DataFrame(rows)
        cats = [c for c in cats if df[c].sum() > 0]
        f, ax = style.fig(h=2.8, w=8.6)
        bottom = np.zeros(len(df))
        for c, colr in zip(cats, (style.GREEN, style.BRAND_BLUE, style.LIGHT_BLUE), strict=False):
            ax.bar(df["Period"], df[c], bottom=bottom, color=colr, label=c, width=0.55)
            bottom += df[c].to_numpy()
        ax.set_ylabel("Work requests", fontsize=8)
        f.tight_layout(rect=(0, 0.1, 1, 1))
        f.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        return self.img(f, "request_outcomes")

    def matched(self) -> str:
        d = self.d
        ms = d.g("evaluation_matched_shadow")
        pend = d.pending.set_index(["method", "period"])
        key = {
            "vendor": "rule_floor",
            "nbm": "nbm",
            "model_layer_instrumented": "model_layer_instrumented",
            "controller": "controller",
        }
        rows = []
        for lab, per, pk in (
            ("July 2025 to June 2026", YEAR, "year"),
            ("April to June 2026", QUARTER, "quarter"),
        ):
            for x in ms[ms["period"] == per].itertuples():
                if x.method not in key:
                    continue
                n = 20 if x.method == "controller" else 14
                p = pend.loc[(key[x.method], pk)]
                rows.append(
                    {
                        "Period": lab,
                        "Method": LABEL[x.method],
                        "Caught at the budget, %": 100 * x.interp_caught_share,
                        "Chance, %": 100 * x.interp_chance_share,
                        "Budget, false episodes per asset-month": x.budget_false_per_asset_month,
                        "Inspection hours a month": x.budget_false_per_asset_month * n * d.inspection_hours,
                        "False episodes at the operating point": int(p["false"]),
                        "Outcome pending": int(p["pending"]),
                        "False per asset-month, outcome known": p["false_closed_rate"],
                    }
                )
        return style.table(
            pd.DataFrame(rows),
            {
                "Caught at the budget, %": 0,
                "Chance, %": 0,
                "Budget, false episodes per asset-month": 3,
                "Inspection hours a month": 1,
                "False per asset-month, outcome known": 2,
            },
            cls="data nowrap1",
        )

    def agreement(self) -> str:
        d = self.d
        x = d.g("diagnosis_agreement_shadow")
        x = x[(x["period"] == YEAR) & (x["outcome"] == "confirmed")]
        m = (
            pd.crosstab(x["suspected"], x["confirmed"])
            .reset_index()
            .rename(columns={"suspected": "Suspected condition"})
        )
        m.columns.name = None
        return style.table(m)

    def survival(self) -> tuple[str, str]:
        from lifelines import KaplanMeierFitter

        d = self.d
        lives, sw = d.g("component_lives"), d.g("survival_shared_shape_weibull").set_index("component_class")
        names = {
            "flood_pump": "Flood coolant pump",
            "tsc_pump": "TSC pump",
            "spindle": "Spindle",
            "compressor": "Compressor",
        }
        f, axs = style.fig(h=2.8, w=8.6, ncols=4, sharey=True)
        for ax, cls in zip(axs, names, strict=True):
            lv = lives[lives["component_class"] == cls]
            s = sw.loc[cls]
            k = KaplanMeierFitter().fit(lv["duration_days"], lv["event"], entry=lv["entry_days"])
            sf = k.survival_function_
            ax.step(sf.index, sf.iloc[:, 0], where="post", color=style.DARK_GREY, lw=1, label="Empirical")
            tt = np.linspace(0, max(lv["duration_days"].max(), 10), 200)
            ax.plot(
                tt,
                np.exp(-((tt / s.scale_lambda_days) ** s.shape_rho)),
                color=style.BRAND_BLUE,
                lw=1.3,
                label="Fleet Weibull",
            )
            ax.set_title(names[cls], fontsize=9)
            ax.set_xlabel("Days", fontsize=8)
        axs[0].set_ylabel("Share surviving", fontsize=8)
        f.tight_layout(rect=(0, 0.1, 1, 1))
        f.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8)
        img = self.img(f, "survival")
        rows = []
        for cls, lab in names.items():
            s = sw.loc[cls]
            rows.append(
                {
                    "Class": lab,
                    "Lives": int(s.lives),
                    "Failures": int(s.events),
                    "Censored": int(s.lives - s.events),
                    "Characteristic life, days": s.scale_lambda_days,
                    "90% interval": f"{s.scale_low:,.0f} to {s.scale_high:,.0f}",
                    "Median life, days": s.median_life_days,
                    "Median, 90% interval": f"{s.median_low:,.0f} to {s.median_high:,.0f}",
                }
            )
        return img, style.table(pd.DataFrame(rows), {"Characteristic life, days": 0, "Median life, days": 0})

    def remaining(self) -> str:
        d = self.d
        pg = d.g("prognosis")
        r0 = pg[pg["wo_id"] == "HMC-04-WO00029"].sort_values("lambda")
        c = d.conds[d.conds["wo_id"] == "HMC-04-WO00029"].iloc[0]
        rf = d.inputs["rf"]
        rf = rf[
            (rf["method"] == "rule_floor")
            & (rf["sweep"] == d.cfg["rules"]["vendor"]["alert_frac_zone_b_width"])
        ]
        window = pd.Timedelta(days=d.cfg["evaluation"]["catch"]["opened_within_days"])
        hit = rf[
            ev._scope_mask(rf, c.asset_key, c.point_scope)
            & (rf["opened"] <= c.credit_ts)
            & (rf["closed"] > c.credit_ts)
            & (rf["opened"] >= c.credit_ts - window)
        ].sort_values("opened")
        det = utc(hit["opened"]).iat[0]
        pid = hit["point_id"].iat[0] or c.point_scope
        s = d.st.read_pd("silver", "hourly_indicators_prognosis_instances")
        s["hour_utc"] = utc(s["hour_utc"])
        s = s[
            (s["point_id"] == pid)
            & ~s["implausible"].fillna(False).astype(bool)
            & (s["hour_utc"] <= c.credit_ts)
        ]
        s = s.set_index("hour_utc")["vel_max_dir_mm_s"].resample("1D").median().dropna()
        dd = (s.index - det.floor("D")) / pd.Timedelta(days=1)
        life = (c.credit_ts - det.floor("D")) / pd.Timedelta(days=1)
        thr, phi = float(r0["threshold_mm_s"].iat[0]), float(r0["phi_mm_s"].iat[0])
        f, ax = style.fig(h=3.3, w=8.6)
        ax.plot(
            dd, s.to_numpy(), marker="o", ms=3, color=style.DARK_GREY, lw=1, label="Daily median velocity"
        )
        ax.axhline(thr, color=style.RED, lw=1, label=f"Threshold, {thr:.1f} mm/s")
        ax.axvline(life, color="#000000", lw=1, ls="--", label="Failure")
        for k, (_, r) in enumerate(r0.iterrows()):
            y = thr + 0.3 + 0.35 * k
            hi = min(r["t_days"] + r["pi_high"], life + 60)
            ax.plot(
                [r["t_days"] + r["pi_low"], hi],
                [y, y],
                color=style.BRAND_BLUE,
                lw=4,
                alpha=0.7,
                label="Predicted range" if k == 0 else None,
            )
            ax.plot(r["t_days"], y, marker=">", color=style.BRAND_BLUE, ms=5)
            ax.plot(
                r["t_days"] + r["pred_rul_days"],
                y,
                marker="|",
                color="#000000",
                ms=10,
                label="Point estimate" if k == 0 else None,
            )
            ax.text(hi + 1, y, f"at {round(100 * r['lambda'])}%", va="center", fontsize=7)
        ax.set_xlim(-30, life + 70)
        ax.set_xlabel("Days since detection", fontsize=8)
        ax.set_ylabel("mm/s", fontsize=8)
        f.tight_layout(rect=(0, 0.1, 1, 1))
        tt = np.linspace(0, life, 100)
        ax.plot(tt, np.full_like(tt, phi), color=style.GREY, lw=1, ls=":", label="Healthy level")
        f.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        return self.img(f, "remaining_time")

    def flow(self, name: str, columns: list[tuple[str, list[str]]], w: float = 8.6, h: float = 3.0) -> str:
        import textwrap

        fs = 8 if len(columns) <= 3 else 7
        width = 26 if len(columns) <= 3 else 15
        columns = [(c, [textwrap.fill(x, width) for x in items]) for c, items in columns]
        f, ax = plt.subplots(figsize=(w, h))
        ax.set_axis_off()
        ax.set_xlim(0, len(columns))
        ax.set_ylim(0, 1)
        for i, (title, items) in enumerate(columns):
            ax.text(
                i + 0.5,
                0.95,
                textwrap.fill(title, width),
                ha="center",
                va="top",
                fontsize=fs + 1,
                fontweight="bold",
                color=style.BRAND_BLUE,
            )
            n = len(items)
            for j, it in enumerate(items):
                y = 0.82 - (j + 1) * (0.8 / (n + 1))
                hh = 0.065 * len(it.splitlines()) + 0.05
                ax.add_patch(
                    FancyBboxPatch(
                        (i + 0.08, y - hh / 2),
                        0.84,
                        hh,
                        boxstyle="round,pad=0.01",
                        fc="#EEF3F7",
                        ec=style.ACCENT,
                    )
                )
                ax.text(i + 0.5, y, it, ha="center", va="center", fontsize=fs)
            if i < len(columns) - 1:
                ax.annotate(
                    "",
                    xy=(i + 1.06, 0.45),
                    xytext=(i + 0.94, 0.45),
                    arrowprops={"arrowstyle": "->", "color": style.DARK_GREY},
                )
        return style.save(f, f"model_{name}", name, up=1)

    def layers(self) -> str:
        """Equipment, data sources and monitoring layers, left to right, with the existing and added layers."""
        existing, added, fitted, plain = "#E6E6E6", style.ACCENT, "#F2E6C9", "#FFFFFF"
        f, ax = plt.subplots(figsize=(8.6, 4.4))
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(0.95, 6.1)

        def box(x0: float, yc: float, w: float, h: float, title: str, sub: str, fc: str) -> None:
            dark = fc == added
            ax.add_patch(
                FancyBboxPatch(
                    (x0, yc - h / 2), w, h, boxstyle="round,pad=0.02", fc=fc, ec=style.DARK_GREY, lw=0.8
                )
            )
            ax.text(
                x0 + w / 2,
                yc + 0.11,
                title,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white" if dark else style.TEXT,
            )
            ax.text(
                x0 + w / 2,
                yc - 0.15,
                sub,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if dark else style.DARK_GREY,
            )

        def arrow(a: tuple[float, float], b: tuple[float, float]) -> None:
            ax.annotate(
                "",
                xy=b,
                xytext=a,
                arrowprops={"arrowstyle": "-|>", "color": style.DARK_GREY, "lw": 0.8, "mutation_scale": 9},
            )

        equipment = [
            ("HMCs", "4 machines", True),
            ("Compressors", "2 units", True),
            ("VMCs", "8 machines", False),
            ("Lathes", "5 machines", False),
            ("Mill-turns", "2 machines", False),
            ("Grinder", "1 machine", False),
        ]
        ex0, ew, eh = 0.05, 1.55, 0.62
        ys = [5.6 - 0.85 * i for i in range(len(equipment))]
        sx0, sw = 2.55, 2.45
        sensor_y, sensor_h = 5.0, 1.0
        ctrl_y, ctrl_h = 2.6, 1.3
        for (title, sub, has_sensor), y in zip(equipment, ys, strict=True):
            box(ex0, y, ew, eh, title, sub, fitted if has_sensor else plain)
        box(sx0, sensor_y, sw, sensor_h, "Sensor data", "20 points on 14 assets", plain)
        box(sx0, ctrl_y, sw, ctrl_h, "Controller data", "Machine and compressor controllers", plain)
        for k, y in enumerate(ys[:2]):
            arrow((ex0 + ew + 0.04, y), (sx0 - 0.04, sensor_y + 0.2 - 0.4 * k))
        for k, y in enumerate(ys):
            arrow((ex0 + ew + 0.04, y), (sx0 - 0.04, ctrl_y + 0.5 - 0.2 * k))
        lx0, lw_ = 5.65, 2.15
        vib_y, ven_y, ch_y = 5.55, 4.45, ctrl_y
        box(lx0, vib_y, lw_, 0.7, "Vibration model", "Autoencoder per point", added)
        box(lx0, ven_y, lw_, 0.7, "Vendor rules", "Threshold alerts", existing)
        box(lx0, ch_y, lw_, 0.7, "Controller health", "Load drift, temperature residuals", existing)
        arrow((sx0 + sw + 0.04, sensor_y + 0.2), (lx0 - 0.04, vib_y))
        arrow((sx0 + sw + 0.04, sensor_y - 0.2), (lx0 - 0.04, ven_y))
        arrow((sx0 + sw + 0.04, ctrl_y), (lx0 - 0.04, ch_y))
        fx0, fw = 8.25, 1.72
        fus_y, q_y = 4.0, 1.75
        box(fx0, fus_y, fw, 0.8, "Fusion layer", "One status per asset", added)
        box(fx0, q_y, fw, 0.8, "Maintenance queue", "Cases and work requests", plain)
        for y, end in ((vib_y, fus_y + 0.25), (ven_y, fus_y), (ch_y, fus_y - 0.25)):
            arrow((lx0 + lw_ + 0.04, y), (fx0 - 0.04, end))
        arrow((fx0 + fw / 2, fus_y - 0.44), (fx0 + fw / 2, q_y + 0.44))
        handles = [
            Rectangle((0, 0), 1, 1, fc=c, ec=style.DARK_GREY, lw=0.6) for c in (existing, added, fitted)
        ]
        f.legend(
            handles,
            ["Existing", "Added", "Wireless sensors fitted"],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=8,
        )
        f.tight_layout(rect=(0, 0.05, 1, 1))
        return style.save(f, "model_monitoring_layers", "Equipment, data sources and monitoring layers", up=1)

    def spindle(self) -> str:
        f, ax = plt.subplots(figsize=(8.6, 2.8))
        ax.set_axis_off()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.add_patch(Rectangle((1.2, 1.0), 7.2, 2.0, fc="#E8ECEF", ec=style.DARK_GREY))
        ax.add_patch(Rectangle((0.2, 1.75), 9.4, 0.5, fc="#C8D2DA", ec=style.DARK_GREY))
        for x, lab in ((1.8, "Front bearing set"), (7.2, "Rear bearing set")):
            ax.add_patch(Rectangle((x, 1.0), 1.0, 0.75, fc="#B0BEC9", ec=style.DARK_GREY))
            ax.add_patch(Rectangle((x, 2.25), 1.0, 0.75, fc="#B0BEC9", ec=style.DARK_GREY))
            ax.text(x + 0.5, 0.65, lab, ha="center", fontsize=8)
        ax.add_patch(Rectangle((3.3, 1.0), 3.4, 0.75, fc="#D9C9A8", ec=style.DARK_GREY))
        ax.add_patch(Rectangle((3.3, 2.25), 3.4, 0.75, fc="#D9C9A8", ec=style.DARK_GREY))
        ax.text(5.0, 0.65, "Integral motor", ha="center", fontsize=8)
        ax.text(0.2, 2.6, "Tool end", fontsize=8)
        for x, lab in ((2.3, "SP-F"), (7.7, "SP-R")):
            ax.add_patch(Rectangle((x - 0.25, 3.0), 0.5, 0.35, fc=style.AMBER, ec=style.DARK_GREY))
            ax.text(x, 3.55, lab, ha="center", fontsize=9, fontweight="bold", color=style.BRAND_BLUE)
        return style.save(f, "model_spindle", "HMC spindle sensor positions", up=1)

    def record(self) -> str:
        f, ax = plt.subplots(figsize=(8.6, 1.3))
        ax.set_axis_off()
        spans = [
            ("Training", "2023-07-01", "2025-07-01", "#D9D9D9"),
            ("Validation", "2025-07-01", "2026-04-01", style.LIGHT_BLUE),
            ("Shadow quarter", "2026-04-01", "2026-07-01", style.GREEN),
        ]
        t0 = pd.Timestamp("2023-07-01")
        for lab, a, b, c in spans:
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            ax.barh(0, (b - a).days, left=(a - t0).days, color=c, height=0.6)
            mid = (a - t0).days + (b - a).days / 2
            if (b - a).days > 200:
                ax.text(mid, 0, lab, ha="center", va="center", fontsize=9, color="#222222")
            else:
                ax.text(mid, 0.42, lab, ha="center", va="bottom", fontsize=9, color="#222222")
            ax.text((a - t0).days, -0.55, a.strftime("%b %Y"), ha="center", fontsize=8)
        ax.text((pd.Timestamp("2026-07-01") - t0).days + 40, -0.55, "Jun 2026", ha="left", fontsize=8)
        ax.set_xlim(-30, (pd.Timestamp("2026-07-01") - t0).days + 120)
        ax.set_ylim(-0.8, 0.8)
        return style.save(f, "model_record", "Record periods", up=1)

    def conditions_chart(self) -> str:
        c = self.d.conds
        t = c.pivot_table(
            index="condition_type", columns="period", values="wo_id", aggfunc="count", fill_value=0
        )[["training", "validation", "shadow"]]
        t = t.loc[t.sum(axis=1).sort_values().index]
        f, ax = style.fig(h=3.4, w=8.6, grid="x")
        left = np.zeros(len(t))
        for col, colr, lab in (
            ("training", "#BBBBBB", "Training"),
            ("validation", style.LIGHT_BLUE, "Validation"),
            ("shadow", style.GREEN, "Shadow quarter"),
        ):
            ax.barh(t.index, t[col], left=left, color=colr, label=lab)
            left += t[col].to_numpy()
        ax.set_xlabel("Confirmed conditions", fontsize=8)
        f.tight_layout(rect=(0, 0.1, 1, 1))
        f.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        return self.img(f, "conditions_by_period")

    def healthy(self) -> str:
        nm = self.d.g("nbm_models")
        nm["segment_start"] = utc(nm["segment_start"])
        rows = []
        for p, g in nm.groupby("point_id"):
            g = g.sort_values("segment_start")
            rows.append(
                {
                    "Point": p,
                    "Segments": len(g),
                    "Repairs and remounts": int((g["segment"] > 0).sum()),
                    "Healthy hours, all segments": int(g["n_train_final"].fillna(0).sum()),
                    "Latest segment start": local(g["segment_start"].iat[-1]),
                }
            )
        return style.table(pd.DataFrame(rows))

    def signatures(self) -> str:
        d = self.d
        e = pd.concat(
            [
                _op(d, d.g("evaluation_events_validation")),
                _op(d, d.g("evaluation_events_shadow").query("period == @QUARTER")),
            ]
        )
        where = {
            "bearing (unplanned)": (
                "Bearing points; controller on uninstrumented machines",
                "Envelope fault bands, HF, velocity",
            ),
            "lubrication": ("Spindle points", "Broadband HF without fault frequencies"),
            "imbalance": ("Pump and compressor points", "1x radial"),
            "spindle cooling": (
                "Spindle points (housing temperature) and the controller thermistor",
                "Sensor temperature; thermistor residual",
            ),
            "rotary union": ("Controller and spindle points", "TSC pressure drop with HF"),
            "pump seal or impeller": ("Pump points", "HF and 1x; TSC pressure"),
            "other machine stoppage": ("Controller", "Axis load drift"),
            "crash": ("Controller event log", "None before the event"),
            "compressor trip": ("Compressor controller", "Discharge temperature residual"),
            "cooler (compressor)": ("Compressor controller", "Discharge temperature residual"),
            "looseness": ("Pump and compressor points", "1x to 10x harmonics"),
        }
        rows = []
        for ty, g in e.groupby("condition_type"):
            caught = sorted(
                {
                    LABEL[m]
                    for m in g.loc[g["caught"].astype(bool), "method"]
                    if m in ("rule_floor", "nbm", "controller", "controller_uninstrumented")
                }
                - {LABEL["controller_uninstrumented"]}
                | (
                    {LABEL["controller"]}
                    if (g["caught"].astype(bool) & (g["method"] == "controller_uninstrumented")).any()
                    else set()
                )
            )
            w, ind = where.get(ty, ("", ""))
            rows.append(
                {
                    "Condition type": ty,
                    "Conditions": g["wo_id"].nunique(),
                    "Where it shows": w,
                    "Indicator": ind,
                    "Caught by": ", ".join(caught) or "None",
                }
            )
        return style.table(pd.DataFrame(rows))

    def versions(self) -> str:
        reg = json.loads(self.d.st.file("mlflow/registry.json").read_text(encoding="utf-8"))
        name = next(iter(reg))
        champion = str(reg[name]["aliases"].get("champion"))
        rows = []
        for v in reg[name]["versions"]:
            tg = v["tags"]
            rows.append(
                {
                    "Version": v["version"],
                    "Status": "registered before promotion gates"
                    if str(v["version"]) == "1"
                    else tg.get("status", "").replace("_", " "),
                    "Month gated": tg.get("false_episodes_month", ""),
                    "Selection": tg.get("gate_selection", ""),
                    "Parity": tg.get("gate_parity", ""),
                    "Segments": tg.get("gate_segments", ""),
                    "False episodes": tg.get("gate_false_episodes", ""),
                    "Rate": float(tg["false_episodes_rate"]) if "false_episodes_rate" in tg else np.nan,
                    "Budget": float(tg["false_episodes_budget"]) if "false_episodes_budget" in tg else np.nan,
                    "Champion": "yes" if str(v["version"]) == champion else "",
                }
            )
        return style.table(pd.DataFrame(rows), {"Rate": 3, "Budget": 3})

    def indicators(self) -> str:
        c = self.d.cfg
        vb, hb = c["indicators"]["velocity_band_hz"], c["indicators"]["hf_band_hz"]
        rows = [
            ("Velocity RMS", f"{vb[0]} to {vb[1]:,} Hz", "Overall vibration severity (ISO 20816)"),
            ("Acceleration RMS", "10 to 10,000 Hz", "Overall acceleration"),
            (
                "High-frequency acceleration RMS",
                f"{hb[0]:,} to {hb[1]:,} Hz",
                "Early bearing distress, lubrication, contamination",
            ),
            ("Crest factor and kurtosis", "Burst", "Impulsiveness"),
            (
                "Order bands 1x to 10x",
                "Plus or minus 3% around each shaft order",
                "Imbalance (1x), misalignment (2x), looseness (harmonics)",
            ),
            (
                "Envelope fault bands",
                "BPFO, BPFI, BSF, FTF and harmonics, demodulated in the point's band",
                "Bearing defects by location",
            ),
            (
                "One-third-octave bands",
                "32 bands, 10 Hz to 10 kHz",
                "Spectral shape for the per-point models",
            ),
            ("Temperature", "Sensor", "Heat at the housing"),
        ]
        return style.table(pd.DataFrame(rows, columns=["Indicator", "Band", "What it measures"]))

    def definitions(self) -> str:
        return style.table(appendix.definitions(appendix.load()))

    def limitations(self) -> str:
        rows = appendix.limitations(appendix.load()) + self.t.items(S, "Exhibit: Limitations")
        return style.table(pd.DataFrame({"Limitation": rows}))

    # ---------------------------------------------------------------- page
    def build(self) -> str:
        d, t, x = self.d, self.t, self.exhibit
        self.h(2, "1. Summary", "s1")
        for s in (
            "The shop monitors machine condition",
            "Through June 2026, the shop maintained",
            "This project adds two layers",
        ):
            self.text(s)
        x("Exhibit: Equipment, data sources and monitoring layers", self.layers())
        for s in (
            "The fusion layer is the shop's new operating output",
            "Over the evaluation year, July 2025 to June 2026",
            "Of the 12 conditions on instrumented assets",
            "The fusion layer caught 6 of the 12",
            "The differences rest on a few conditions",
            "Through-spindle coolant pressure, measured",
        ):
            self.text(s)
        note = re.search(r'"(.+?)"', t.lines(S, "Exhibit: Vendor rules")[1]).group(1)
        x("Exhibit: Vendor rules, vibration model and fusion layer", self.summary_panels(note))
        self.h(2, "2. Model results", "s2")
        self.h(3, "2.1 Detection and warning", "s21", True)
        self.text("The model layer caught more conditions")
        self.text("In the quarter, the model layer opened")
        t.lines(S, "The event timeline as specified")
        t.lines(S, "The compact table as specified")
        x("Exhibit: Conditions confirmed in the shadow quarter", self.timeline())
        x("Exhibit: Warning days and miss reasons", self.compact())
        self.text("At the vendor rules' false-alert rate for the year")
        self.text("Chance at that rate")
        x("Exhibit: Conditions caught against false episodes", self.curves())
        self.text("Over the year, warning on conditions caught")
        x("Exhibit: Warning days per caught condition", self.warning())
        self.text("On the HMC cell, every catch")
        self.text(
            "On machine components without sensors, controller data caught 1 of 6 conditions: the VMC-06"
        )
        x("Exhibit: Conditions by the layers", self.coverage())
        self.text("Both rotary-union leaks were missed")
        x("Exhibit (NEW): Through-spindle coolant pressure", self.leaks())
        self.text("The year's results rest on 12 conditions")
        self.h(3, "2.2 Work requests and diagnosis", "s22", True)
        self.text("Most work requests closed without")
        self.text("In validation, 25 of 29 requests")
        x("Exhibit: Work requests by outcome", self.outcomes())
        self.text("At the vendor rules' false-alert rate, false alerts cost")
        self.text("Of the false episodes in the quarter")
        x("Exhibit: Detection at the vendor rules'", self.matched())
        self.text("Requests correctly identified")
        self.text("They misread")
        self.text("Some CMMS finding codes")
        x("Exhibit: Suspected condition against confirmed finding", self.agreement())
        self.h(3, "2.3 Component life and remaining time", "s23", True)
        self.text("Component failure rates rise")
        self.text("Across compressors, flood coolant pumps")
        self.text("At the TSC pumps' two-year replacement interval")
        img, tab = self.survival()
        x("Exhibit: Survival by component class", img)
        x("Exhibit: Component lives, failures and median life", tab)
        self.text("Remaining-time estimates were tested")
        self.text("Remaining-time estimates are not offered")
        x("Exhibit: Remaining-time estimates against", self.remaining())
        self.h(2, "3. Model overview", "s3")
        self.h(3, "3.1 What the system does", "s31", True)
        self.text("The system reads three sources")
        self.text("It scores each hour against three layers")
        self.text("Scoring ran monthly on Azure Databricks")
        x(
            "Exhibit: Inputs, scoring and outputs",
            self.flow(
                "flow",
                [
                    (
                        "Reads",
                        ["Wireless sensor points (20)", "Controller data (20 machines)", "CMMS work orders"],
                    ),
                    (
                        "Scores",
                        ["Vendor rules", "Per-point models", "Controller health", "Fusion and advisory"],
                    ),
                    ("Produces", ["Asset status", "Cases", "Work requests"]),
                ],
            ),
        )
        self.text("The queue lists every monitored asset")
        x(
            "Exhibit: Condition monitoring queue",
            style.screenshot("screenshots/queue.png", "", up=1).split("</div>", 1)[1],
        )
        self.h(3, "3.2 Monitored assets and signals", "s32", True)
        self.text("HMCs")
        sens = d.sensors.merge(
            d.st.read_pd("bronze", "reg_bearings").rename(columns={"part": "bearing"}),
            on="bearing",
            how="left",
        )
        loc = sens["location"].str.replace("_", " ").str.replace(r"\bde\b", "drive end", regex=True)
        spindle = sens["point_id"].str.contains("-SP-")
        axes = np.where(
            spindle, "Resultant and radial (" + sens["radial_axis"].astype(str) + ")", "Three (x, y, z)"
        )
        pts = pd.DataFrame(
            {
                "Point": sens["point_id"],
                "Asset": sens["asset_id"],
                "Location": loc,
                "Axes reported": axes,
                "Bearing": sens["bearing"],
                "Bearing type": sens["kind"].str.replace("_", " "),
            }
        )
        x("Exhibit: Sensor points by asset", style.table(pts))
        x("Exhibit: HMC spindle sensor positions", self.spindle())
        ctrl = pd.DataFrame(
            [
                (
                    "Controller, 1 Hz stream (shadow quarter) and per-minute aggregates",
                    "Spindle speed and load, axis loads, spindle thermistor, feed override; hydraulic and through-spindle coolant pressure on the HMCs",
                    20,
                ),
                ("Controller event log", "Alarms, collisions, tool breaks, stops", 20),
                (
                    "Compressor controllers",
                    "Load state, discharge pressure and temperature, run hours, room temperature",
                    2,
                ),
            ],
            columns=["Source", "Signals", "Machines"],
        )
        x("Exhibit: Controller data by source", style.table(ctrl))
        self.h(3, "3.3 Training data", "s33", True)
        self.text("The record runs from July 2023")
        self.text("The record holds 64 confirmed conditions")
        x("Exhibit: Record periods", self.record())
        x("Exhibit: Confirmed conditions by type and period", self.conditions_chart())
        x("Exhibit: Healthy hours per sensor point", self.healthy())
        self.h(3, "3.4 Conditions and their signatures", "s34", True)
        self.text("Condition types, where each appears")
        x("Exhibit: Conditions, signals and catching layers", self.signatures())
        self.t.lines(S, "Appendix")
        self.parts.append("<h2 id='appendix'>Appendix</h2>")
        self.toc.append(("appendix", "Appendix"))
        self.ap("A. Scoring layers", "aA")
        self.text("Sensitivity: the bearing-wear-out-only")
        x(
            "Exhibit: Scoring flow",
            self.flow(
                "scoring_flow",
                [
                    ("Rule floor", ["ISO 20816 zones", "Vendor alerts"]),
                    ("Per-point models", ["Autoencoder per point", "99.5th pct, 6 h"]),
                    ("Controller health", ["Load drift", "Thermal residuals"]),
                    ("Fusion", ["Status per asset", "Cases"]),
                    ("Advisory", ["Work requests"]),
                ],
                h=3.2,
            ),
        )
        self.ap("B. Condition indicators", "aB")
        self.text("Indicators computed from each hourly burst")
        x("Exhibit: Indicators", self.indicators())
        self.ap("C. Evaluation definitions", "aC")
        x("Exhibit: Definitions", self.definitions())
        self.ap("D. Model selection", "aD")
        self.text("Four normal-behavior candidates")
        self.text("The autoencoder was selected")
        cand = d.g("nbm_candidates")
        names = {
            "autoencoder_third_octave": "Autoencoder, one-third-octave bands",
            "mahalanobis": "Mahalanobis distance",
            "pca_t2_spe": "PCA, T-squared and SPE",
            "isolation_forest": "Isolation forest",
        }
        cand = pd.DataFrame(
            {
                "Candidate": cand["candidate"].map(names).fillna(cand["candidate"]),
                "Caught in training, %": 100 * cand["training_caught"],
                "Chance, %": 100 * cand["training_chance"],
                "False episodes per asset-month": cand["operating_false_per_asset_month"],
                "Selected": cand["champion"].map({True: "yes", False: ""}),
            }
        )
        x(
            "Exhibit: Candidate models on the training period",
            style.table(
                cand, {"Caught in training, %": 0, "Chance, %": 0, "False episodes per asset-month": 3}
            ),
        )
        self.ap("E. Model governance", "aE")
        x("Exhibit: Registered model versions and gate results", self.versions())
        self.ap("F. Supervised layer", "aF")
        self.text("The training period holds 6 failures")
        self.ap("G. Validation results, July 2025 to March 2026", "aG")
        hv = d.g("evaluation_headline_validation")
        hv = hv[hv["operating_point"].astype(bool)]
        ops = set(zip(hv["method"], hv["sweep"].astype(float), strict=True))
        hv = pd.DataFrame(
            {
                "Method": hv["method"].map(LABEL),
                "Conditions": hv["conditions"],
                "Caught": hv["caught"],
                "Chance, %": 100 * hv["chance_share"],
                "Median warning, days": hv["warning_days_median"],
                "False episodes per asset-month": hv["false_per_asset_month"],
            }
        )
        bt = d.g("evaluation_headline_by_type_validation")
        bt = bt[[(m, float(w)) in ops for m, w in zip(bt["method"], bt["sweep"], strict=True)]]
        bt = pd.DataFrame(
            {
                "Method": bt["method"].map(LABEL),
                "Condition type": bt["condition_type"],
                "Conditions": bt["conditions"],
                "Caught": bt["caught"],
                "Median warning, days": bt["warning_days_median"],
            }
        )
        x(
            "Exhibit: Validation headline and results by condition type",
            style.table(hv, {"Chance, %": 0, "Median warning, days": 1, "False episodes per asset-month": 3})
            + style.table(bt, {"Median warning, days": 1}),
        )
        self.ap("H. Platform", "aH")
        self.text("The system runs on Azure Databricks")
        self.text("It is deployed from a Declarative Automation Bundle")
        x(
            "Exhibit: Architecture",
            self.flow(
                "architecture",
                [
                    ("Landing", ["Exports to ADLS", "Unity Catalog volume"]),
                    ("Pipeline", ["Bronze, silver, gold", "Expectations"]),
                    ("Job", ["Train, promote", "Score, monitor"]),
                    ("Registry", ["MLflow", "champion alias"]),
                    ("Monitoring", ["Data profiling", "Retraining rule"]),
                    ("Deployment", ["Bundle dev and prod", "GitHub Actions"]),
                ],
                h=3.2,
            ),
        )
        self.ap("I. Limitations", "aI")
        x("Exhibit: Limitations", self.limitations())
        body = "".join(self.parts)
        title, sub = t.field(S, "Title"), t.field(S, "Subtitle")
        style.write("reports/model_report.html", style.report_shell(title, sub, body, self.toc))
        return "docs/reports/model_report.html"

    def ap(self, label: str, anchor: str) -> None:
        rest = self.t.lines(S, f"{label}**")[1:]
        self.parts.append(f"<h3 id='{anchor}'>{label}</h3>")
        if rest:
            self.parts.append(self.t.render(rest))
        self.toc.append((anchor, label, "sub"))


def mdates(ts: pd.Timestamp) -> float:
    t = pd.Timestamp(ts)
    t = t.tz_convert(TZ).tz_localize(None) if t.tzinfo else t
    return md.date2num(t.to_pydatetime())


def build(d: Data, t: PageText) -> str:
    return ModelReport(d, t).build()

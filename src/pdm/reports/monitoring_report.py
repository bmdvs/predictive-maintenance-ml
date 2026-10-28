"""The monitoring report (docs/reports/monitoring_report.html): the page text of section C with its exhibits.

Inputs: the Data object (gold and monitoring snapshot tables, registry metadata) and the page text.
Outputs: the report and its figures.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pdm.reports import style
from pdm.reports.data import END, LABEL, METHODS, Q2, Data, local
from pdm.reports.model_report import exhibit_title
from pdm.reports.text import PageText

S = "C"
MONTHS = ("2026-04", "2026-05", "2026-06")
MONTH_LABEL = {m: pd.Timestamp(m + "-01").strftime("%B") for m in MONTHS}
WINDOW_ONLY_LABEL = "window only"


class MonitoringReport:
    def __init__(self, d: Data, t: PageText) -> None:
        self.d, self.t = d, t
        self.toc: list[tuple[str, ...]] = []
        self.parts: list[str] = []
        mon = d.cfg["monitoring"]
        self.lo, self.hi = mon["tolerance"]["alerts_per_asset_month_ratio"]
        self.thr = mon["drift"]["feature_threshold"]
        self.share = mon["drift"]["feature_share_drifted"]
        self.perf = d.m("event_performance")
        self.rd = d.m("retraining_decision")
        self.drift = d.m("feature_drift")
        dq = d.m("data_quality_monthly")
        self.dq = dq[dq["month"].isin(MONTHS)].copy()
        self.dq["exception"] = (self.dq["implausible_rows"] > 0) | (self.dq["no_measurement_share"] > 0.30)

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
        return style.save(f, f"monitoring_{name}", name, up=1)

    # ---------------------------------------------------------------- derived
    def reference_rate(self, method: str) -> float:
        p = self.perf[self.perf["method"] == method].merge(
            self.rd[self.rd["method"] == method][["month", "alerts_rate_ratio"]], on="month"
        )
        p = p[p["alerts_rate_ratio"] > 0]
        return float((p["alerts_per_asset_month"] / p["alerts_rate_ratio"]).median()) if len(p) else 0.0

    @property
    def point_drift(self) -> pd.DataFrame:
        f = self.drift[self.drift["feature"] != "nbm_score"]
        g = (
            f.assign(over=f["wasserstein_scaled"] > self.thr)
            .groupby(["month", "point_id"])["over"]
            .mean()
            .reset_index()
        )
        g["flagged"] = g["over"] > self.share
        return g

    # ---------------------------------------------------------------- exhibits
    def tiles(self) -> str:
        m = "model_layer_instrumented"
        ref = self.reference_rate(m)
        pdr = self.point_drift
        out = ""
        for mo in MONTHS:
            p = self.perf[(self.perf["method"] == m) & (self.perf["month"] == mo)].iloc[0]
            recall = (
                "no conditions"
                if pd.isna(p.recall)
                else f"{int(round(p.recall * p.conditions))} of {int(p.conditions)}"
            )
            items = [
                ("Event recall", recall),
                ("Alerts per asset-month", f"{p.alerts_per_asset_month:.2f} against {ref:.2f}"),
                ("Points flagged for drift", f"{int(pdr[pdr['month'] == mo]['flagged'].sum())} of 20"),
                ("Data-quality exceptions", str(int(self.dq[self.dq["month"] == mo]["exception"].sum()))),
            ]
            out += f"<div class='kpi-head'>{MONTH_LABEL[mo]} 2026</div>" + style.tiles(items)
        return out

    def rates(self) -> str:
        f, axs = style.fig(h=2.9, w=8.6, ncols=4, sharey=False)
        x = np.arange(len(MONTHS))
        for ax, m in zip(axs, METHODS, strict=True):
            p = self.perf[self.perf["method"] == m].set_index("month").reindex(MONTHS)
            ref = self.reference_rate(m)
            ax.axhspan(self.lo * ref, self.hi * ref, color="#E6EEF4", lw=0, label="Tolerance band")
            ax.axhline(ref, color=style.DARK_GREY, lw=1, ls="--", label="Validation rate")
            ax.plot(
                x,
                p["alerts_per_asset_month"],
                marker="o",
                color=style.METHOD_COLOR[m],
                lw=1.6,
                label="Shadow month",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([MONTH_LABEL[mo][:3] for mo in MONTHS], fontsize=8)
            ax.set_title(LABEL[m], fontsize=9)
            ax.set_ylim(0, max(0.65, float(p["alerts_per_asset_month"].max()) * 1.15, self.hi * ref * 1.1))
        axs[0].set_ylabel("Alerts per asset-month", fontsize=8)
        f.tight_layout(rect=(0, 0.1, 1, 1))
        f.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        return self.img(f, "alert_rates")

    def performance(self) -> str:
        rows = []
        for m in METHODS:
            for mo in MONTHS:
                p = self.perf[(self.perf["method"] == m) & (self.perf["month"] == mo)].iloc[0]
                r = self.rd[(self.rd["method"] == m) & (self.rd["month"] == mo)].iloc[0]
                rows.append(
                    {
                        "Method": LABEL[m],
                        "Month": MONTH_LABEL[mo],
                        "Conditions": int(p.conditions),
                        "Caught": "" if pd.isna(p.recall) else int(round(p.recall * p.conditions)),
                        "Median warning, days": p.lead_days_median,
                        "Alerts": int(p.alerts),
                        "Alerts per asset-month": p.alerts_per_asset_month,
                        "Ratio to validation": r.alerts_rate_ratio,
                        "Primary trigger": "fires" if bool(r.primary_trigger) else "",
                    }
                )
        return style.table(
            pd.DataFrame(rows),
            {"Median warning, days": 1, "Alerts per asset-month": 2, "Ratio to validation": 2},
            cls="data nowrap1",
        )

    def heatmap(self) -> str:
        hm = self.point_drift.pivot(index="point_id", columns="month", values="over").reindex(
            columns=list(MONTHS)
        )
        f, ax = style.fig(h=5.0, w=6.4)
        ax.grid(False)
        im = ax.imshow(hm.to_numpy(dtype=float), aspect="auto", cmap="Oranges", vmin=0, vmax=1)
        ax.set_yticks(range(len(hm)))
        ax.set_yticklabels(hm.index, fontsize=7)
        ax.set_xticks(range(len(MONTHS)))
        ax.set_xticklabels([MONTH_LABEL[mo] for mo in MONTHS], fontsize=8)
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                v = hm.iat[i, j]
                ax.text(
                    j,
                    i,
                    "" if pd.isna(v) else f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if v > 0.6 else "#222222",
                )
        cb = f.colorbar(im, ax=ax, shrink=0.6)
        cb.set_label("Share of features drifted", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        return self.img(f, "feature_drift")

    def quality(self) -> str:
        q = self.dq.pivot_table(
            index="point_id", columns="month", values=["no_measurement_share", "implausible_rows"]
        )
        f, ax = style.fig(h=3.0, w=8.6)
        nm = q["no_measurement_share"].reindex(columns=list(MONTHS))
        x = np.arange(len(nm))
        for k, (mo, colr) in enumerate(
            zip(MONTHS, (style.LIGHT_BLUE, style.ACCENT, style.BRAND_BLUE), strict=True)
        ):
            ax.bar(x + (k - 1) * 0.27, nm[mo] * 100, width=0.27, color=colr, label=MONTH_LABEL[mo])
        ax.axhline(30, color=style.RED, lw=0.8, ls="--", label="Exception level")
        ax.set_xticks(x)
        ax.set_xticklabels(nm.index, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("No-measurement share, %", fontsize=8)
        f.tight_layout(rect=(0, 0.1, 1, 1))
        f.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8)
        img = self.img(f, "data_quality")
        ex = self.dq[self.dq["exception"]]
        t = ex.pivot_table(
            index="point_id", columns="month", values="implausible_rows", aggfunc="sum"
        ).reindex(columns=list(MONTHS))
        nmx = ex.pivot_table(
            index="point_id", columns="month", values="no_measurement_share", aggfunc="max"
        ).reindex(columns=list(MONTHS))
        rows = []
        for p in t.index:
            row = {"Point": p}
            for mo in MONTHS:
                v, s = t.at[p, mo], nmx.at[p, mo]
                if pd.isna(v):
                    row[MONTH_LABEL[mo]] = ""
                    continue
                parts = [f"{int(v)} implausible"] if v > 0 else []
                if s > 0.30:
                    parts.append(f"{s:.0%} no measurement")
                row[MONTH_LABEL[mo]] = ", ".join(parts)
            rows.append(row)
        return img + style.table(pd.DataFrame(rows))

    def versions(self) -> pd.DataFrame:
        reg = json.loads(self.d.st.file("mlflow/registry.json").read_text(encoding="utf-8"))
        name = next(iter(reg))
        champion = str(reg[name]["aliases"].get("champion"))
        rows = []
        for v in reg[name]["versions"]:
            tg = v["tags"]
            if tg.get("false_episodes_month", "") < "2026-04":
                continue
            rows.append(
                {
                    "Cycle": pd.Timestamp(tg["false_episodes_month"] + "-01").strftime("%B %Y"),
                    "Version": int(v["version"]),
                    "Selection": tg.get("gate_selection", ""),
                    "Parity": tg.get("gate_parity", ""),
                    "Segments": tg.get("gate_segments", ""),
                    "False episodes": tg.get("gate_false_episodes", ""),
                    "Rate": float(tg["false_episodes_rate"]),
                    "Budget": float(tg["false_episodes_budget"]),
                    "Status": tg.get("status", "").replace("_", " "),
                    "Champion at the record end": "yes" if str(v["version"]) == champion else "",
                }
            )
        return pd.DataFrame(rows).sort_values("Version")

    def resets(self) -> str:
        nm = self.d.g("nbm_models")
        nm["segment_start"] = pd.to_datetime(nm["segment_start"], utc=True)
        r = nm[(nm["segment_start"] >= Q2) & (nm["segment_start"] < END)].sort_values(
            ["segment_start", "point_id"]
        )
        wo = self.d.wo
        rows = []
        for x in r.itertuples():
            rows.append(
                {
                    "Point": x.point_id,
                    "Baseline start": local(x.segment_start),
                    "Reason": str(x.reason).replace("_", " "),
                    "Drift reference": self.reference_of(x.point_id),
                }
            )
        _ = wo
        return style.table(pd.DataFrame(rows))

    def reference_of(self, point: str) -> str:
        r = self.drift[(self.drift["point_id"] == point) & (self.drift["month"] == MONTHS[-1])]["reference"]
        if not len(r):
            return "still forming at the record end"
        return str(r.iat[0])

    def rule(self) -> str:
        rows = [
            (
                "Primary trigger",
                f"Event recall more than {self.d.cfg['monitoring']['tolerance']['recall_drop']:.2f} below the validation recall, or alerts per asset-month outside {self.lo} to {self.hi} times the validation rate, for {self.d.cfg['monitoring']['consecutive_months']} consecutive months",
            ),
            (
                "Secondary trigger",
                f"Share of point-features, or of point scores, with scaled Wasserstein distance above {self.thr} greater than {self.share:.2f}",
            ),
            (
                "Verdict",
                "Retrain when the primary trigger fires; the secondary trigger is reported beside it",
            ),
            ("Action", str(self.rd.loc[self.rd["retrain"].astype(bool), "action"].iat[0]).capitalize()),
        ]
        rule = style.table(pd.DataFrame(rows, columns=["Part", "Definition"]))
        rd = self.rd.copy()
        rows = []
        for x in rd.itertuples():
            if x.method not in METHODS:
                continue
            rows.append(
                {
                    "Method": LABEL[x.method],
                    "Month": MONTH_LABEL[x.month],
                    "Recall": x.recall,
                    "Validation recall": x.recall_reference,
                    "Alert-rate ratio": x.alerts_rate_ratio,
                    "Primary": "fires" if x.primary_trigger else "",
                    "Feature drift share": x.drift_share,
                    "Score drift share": x.score_drift_share,
                    "Secondary": "fires" if x.secondary_trigger else "",
                    "Retrain": "yes" if x.retrain else "",
                }
            )
        return rule + style.table(
            pd.DataFrame(rows),
            {
                "Recall": 2,
                "Validation recall": 2,
                "Alert-rate ratio": 2,
                "Feature drift share": 3,
                "Score drift share": 3,
            },
            cls="data nowrap1",
        )

    def references(self) -> str:
        defs = pd.DataFrame(
            [
                (
                    "Validation",
                    "July 2025 to March 2026, for points not re-baselined after the validation start",
                ),
                (
                    "Re-baseline window plus scored days",
                    "For a point re-baselined after the validation start: its latest post-repair re-baseline window that passed the post-repair check plus the first 30 scored days with no open case on the point",
                ),
                (
                    "Window only",
                    "The re-baseline window alone when a case opens within those 30 days; the window is not scored, so the point has no score drift and feature drift is against the narrow window",
                ),
                (
                    "Compared hours",
                    "From the reference's end on; hours inside a reference, or after the start of one still forming at the horizon, are not compared",
                ),
            ],
            columns=["Reference", "Definition"],
        )
        last = self.drift.sort_values("month").groupby("point_id")["reference"].last()
        pts = sorted(self.d.sensors["point_id"])
        ref = pd.DataFrame({"Point": pts, "Reference in June": [last.get(p, "still forming") for p in pts]})
        return style.table(defs) + style.table(ref)

    def per_point(self) -> str:
        f = self.drift[self.drift["feature"] != "nbm_score"]
        g = (
            f.assign(over=f["wasserstein_scaled"] > self.thr)
            .groupby(["point_id", "month"])
            .agg(share=("over", "mean"), median=("wasserstein_scaled", "median"))
        )
        sc = self.drift[self.drift["feature"] == "nbm_score"].set_index(["point_id", "month"])[
            "wasserstein_scaled"
        ]
        rows = []
        for p in sorted(self.d.sensors["point_id"]):
            row = {"Point": p}
            for mo in MONTHS:
                row[f"{MONTH_LABEL[mo]} feature share"] = g["share"].get((p, mo), np.nan)
                row[f"{MONTH_LABEL[mo]} score"] = sc.get((p, mo), np.nan)
            rows.append(row)
        drift = style.table(pd.DataFrame(rows), {c: 2 for r in rows[:1] for c in r if c != "Point"})
        q = self.dq.pivot_table(
            index="point_id", columns="month", values=["no_measurement_share", "implausible_rows"]
        )
        rows = []
        for p in q.index:
            row = {"Point": p}
            for mo in MONTHS:
                row[f"{MONTH_LABEL[mo]} no measurement, %"] = 100 * q[("no_measurement_share", mo)].get(
                    p, np.nan
                )
                row[f"{MONTH_LABEL[mo]} implausible"] = int(q[("implausible_rows", mo)].get(p, 0))
            rows.append(row)
        dq = style.table(pd.DataFrame(rows), {c: 0 for r in rows[:1] for c in r if c != "Point"})
        return drift + dq

    # ---------------------------------------------------------------- page
    def build(self) -> str:
        t, x = self.t, self.exhibit
        self.h(2, "1. Summary", "s1")
        self.text("The monitoring rule calls for")
        self.text("The drift trigger fired in April")
        x("Exhibit: Monitoring by month", self.tiles())
        self.parts.append("<h2 id='s2'>2. Monitoring results</h2>")
        self.toc.append(("s2", "2. Monitoring results"))
        self.h(3, "2.1 Performance and alert rates", "s21", True)
        self.text("The model layer caught both conditions")
        self.text("The vendor rules stayed within")
        x("Exhibit: Alerts per asset-month", self.rates())
        x("Exhibit: Event performance by month", self.performance())
        self.h(3, "2.2 Drift and data quality", "s22", True)
        self.text("Feature drift flagged")
        self.text("The drift trigger fired for two reasons")
        self.text("Data-quality exceptions rose")
        x("Exhibit: Feature drift by point", self.heatmap())
        x("Exhibit: Data quality by point", self.quality())
        self.h(3, "2.3 Model updates and baseline resets", "s23", True)
        self.text("Model versions.")
        x("Exhibit: Champion per monthly cycle", style.table(self.versions(), {"Rate": 3, "Budget": 3}))
        x("Exhibit: Post-repair baseline resets", self.resets())
        self.t.lines(S, "Appendix")
        self.parts.append("<h2 id='appendix'>Appendix</h2>")
        self.toc.append(("appendix", "Appendix"))
        self.parts.append("<h3 id='aA'>Retraining rule</h3>")
        self.toc.append(("aA", "Retraining rule", "sub"))
        self.text("Acted-on alerts remove")
        self.parts.append(self.rule())
        self.parts.append("<h3 id='aB'>Drift references</h3>")
        self.toc.append(("aB", "Drift references", "sub"))
        self.parts.append(self.references())
        self.parts.append("<h3 id='aC'>Per-point tables</h3>")
        self.toc.append(("aC", "Per-point tables", "sub"))
        self.parts.append(self.per_point())
        body = "".join(self.parts)
        style.write(
            "reports/monitoring_report.html",
            style.report_shell(t.field(S, "Title"), t.field(S, "Subtitle"), body, self.toc),
        )
        return "docs/reports/monitoring_report.html"


def build(d: Data, t: PageText) -> str:
    return MonitoringReport(d, t).build()

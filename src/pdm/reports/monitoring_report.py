"""The monitoring report (docs/reports/monitoring_report.html): the page text of section C with its exhibits.

Inputs: the prod monitoring tables of the version 13 run (build/monitoring_cache: event performance, feature
drift, data quality, re-centering events, the champion's registry tags), the monthly evaluation on the headline
definitions (build/report_data/monitoring_month.parquet, the same evaluation as the model report), the rebased
references, score drift and verdicts (build/report_data/monitoring_references, monitoring_score_drift and
monitoring_decision: the ML Model's alert-rate and score references from the fold-2 model's out-of-sample scores,
reference catch shares on the headline definitions over the training folds), config.yaml and the page text.
Outputs: the report and its figures.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pdm import paths
from pdm.reports import style
from pdm.reports.data import Data
from pdm.reports.model_report import exhibit_title
from pdm.reports.text import PageText

S = "C"
CACHE = paths.REPO_ROOT / "build" / "monitoring_cache"
DATA = paths.REPO_ROOT / "build" / "report_data"
MONTHS = tuple(pd.date_range("2025-07-01", "2026-06-01", freq="MS").strftime("%Y-%m"))
MONTH_LABEL = {m: pd.Timestamp(m + "-01").strftime("%b %Y") for m in MONTHS}
SHORT = {m: pd.Timestamp(m + "-01").strftime("%b") for m in MONTHS}
METHODS = ("Vendor Rules", "ML Model")
COLOR = {"Vendor Rules": style.DARK_GREY, "ML Model": style.BRAND_BLUE}
TZ = "America/Chicago"


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"{name}.parquet")


class MonitoringReport:
    def __init__(self, d: Data, t: PageText) -> None:
        self.d, self.t = d, t
        self.toc: list[tuple[str, ...]] = []
        self.parts: list[str] = []
        mon = d.cfg["monitoring"]
        self.lo, self.hi = mon["tolerance"]["alerts_per_asset_month_ratio"]
        self.thr = mon["drift"]["feature_threshold"]
        self.share = mon["drift"]["feature_share_drifted"]
        self.perf = _read("event_performance")
        self.rd = pd.read_parquet(DATA / "monitoring_decision.parquet")
        self.ref = pd.read_parquet(DATA / "monitoring_references.parquet").set_index("method")
        fd = _read("feature_drift")
        self.drift = pd.concat(
            [fd[fd["feature"] != "nbm_score"], pd.read_parquet(DATA / "monitoring_score_drift.parquet")],
            ignore_index=True,
        )
        self.mm = pd.read_parquet(DATA / "monitoring_month.parquet")
        dq = _read("data_quality_monthly")
        self.dq = dq[dq["month"].isin(MONTHS)].copy()
        self.dq["exception"] = (self.dq["implausible_rows"] > 0) | (self.dq["no_measurement_share"] > 0.30)
        self.points = sorted(self.dq["point_id"].unique())
        self.check_verdicts()

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
    def headline(self, method: str, month: str) -> pd.Series | None:
        r = self.mm[(self.mm["method"] == method) & (self.mm["month"] == month)]
        return r.iloc[0] if len(r) else None

    def recall(self, method: str, month: str) -> float:
        """Catch share on the headline definitions."""
        h = self.headline(method, month)
        return h["caught"] / h["conditions"] if h["conditions"] else np.nan

    def check_verdicts(self) -> None:
        """The verdicts read in equal the rule applied here to the headline catch shares and the references."""
        tol = self.d.cfg["monitoring"]["tolerance"]["recall_drop"]
        need = self.d.cfg["monitoring"]["consecutive_months"]
        for m in METHODS:
            streak = 0
            ref = float(self.ref.at[m, "recall"])
            for x in self.rd[self.rd["method"] == m].sort_values("month").itertuples():
                r = self.recall(m, x.month)
                bad = (pd.notna(r) and ref - r > tol) or not (self.lo <= x.alerts_rate_ratio <= self.hi)
                streak = streak + 1 if bad else 0
                assert (streak >= need) == bool(x.retrain), (m, x.month)

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
        m = "ML Model"
        ref = float(self.ref.at[m, "alerts_per_asset_month"])
        pdr = self.point_drift
        out = ""
        for mo in MONTHS:
            p = self.perf[(self.perf["method"] == m) & (self.perf["month"] == mo)].iloc[0]
            h = self.headline(m, mo)
            caught = (
                "no conditions" if not h["conditions"] else f"{int(h['caught'])} of {int(h['conditions'])}"
            )
            items = [
                ("Conditions caught", caught),
                ("Alerts per asset-month", f"{p.alerts_per_asset_month:.2f} against {ref:.3f}"),
                (
                    "Points flagged for drift",
                    f"{int(pdr[pdr['month'] == mo]['flagged'].sum())} of {len(self.points)}",
                ),
                ("Data-quality exceptions", str(int(self.dq[self.dq["month"] == mo]["exception"].sum()))),
            ]
            out += f"<div class='kpi-head'>{MONTH_LABEL[mo]}</div>" + style.tiles(items)
        return out

    def rates(self) -> str:
        f, axs = style.fig(h=2.9, w=6.4, ncols=2, sharey=False)
        x = np.arange(len(MONTHS))
        for ax, m in zip(axs, METHODS, strict=True):
            p = self.perf[self.perf["method"] == m].set_index("month").reindex(MONTHS)
            ref = float(self.ref.at[m, "alerts_per_asset_month"])
            ax.axhspan(self.lo * ref, self.hi * ref, color="#E6EEF4", lw=0, label="Tolerance band")
            ax.axhline(ref, color=style.DARK_GREY, lw=1, ls="--", label="Training reference rate")
            ax.plot(
                x,
                p["alerts_per_asset_month"],
                marker="o",
                ms=3,
                color=COLOR[m],
                lw=1.4,
                label="Evaluation month",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT[mo][0] for mo in MONTHS], fontsize=8)
            ax.set_title(m, fontsize=9)
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
                h = self.headline(m, mo)
                rows.append(
                    {
                        "Method": m,
                        "Month": MONTH_LABEL[mo],
                        "Conditions": int(p.conditions),
                        "Caught": int(h["caught"]),
                        "Median warning, days": h["warning_days_median"],
                        "Alerts": int(p.alerts),
                        "Alerts per asset-month": p.alerts_per_asset_month,
                        "False alerts per asset-month": h["false_per_asset_month"],
                        "False ALARMs per asset-month": h["false_alarms_per_asset_month"],
                        "Alert-hours per asset-month": h["alert_hours_per_asset_month"],
                        "Ratio to reference": r.alerts_rate_ratio,
                        "Primary trigger": "fires" if bool(r.primary_trigger) else "",
                    }
                )
        return style.table(
            pd.DataFrame(rows),
            {
                "Median warning, days": 1,
                "Alerts per asset-month": 2,
                "False alerts per asset-month": 3,
                "False ALARMs per asset-month": 3,
                "Alert-hours per asset-month": 0,
                "Ratio to reference": 2,
            },
            cls="data nowrap1",
        )

    def _heat(self, hm: pd.DataFrame, name: str, label: str, vmax: float, fmt: str) -> str:
        f, ax = style.fig(h=8.6, w=8.6)
        ax.grid(False)
        im = ax.imshow(hm.to_numpy(dtype=float), aspect="auto", cmap="Oranges", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(hm)))
        ax.set_yticklabels(hm.index, fontsize=7)
        ax.set_xticks(range(len(MONTHS)))
        ax.set_xticklabels([SHORT[mo] for mo in MONTHS], fontsize=8)
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                v = hm.iat[i, j]
                ax.text(
                    j,
                    i,
                    "" if pd.isna(v) else format(v, fmt),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if pd.notna(v) and v > 0.6 * vmax else "#222222",
                )
        cb = f.colorbar(im, ax=ax, shrink=0.5)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        return self.img(f, name)

    def heatmap(self) -> str:
        hm = self.point_drift.pivot(index="point_id", columns="month", values="over").reindex(
            index=self.points, columns=list(MONTHS)
        )
        return self._heat(hm, "feature_drift", "Share of features drifted", 1.0, ".1f")

    def quality(self) -> str:
        nm = self.dq.pivot_table(index="point_id", columns="month", values="no_measurement_share").reindex(
            index=self.points, columns=list(MONTHS)
        )
        img = self._heat(100 * nm, "data_quality", "No-measurement share, %", 100.0, ".0f")
        rows = []
        for p, g in self.dq.groupby("point_id"):
            ex = g[g["exception"]]
            if not len(ex):
                continue
            rows.append(
                {
                    "Point": p,
                    "Months with implausible rows": int((ex["implausible_rows"] > 0).sum()),
                    "Implausible rows": int(ex["implausible_rows"].sum()),
                    "Months above 30% no measurement": int((ex["no_measurement_share"] > 0.30).sum()),
                }
            )
        return img + style.table(pd.DataFrame(rows))

    def versions(self) -> pd.DataFrame:
        mv = json.loads((CACHE / "registry_v13.json").read_text(encoding="utf-8"))
        mv = mv.get("model_version", mv)
        tg = {x["key"]: x["value"] for x in mv.get("tags", [])}
        champion = any(a.get("alias") == "champion" for a in mv.get("aliases", []))
        return pd.DataFrame(
            [
                {
                    "Version": int(mv["version"]),
                    "Settings": tg.get("settings", ""),
                    "Selection": tg.get("gate_selection", ""),
                    "Parity": tg.get("gate_parity", ""),
                    "Points": tg.get("gate_points", ""),
                    "Numerics": tg.get("gate_numerics", ""),
                    "False episodes": tg.get("gate_false_episodes", ""),
                    "Rate": float(tg["false_episodes_rate"]),
                    "Limit": float(tg["false_episodes_limit"]),
                    "Status": tg.get("status", "").replace("_", " "),
                    "Champion for the year": "yes" if champion else "",
                }
            ]
        )

    def resets(self) -> str:
        r = _read("recentering_events")
        for c in ("segment_start", "level_ready"):
            r[c] = pd.to_datetime(r[c], utc=True)
        r = r.sort_values(["segment_start", "point_id"])

        def loc(ts: pd.Timestamp) -> str:
            return "" if pd.isna(ts) else ts.tz_convert(TZ).strftime("%Y-%m-%d")

        rows = [
            {
                "Point": x.point_id,
                "Start": loc(x.segment_start),
                "Reason": str(x.reason).replace("_", " "),
                "Level set": loc(x.level_ready) or "no level by the record end",
                "Plausibility flag": "flagged" if bool(x.plausibility_flag) else "",
                "Drift reference in June": self.reference_of(x.point_id),
            }
            for x in r.itertuples()
        ]
        return style.table(pd.DataFrame(rows))

    def reference_of(self, point: str) -> str:
        r = self.drift[(self.drift["point_id"] == point) & (self.drift["month"] == MONTHS[-1])]["reference"]
        if not len(r):
            return "still forming at the record end"
        return str(r.iat[0])

    def rule(self) -> str:
        tol = self.d.cfg["monitoring"]["tolerance"]
        win = str(self.ref["reference_window"].iat[0])
        ml_win = str(self.ref.at["ML Model", "rate_window"])
        rows = [
            (
                "Primary trigger",
                f"Catch share more than {tol['recall_drop']:.2f} below the reference catch share on the training folds (July 2024 to June 2025), or alerts per asset-month outside {self.lo} to {self.hi} times the training reference rate ({win}; for the ML Model, the fold-2 model's alerts from {ml_win}), for {self.d.cfg['monitoring']['consecutive_months']} consecutive months",
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
        rows = []
        for m in METHODS:
            for x in self.rd[self.rd["method"] == m].sort_values("month").itertuples():
                rows.append(
                    {
                        "Method": m,
                        "Month": MONTH_LABEL[x.month],
                        "Catch share": self.recall(m, x.month),
                        "Reference catch share": float(self.ref.at[m, "recall"]),
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
                "Catch share": 2,
                "Reference catch share": 2,
                "Alert-rate ratio": 2,
                "Feature drift share": 3,
                "Score drift share": 3,
            },
            cls="data nowrap1",
        )

    def references(self) -> str:
        win = str(self.ref["reference_window"].iat[0])
        defs = pd.DataFrame(
            [
                (
                    "Training reference",
                    f"{win}, the last six months of training, for points not re-centered after its start; the ML Model score from the fold-2 model, scored from {self.ref.at['ML Model', 'rate_window'].split(' to ')[0]}",
                ),
                (
                    "Re-centering plus scored days",
                    "For a point re-centered after a repair or remount: its re-centering window plus the first 30 scored days with no open episode on the point",
                ),
                (
                    "Window only",
                    "The re-centering window alone when an episode opens within those 30 days; the window is not scored, so the point has no score drift and feature drift is against the narrow window",
                ),
                (
                    "Compared hours",
                    "From the reference's end on; hours inside a reference, or after the start of one still forming at the horizon, are not compared",
                ),
            ],
            columns=["Reference", "Definition"],
        )
        last = self.drift.sort_values("month").groupby("point_id")["reference"].last()
        ref = pd.DataFrame(
            {"Point": self.points, "Reference in June": [last.get(p, "still forming") for p in self.points]}
        )
        return style.table(defs) + style.table(ref)

    def per_point(self) -> str:
        f = self.drift[self.drift["feature"] != "nbm_score"]
        share = (
            f.assign(over=f["wasserstein_scaled"] > self.thr)
            .groupby(["point_id", "month"])["over"]
            .mean()
            .unstack()
            .reindex(index=self.points, columns=list(MONTHS))
        )
        sc = (
            self.drift[self.drift["feature"] == "nbm_score"]
            .set_index(["point_id", "month"])["wasserstein_scaled"]
            .unstack()
            .reindex(index=self.points, columns=list(MONTHS))
        )
        nm = 100 * self.dq.pivot_table(
            index="point_id", columns="month", values="no_measurement_share"
        ).reindex(index=self.points, columns=list(MONTHS))
        im = (
            self.dq.pivot_table(index="point_id", columns="month", values="implausible_rows", aggfunc="sum")
            .reindex(index=self.points, columns=list(MONTHS))
            .fillna(0)
        )
        out = ""
        for title, t, dec in (
            ("Feature drift share", share, 2),
            ("Score drift, scaled distance", sc, 2),
            ("No measurement, %", nm, 0),
            ("Implausible rows", im, 0),
        ):
            df = t.rename(columns=SHORT).reset_index().rename(columns={"point_id": "Point"})
            out += f"<div class='kpi-head'>{title}</div>" + style.table(
                df, {SHORT[m]: dec for m in MONTHS}, cls="data nowrap1"
            )
        return out

    # ---------------------------------------------------------------- page
    def build(self) -> str:
        t, x = self.t, self.exhibit
        self.h(2, "1. Summary", "s1")
        self.text("The monitoring rule calls for")
        self.text("The drift trigger fired in every")
        x("Exhibit: Monitoring by month", self.tiles())
        self.parts.append("<h2 id='s2'>2. Monitoring results</h2>")
        self.toc.append(("s2", "2. Monitoring results"))
        self.h(3, "2.1 Performance and alert rates", "s21", True)
        self.text("The ML Model caught")
        self.text("Vendor Rules stayed within")
        x("Exhibit: Alerts per asset-month", self.rates())
        x("Exhibit: Event performance by month", self.performance())
        self.h(3, "2.2 Drift and data quality", "s22", True)
        self.text("Feature drift flagged")
        self.text("The drift trigger fired for two reasons")
        self.text("Data-quality exceptions ran")
        x("Exhibit: Feature drift by point", self.heatmap())
        x("Exhibit: Data quality by point", self.quality())
        self.h(3, "2.3 Champion and re-centering", "s23", True)
        self.text("Model versions.")
        x("Exhibit: Champion,", style.table(self.versions(), {"Rate": 3, "Limit": 3}))
        x("Exhibit: Post-repair re-centering", self.resets())
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

"""The monitoring report (docs/reports/monitoring_report.html): the ML Model's status and retraining decision, the
monthly results behind it, a monitoring log and the rule's definitions, with the page text of section C.

Inputs: the prod monitoring tables of the version 13 run (build/monitoring_cache: event performance, feature
drift, data quality, re-centering events, the champion's registry tags), the monthly evaluation on the headline
definitions (build/report_data/monitoring_month.parquet, the same evaluation as the model report), the rebased
references, score drift and verdicts (build/report_data/monitoring_references, monitoring_score_drift and
monitoring_decision: the ML Model's alert-rate and score references from the fold-2 model's out-of-sample scores,
reference catch shares on the headline definitions over the training folds), config.yaml and the page text.
Outputs: the report and its figures. Presentation only: every status is the configured rule applied to these
tables, and the builder checks it against the verdicts the monitor wrote.
"""

from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

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
ML = "ML Model"
TZ = "America/Chicago"
# status of a rule in a month: passes, condition met one month (or secondary trigger), primary trigger, not evaluated
STATUS = {
    "pass": (style.GREEN, "✓", "Pass"),
    "watch": (style.AMBER, "◐", "Watch"),
    "refit": (style.RED, "⚠", "Refit"),
    "none": (style.GREY, "·", "Not evaluated"),
}
HMC_SHADE, CMP_SHADE = "#7A7A7A", "#C4C4C4"
LABEL_H = 0.06  # height of a count label on the catch-share axis (0 to 1.15)


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"{name}.parquet")


def asset_of(point: str) -> str:
    for end in ("-F", "-R", "-MDE", "-AE"):
        if point.endswith(end):
            return point[: -len(end)]
    return point


class MonitoringReport:
    def __init__(self, d: Data, t: PageText) -> None:
        self.d, self.t = d, t
        self.toc: list[tuple[str, ...]] = []
        self.parts: list[str] = []
        mon = d.cfg["monitoring"]
        self.lo, self.hi = mon["tolerance"]["alerts_per_asset_month_ratio"]
        self.tol = mon["tolerance"]["recall_drop"]
        self.need = mon["consecutive_months"]
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
        self.ml = self.rd[self.rd["method"] == ML].set_index("month").reindex(list(MONTHS))
        self.status = self.statuses()
        self.check_verdicts()

    # ---------------------------------------------------------------- page assembly
    def text(self, start: str) -> None:
        self.parts.append(self.t.html(S, start))

    def h(self, level: int, start: str, anchor: str, sub: bool = False) -> None:
        title = self.t.heading(S, start).strip("*")
        self.parts.append(f"<h{level} id='{anchor}'>{html.escape(title)}</h{level}>")
        self.toc.append((anchor, title, "sub") if sub else (anchor, title))

    def exhibit(self, start: str, html_: str) -> None:
        self.parts.append(style.chart(exhibit_title(self.t, S, start), html_))

    def img(self, f, name: str) -> str:
        return style.save(f, f"monitoring_{name}", name.replace("_", " "), up=1)

    # ---------------------------------------------------------------- the rule, month by month
    def headline(self, method: str, month: str) -> pd.Series:
        return self.mm[(self.mm["method"] == method) & (self.mm["month"] == month)].iloc[0]

    def catch(self, method: str, month: str) -> tuple[int, int, float]:
        h = self.headline(method, month)
        n, c = int(h["conditions"]), int(h["caught"])
        return c, n, (c / n if n else np.nan)

    @property
    def ref_catch(self) -> float:
        return float(self.ref.at[ML, "recall"])

    @property
    def limit(self) -> float:
        return self.ref_catch - self.tol

    def statuses(self) -> pd.DataFrame:
        """Per month, each rule's status for the ML Model: a primary condition met one month is watch, met for the
        configured run of consecutive months is refit; a month with no conditions does not evaluate catch share;
        the drift rules are watch whenever their share is above the trigger."""
        rows, sc, sa = [], 0, 0
        for mo in MONTHS:
            c, n, cs = self.catch(ML, mo)
            ratio = float(self.ml.at[mo, "alerts_rate_ratio"])
            cbad = bool(n) and self.ref_catch - cs > self.tol
            abad = not (self.lo <= ratio <= self.hi)
            sc, sa = (sc + 1 if cbad else 0), (sa + 1 if abad else 0)

            def st(bad: bool, streak: int) -> str:
                return "refit" if streak >= self.need else ("watch" if bad else "pass")

            fsh, ssh = float(self.ml.at[mo, "drift_share"]), float(self.ml.at[mo, "score_drift_share"])
            rows.append(
                {
                    "month": mo,
                    "caught": c,
                    "conditions": n,
                    "catch_share": cs,
                    "catch": st(cbad, sc) if n else "none",
                    "catch_bad": cbad,
                    "ratio": ratio,
                    "rate": st(abad, sa),
                    "rate_bad": abad,
                    "feature_share": fsh,
                    "feature": "watch" if fsh > self.share else "pass",
                    "score_share": ssh,
                    "score": "watch" if ssh > self.share else "pass",
                }
            )
        return pd.DataFrame(rows).set_index("month")

    def check_verdicts(self) -> None:
        """The verdicts the monitor wrote equal the rule applied here to the headline catch shares and the
        references, for both methods; the per-rule statuses agree with them."""
        for m in METHODS:
            streak = 0
            ref = float(self.ref.at[m, "recall"])
            for x in self.rd[self.rd["method"] == m].sort_values("month").itertuples():
                _, n, r = self.catch(m, x.month)
                bad = (bool(n) and ref - r > self.tol) or not (self.lo <= x.alerts_rate_ratio <= self.hi)
                streak = streak + 1 if bad else 0
                assert (streak >= self.need) == bool(x.retrain), (m, x.month)
        s = self.status
        refit = (s["catch"] == "refit") | (s["rate"] == "refit")
        assert refit.equals(self.ml["retrain"].astype(bool)), "per-rule statuses differ from the verdicts"

    def verdict(self, mo: str) -> str:
        s = self.status.loc[mo]
        if bool(self.ml.at[mo, "retrain"]):
            return "refit"
        return "watch" if s["catch_bad"] or s["rate_bad"] else "pass"

    @staticmethod
    def sym(status: str, text: str = "", bold: bool = True) -> str:
        c, mark, _ = STATUS[status]
        w = "font-weight:700;" if bold else ""
        return f"<span style='color:{c};{w}'>{mark}{(' ' + html.escape(text)) if text else ''}</span>"

    # ---------------------------------------------------------------- section 1
    def status_block(self) -> str:
        fired = [SHORT[m] for m in MONTHS if bool(self.ml.at[m, "retrain"])]
        first = next(m for m in MONTHS if bool(self.ml.at[m, "retrain"]))
        yr = self.mm[self.mm["method"] == ML]
        caught, conds = int(yr["caught"].sum()), int(yr["conditions"].sum())
        rate_ref = float(self.ref.at[ML, "alerts_per_asset_month"])
        rate = float(self.perf[self.perf["method"] == ML]["alerts_per_asset_month"].mean())
        false_ = {m: float(self.mm[self.mm["method"] == m]["false_per_asset_month"].mean()) for m in METHODS}
        ex = self.dq[self.dq["exception"]]
        n_cmp = int(ex["point_id"].str.startswith("CMP").sum())
        v = self.versions().iloc[0]
        drift_months = int(self.ml["secondary_trigger"].astype(bool).sum())
        year = first[:4]
        rows = [
            ("Model", f"{v['Settings'].split(' 200')[0]}, version {v['Version']}, champion all year", None),
            ("Period monitored", "July 2025 to June 2026", None),
            (
                "Catch share, year",
                f"{caught} of {conds} ({caught / conds:.2f}) against a {self.ref_catch:.2f} reference",
                "pass" if self.ref_catch - caught / conds <= self.tol else "refit",
            ),
            ("Primary rule trigger", f"fired {' and '.join(fired)} {year}", "refit"),
            (
                "Alert rate, year",
                f"{rate:.2f} per asset-month, {rate / rate_ref:.2f} of the {rate_ref:.3f} reference",
                "pass" if self.lo <= rate / rate_ref <= self.hi else "watch",
            ),
            (
                "False alerts per asset-month",
                f"{false_[ML]:.3f}, against Vendor Rules' {false_['Vendor Rules']:.3f}",
                "pass",
            ),
            (
                "Secondary rule trigger",
                f"fired in {'all ' if drift_months == len(MONTHS) else ''}{drift_months} months",
                "watch",
            ),
            ("Data-quality exceptions", f"{len(ex)} point-months, {n_cmp} on idle compressors", "watch"),
        ]
        meta = "".join(
            f"<div><span class='meta-label'>{html.escape(k)}</span>"
            f"<span class='meta-val'{f' style=color:{STATUS[c][0]}' if c else ''}>{html.escape(val)}</span></div>"
            for k, val, c in rows
        )
        reasons = self.t.html(S, "Catch share below the 0.54 limit")
        red = STATUS["refit"][0]
        return (
            f"<div class='status-block' style='border-color:{red}'>"
            f"<div class='status-header' style='background:{red}'><span class='status-icon'>{STATUS['refit'][1]}</span>"
            f"<span class='status-label'>REFIT RECOMMENDED</span>"
            f"<span class='status-when'>Triggered {MONTH_LABEL[first][:3]} {first[:4]}</span></div>"
            f"<div class='status-body'><div class='status-meta'>{meta}</div>"
            f"<div><div class='reasons-title'>Trigger Reasons</div><div class='trigger-list'>{reasons}</div></div>"
            "</div></div>"
        )

    def matrix(self) -> str:
        s = self.status
        rate_ref = float(self.ref.at[ML, "alerts_per_asset_month"])
        rows = [
            (
                "Primary",
                f"Catch share more than {self.tol:.2f} below the training reference ({self.ref_catch:.2f})",
                f"Refit after {self.need} consecutive months",
                "catch",
                lambda r: (
                    "no conditions"
                    if r["conditions"] == 0
                    else f"{r['catch_share']:.2f} ({r['caught']} of {r['conditions']})"
                ),
            ),
            (
                "Primary",
                f"Alerts per asset-month outside {self.lo} to {self.hi} times the training reference ({rate_ref:.3f})",
                f"Refit after {self.need} consecutive months",
                "rate",
                lambda r: f"{r['ratio']:.2f} of the reference",
            ),
            (
                "Secondary",
                f"More than {self.share:.0%} of point-features drifted (scaled Wasserstein distance above {self.thr})",
                "Investigate",
                "feature",
                lambda r: f"{r['feature_share']:.3f} of point-features",
            ),
            (
                "Secondary",
                f"More than {self.share:.0%} of point scores drifted (same distance)",
                "Investigate",
                "score",
                lambda r: f"{r['score_share']:.3f} of point scores",
            ),
        ]
        cols = "<col style='width:10%'><col style='width:27%'><col style='width:15%'>" + "".join(
            "<col style='width:4%'>" for _ in MONTHS
        )
        head = "<tr><th>Tier</th><th>Rule</th><th>Action</th>" + "".join(
            f"<th class='mc'>{SHORT[m]}</th>" for m in MONTHS
        )
        body = ""
        for tier, rule, action, key, tip in rows:
            tc = style.BRAND_BLUE if tier == "Primary" else style.DARK_GREY
            cells = ""
            for m in MONTHS:
                r = s.loc[m]
                c, mark, lab = STATUS[r[key]]
                cells += (
                    f"<td class='mc' style='color:{c}' title='{html.escape(SHORT[m])}: {html.escape(lab)}, "
                    f"{html.escape(tip(r))}'>{mark}</td>"
                )
            body += (
                f"<tr><td><span class='tier' style='color:{tc}'>{tier}</span></td><td>{html.escape(rule)}</td>"
                f"<td>{html.escape(action)}</td>{cells}</tr>"
            )
        legend = "".join(
            f"<span class='lg'>{self.sym(k)} {html.escape(desc)}</span>"
            for k, desc in (
                ("pass", "Rule passes"),
                ("watch", "Condition met for one month, or drift above the trigger"),
                ("refit", f"Condition met for {self.need} consecutive months: refit"),
                ("none", "Not evaluated (no confirmed conditions)"),
            )
        )
        return (
            f"<div class='matrix-wrap'><table class='matrix'><colgroup>{cols}</colgroup><thead>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div><div class='matrix-legend'>{legend}</div>"
        )

    # ---------------------------------------------------------------- section 2.1
    def _months_axis(self, ax) -> np.ndarray:
        x = np.arange(len(MONTHS))
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MONTHS], fontsize=8.5)
        ax.set_xlim(-0.6, len(MONTHS) - 0.4)
        return x

    def fig_catch(self) -> str:
        s = self.status
        f, ax = style.fig(h=3.4)
        x = self._months_axis(ax)
        for i, m in enumerate(MONTHS):
            r = s.loc[m]
            if r["conditions"] == 0:
                ax.plot(i, 0.02, "o", ms=7, mfc="white", mec=style.GREY, mew=1.4)
                ax.text(
                    i, 0.07, "no\nconditions", ha="center", va="bottom", fontsize=7, color=style.DARK_GREY
                )
                continue
            ax.bar(i, r["catch_share"], width=0.78, color=STATUS[r["catch"]][0])
            # the count inside the top of the bar; a bar ending just above the limit carries it below the line
            top = r["catch_share"] - 0.02
            if top > self.limit and top - LABEL_H < self.limit + 0.01:
                top = self.limit - 0.02
            ax.text(
                i,
                top,
                f"{r['caught']} of {r['conditions']}",
                ha="center",
                va="top",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=5,
            )
        vr = [self.catch("Vendor Rules", m)[2] for m in MONTHS]
        ax.plot(x, vr, "o", ms=4.5, color=style.GREY, zorder=4, label="Vendor Rules")
        ax.axhline(self.ref_catch, color=style.DARK_GREY, lw=1.1)
        ax.axhline(self.limit, color=style.RED, lw=1.1, ls="--")
        ax.text(
            len(MONTHS) - 0.35, self.ref_catch, f" Reference {self.ref_catch:.2f}", va="center", fontsize=7.5
        )
        ax.text(
            len(MONTHS) - 0.35,
            self.limit,
            f" Limit {self.limit:.2f}",
            va="center",
            fontsize=7.5,
            color=style.RED,
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Catch share", fontsize=8.5)
        hd = [Patch(fc=STATUS[k][0]) for k in ("pass", "watch", "refit")] + [
            Line2D([], [], marker="o", ls="", color=style.GREY)
        ]
        labs = [
            "ML Model, passes",
            "ML Model, below the limit one month",
            "ML Model, primary rule trigger",
            "Vendor Rules",
        ]
        ax.legend(
            hd, labs, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4
        )
        return self.img(f, "catch_share")

    def fig_rate(self) -> str:
        s = self.status
        f, ax = style.fig(h=3.2)
        x = self._months_axis(ax)
        ax.axhspan(self.lo, self.hi, color="#E6EEF4", lw=0)
        ax.axhline(1.0, color=style.DARK_GREY, lw=1)
        vr = self.rd[self.rd["method"] == "Vendor Rules"].set_index("month").reindex(list(MONTHS))
        ax.plot(x, vr["alerts_rate_ratio"], "-", color=style.GREY, lw=1.6, label="Vendor Rules")
        ax.plot(x, s["ratio"], "-", color=style.BRAND_BLUE, lw=1.6, label="ML Model")
        for i, m in enumerate(MONTHS):
            ax.plot(i, s.loc[m, "ratio"], "o", ms=6, color=STATUS[s.loc[m, "rate"]][0], zorder=4)
        ax.text(len(MONTHS) - 0.35, self.hi, f" {self.hi:g}x", va="center", fontsize=7.5)
        ax.text(len(MONTHS) - 0.35, self.lo, f" {self.lo:g}x", va="center", fontsize=7.5)
        ax.set_ylim(0, max(2.3, float(np.nanmax([s["ratio"].max(), vr["alerts_rate_ratio"].max()])) * 1.1))
        ax.set_ylabel("Alerts per asset-month, times reference", fontsize=8.5)
        hd = [
            Line2D([], [], color=style.BRAND_BLUE, lw=1.6, marker="o", mfc=style.GREEN, mec=style.GREEN),
            Line2D([], [], color=style.GREY, lw=1.6),
            Patch(fc="#E6EEF4"),
            Line2D([], [], marker="o", ls="", color=style.AMBER),
        ]
        labs = ["ML Model", "Vendor Rules", f"Band, {self.lo:g} to {self.hi:g}x", "ML Model outside the band"]
        ax.legend(
            hd, labs, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4
        )
        return self.img(f, "alert_rate_ratio")

    def fig_false(self) -> str:
        f, ax = style.fig(h=3.0)
        x = self._months_axis(ax)
        for m, c in ((ML, style.BRAND_BLUE), ("Vendor Rules", style.GREY)):
            g = (
                self.mm[self.mm["method"] == m]
                .set_index("month")
                .reindex(list(MONTHS))["false_per_asset_month"]
            )
            ax.plot(x, g, "-o", ms=4, lw=1.6, color=c, label=m)
            ax.axhline(g.mean(), color=c, lw=0.9, ls="--")
            ax.text(len(MONTHS) - 0.35, g.mean(), f" {g.mean():.3f}", va="center", fontsize=7.5, color=c)
        ax.set_ylim(0, None)
        ax.set_ylabel("False alerts per asset-month", fontsize=8.5)
        hd, lb = ax.get_legend_handles_labels()
        hd.append(Line2D([], [], color=style.DARK_GREY, lw=0.9, ls="--"))
        lb.append("Year rate")
        ax.legend(
            hd, lb, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3
        )
        return self.img(f, "false_alerts")

    # ---------------------------------------------------------------- section 2.2
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

    def fig_drift_share(self) -> str:
        s = self.status
        f, ax = style.fig(h=3.0)
        x = self._months_axis(ax)
        for col, key, ls, lab in (
            ("feature_share", "feature", "-", "Point-features drifted"),
            ("score_share", "score", "--", "Point scores drifted"),
        ):
            ax.plot(x, s[col], ls, color=style.DARK_GREY, lw=1.4, label=lab)
            for i, m in enumerate(MONTHS):
                ax.plot(i, s.loc[m, col], "o", ms=5.5, color=STATUS[s.loc[m, key]][0], zorder=4)
        ax.axhline(self.share, color=style.AMBER, lw=1.1, ls="--")
        ax.text(
            len(MONTHS) - 0.35,
            self.share,
            f" Trigger {self.share:.2f}",
            va="center",
            fontsize=7.5,
            color=style.AMBER,
        )
        ax.set_ylim(0, max(0.7, float(s["score_share"].max()) * 1.1))
        ax.set_ylabel("Share drifted", fontsize=8.5)
        hd, lb = ax.get_legend_handles_labels()
        hd += [
            Line2D([], [], marker="o", ls="", color=style.AMBER),
            Line2D([], [], marker="o", ls="", color=style.GREEN),
        ]
        lb += ["Above the trigger", "Below the trigger"]
        ax.legend(
            hd, lb, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4
        )
        return self.img(f, "drift_share")

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

    def score_heatmap(self) -> str:
        sc = (
            self.drift[self.drift["feature"] == "nbm_score"]
            .set_index(["point_id", "month"])["wasserstein_scaled"]
            .unstack()
            .reindex(index=self.points, columns=list(MONTHS))
        )
        v = sc.to_numpy(dtype=float)
        lo, hi = 10 ** np.floor(np.log10(np.nanmin(v))), 10 ** np.ceil(np.log10(np.nanmax(v)))
        f, ax = style.fig(h=8.6, w=8.6)
        ax.grid(False)
        norm = LogNorm(vmin=lo, vmax=hi)
        im = ax.imshow(np.ma.masked_invalid(v), aspect="auto", cmap="Oranges", norm=norm)
        ax.set_yticks(range(len(sc)))
        ax.set_yticklabels(sc.index, fontsize=7)
        ax.set_xticks(range(len(MONTHS)))
        ax.set_xticklabels([SHORT[mo] for mo in MONTHS], fontsize=8)

        def short(x: float) -> str:
            if x >= 1000:
                return f"{x / 1000:.0f}k"
            if x >= 10:
                return f"{x:.0f}"
            return f"{x:.1f}" if x >= 0.1 else f"{x:.2f}"

        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                x = v[i, j]
                if not np.isfinite(x):
                    continue
                dark = norm(x) > 0.6
                ax.text(
                    j, i, short(x), ha="center", va="center", fontsize=6, color="white" if dark else "#222222"
                )
                if x > self.thr:
                    ax.add_patch(
                        Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=style.TEXT, lw=0.8)
                    )
        cb = f.colorbar(im, ax=ax, shrink=0.5)
        cb.set_label(f"Scaled distance, log scale (outlined above {self.thr})", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        return self.img(f, "score_drift")

    # ---------------------------------------------------------------- section 2.3
    def fig_dq(self) -> str:
        ex = self.dq[self.dq["exception"]]
        cmp_ = (
            ex[ex["point_id"].str.startswith("CMP")]
            .groupby("month")
            .size()
            .reindex(list(MONTHS), fill_value=0)
        )
        hmc = (
            ex[~ex["point_id"].str.startswith("CMP")]
            .groupby("month")
            .size()
            .reindex(list(MONTHS), fill_value=0)
        )
        f, ax = style.fig(h=3.0)
        x = self._months_axis(ax)
        ax.bar(x, hmc, width=0.62, color=HMC_SHADE, label="HMC points")
        ax.bar(x, cmp_, width=0.62, bottom=hmc, color=CMP_SHADE, label="Compressor points")
        for i, (a, b) in enumerate(zip(hmc, cmp_, strict=True)):
            if a:
                ax.text(i, a / 2, str(a), ha="center", va="center", fontsize=7.5, color="white")
            if b:
                ax.text(i, a + b / 2, str(b), ha="center", va="center", fontsize=7.5, color=style.TEXT)
            ax.text(i, a + b + 0.4, str(a + b), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_ylim(0, float((hmc + cmp_).max()) * 1.18)
        ax.set_ylabel("Point-months with an exception", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        return self.img(f, "dq_exceptions")

    def quality(self) -> str:
        nm = self.dq.pivot_table(index="point_id", columns="month", values="no_measurement_share").reindex(
            index=self.points, columns=list(MONTHS)
        )
        return self._heat(100 * nm, "data_quality", "No-measurement share, %", 100.0, ".0f")

    # ---------------------------------------------------------------- section 2.4
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

    def gates(self) -> str:
        v = self.versions().iloc[0]
        badges = ""
        for g in ("Selection", "Parity", "Points", "Numerics", "False episodes"):
            ok = str(v[g]).lower() == "pass"
            k = "pass" if ok else "refit"
            badges += f"<span class='gate' style='border-color:{STATUS[k][0]}'>{self.sym(k)} {html.escape(g)}</span>"
        model = v["Settings"].split(" 200")[0]
        return (
            f"<div class='gates'>{badges}<span class='gate-model'>Version {v['Version']}, "
            f"{html.escape(model)}, {html.escape(str(v['Status']))}</span></div>"
        )

    def fig_resets(self) -> str:
        r = _read("recentering_events")
        for c in ("segment_start", "level_ready"):
            r[c] = pd.to_datetime(r[c], utc=True)
        r["asset"] = r["point_id"].map(asset_of)
        assets = sorted(r["asset"].unique(), key=lambda a: (a.startswith("CMP"), a))
        f, ax = style.fig(h=0.27 * len(assets) + 1.2, grid="x")
        t0 = pd.Timestamp("2025-07-01", tz=TZ).tz_convert("UTC")
        t1 = pd.Timestamp("2026-07-01", tz=TZ).tz_convert("UTC")

        def d(ts: pd.Timestamp) -> float:
            return (ts - t0).total_seconds() / 86400

        y = {a: i for i, a in enumerate(assets)}
        for x in r.itertuples():
            c = style.AMBER if bool(x.plausibility_flag) else style.BRAND_BLUE
            yy = y[x.asset]
            if pd.notna(x.level_ready):
                ax.plot(
                    [d(x.segment_start), d(x.level_ready)], [yy, yy], color=c, lw=2.2, solid_capstyle="butt"
                )
                ax.plot(d(x.segment_start), yy, "o", ms=5, color=c, zorder=3)
            else:
                ax.plot(d(x.segment_start), yy, "o", ms=5.5, mfc="white", mec=c, mew=1.4, zorder=3)
        ax.set_yticks(range(len(assets)))
        ax.set_yticklabels(assets, fontsize=7.5)
        ax.set_ylim(len(assets) - 0.4, -0.6)
        months = pd.date_range("2025-07-01", "2026-06-01", freq="MS", tz=TZ)
        ax.set_xticks([d(m.tz_convert("UTC")) for m in months])
        ax.set_xticklabels([m.strftime("%b") for m in months], fontsize=8.5)
        ax.set_xlim(0, d(t1) + 5)
        hd = [
            Line2D([], [], color=style.BRAND_BLUE, lw=2.2, marker="o"),
            Line2D([], [], color=style.AMBER, lw=2.2, marker="o"),
            Line2D([], [], ls="", marker="o", mfc="white", mec=style.BRAND_BLUE, mew=1.4),
        ]
        labs = [
            "Re-centering, start to level set",
            "Flagged by the plausibility check",
            "No level by the record end",
        ]
        ax.legend(
            hd, labs, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3
        )
        return self.img(f, "recentering")

    # ---------------------------------------------------------------- section 3
    def log(self) -> str:
        s = self.status
        rows = ""
        run = 0
        bad = [bool(s.loc[m, "catch_bad"] or s.loc[m, "rate_bad"]) for m in MONTHS]
        for i, m in enumerate(MONTHS):
            r = s.loc[m]
            v = self.verdict(m)
            run = run + 1 if bad[i] else 0
            parts = []
            if r["catch_bad"]:
                parts.append(f"catch share {r['catch_share']:.2f} ({r['caught']} of {r['conditions']})")
            if r["rate_bad"]:
                parts.append(f"alert rate {r['ratio']:.2f} of the reference")
            if not parts:
                reason = "No primary condition" + ("; no conditions to score" if r["conditions"] == 0 else "")
            else:
                lead = " and ".join(parts)
                lead = lead[0].upper() + lead[1:]
                if v == "refit":
                    nth = {2: "second", 3: "third", 4: "fourth"}.get(run, f"{run}th")
                    reason = f"{lead}, {nth} consecutive month: refit triggered"
                elif i + 1 < len(MONTHS) and bad[i + 1]:
                    reason = f"{lead}, first month outside the limits"
                else:
                    reason = f"{lead}, for one month only"
            label = STATUS[v][2]
            rows += (
                f"<tr><td>{MONTH_LABEL[m]}</td><td>{self.sym(v, label)}</td><td>{html.escape(reason)}</td>"
                f"<td>v{self.versions().iloc[0]['Version']}</td></tr>"
            )
        return (
            "<div class='matrix-wrap'><table class='data log'><thead><tr><th>Month</th><th>Verdict</th>"
            f"<th>Reason</th><th>Champion</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )

    # ---------------------------------------------------------------- appendix
    def rule(self) -> str:
        win = str(self.ref["reference_window"].iat[0])
        ml_win = str(self.ref.at[ML, "rate_window"])
        rows = [
            (
                "Primary rule trigger",
                f"Catch share more than {self.tol:.2f} below the reference catch share on the training folds (July 2024 to June 2025), or alerts per asset-month outside {self.lo} to {self.hi} times the training reference rate ({win}; for the ML Model, the fold-2 model's alerts from {ml_win}), for {self.need} consecutive months",
            ),
            (
                "Secondary rule trigger",
                f"Share of point-features, or of point scores, with scaled Wasserstein distance above {self.thr} greater than {self.share:.2f}",
            ),
            (
                "Verdict",
                "Retrain when the primary rule trigger fires; the secondary rule trigger is reported beside it",
            ),
            ("Action", str(self.rd.loc[self.rd["retrain"].astype(bool), "action"].iat[0]).capitalize()),
        ]
        return style.table(pd.DataFrame(rows, columns=["Part", "Definition"]))

    def references(self) -> str:
        win = str(self.ref["reference_window"].iat[0])
        defs = pd.DataFrame(
            [
                (
                    "Training reference",
                    f"{win}, the last six months of training, for points not re-centered after its start; the ML Model score from the fold-2 model, scored from {self.ref.at[ML, 'rate_window'].split(' to ')[0]}",
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
        return style.table(defs)

    # ---------------------------------------------------------------- page
    def build(self) -> str:
        t, x = self.t, self.exhibit
        self.h(2, "1. Status and Retraining Decision", "s1")
        self.text("The ML Model was monitored monthly")
        self.parts.append(self.status_block())
        self.text("The retraining rule is evaluated every month")
        self.parts.append(self.matrix())
        self.h(2, "2. Monitoring Results", "s2")
        self.h(3, "2.1 Performance", "s21", True)
        self.text("Catch share, the first of the two primary rule triggers")
        x("Exhibit: Catch share by month", self.fig_catch())
        self.text("Alert rate, the second primary rule trigger")
        x("Exhibit: Alert rate against reference by month", self.fig_rate())
        x("Exhibit: False alerts per asset-month by month", self.fig_false())
        self.h(3, "2.2 Drift", "s22", True)
        self.text("The secondary rule trigger flags drift")
        x("Exhibit: Share of points drifted by month", self.fig_drift_share())
        x("Exhibit: Feature drift by point and month", self.heatmap())
        x("Exhibit: Score drift by point and month", self.score_heatmap())
        self.h(3, "2.3 Data Quality", "s23", True)
        self.text("A data-quality exception is a point-month")
        x("Exhibit: Data-quality exceptions by month", self.fig_dq())
        x("Exhibit: Data quality by point and month", self.quality())
        self.h(3, "2.4 Champion and Re-centering", "s24", True)
        self.text("Version 13 passed all five")
        self.parts.append(self.gates())
        x("Exhibit: Post-repair re-centering by asset", self.fig_resets())
        self.h(2, "3. Monitoring Log", "s3")
        self.parts.append(self.log())
        self.text("The refit the rule calls for")
        self.h(2, "Appendix", "appendix")
        self.parts.append("<h3 id='aA'>Retraining rule</h3>")
        self.toc.append(("aA", "Retraining rule", "sub"))
        self.text("Acted-on alerts remove")
        self.parts.append(self.rule())
        self.parts.append("<h3 id='aB'>Drift references</h3>")
        self.toc.append(("aB", "Drift references", "sub"))
        self.parts.append(self.references())
        body = "".join(self.parts)
        page = style.report_shell(t.field(S, "Title"), "", body, self.toc)
        style.write("reports/monitoring_report.html", page.replace("</style>", EXTRA_CSS + "</style>", 1))
        return "docs/reports/monitoring_report.html"


EXTRA_CSS = f"""
.status-block {{ border: 2px solid; border-radius: 6px; overflow: hidden; margin: 18px 0; }}
.status-header {{ display: flex; align-items: center; gap: 12px; padding: 12px 18px; color: #fff; }}
.status-icon {{ font-size: 20px; font-weight: 700; }}
.status-label {{ font-size: 17px; font-weight: 700; letter-spacing: .3px; }}
.status-when {{ margin-left: auto; font-size: 13px; opacity: .9; }}
.status-body {{ padding: 16px 18px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
.status-meta {{ display: flex; flex-direction: column; gap: 7px; }}
.status-meta > div {{ display: flex; gap: 10px; font-size: 14px; }}
.meta-label {{ color: {style.DARK_GREY}; font-weight: 700; font-size: 11.5px; text-transform: uppercase;
  letter-spacing: .5px; width: 150px; flex-shrink: 0; padding-top: 2px; }}
.meta-val {{ font-weight: 600; }}
.reasons-title {{ font-size: 11.5px; font-weight: 700; color: {style.DARK_GREY}; text-transform: uppercase;
  letter-spacing: .5px; margin-bottom: 4px; }}
.trigger-list ul {{ margin: 4px 0 0 18px; padding: 0; }}
.trigger-list li {{ font-size: 14px; margin-bottom: 5px; line-height: 1.5; }}
.matrix-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 10px 0 4px; }}
table.matrix {{ table-layout: fixed; width: 100%; min-width: 720px; border-collapse: collapse; font-size: 13px; }}
table.matrix th {{ background: var(--brand); color: #fff; text-align: left; padding: 6px 7px; font-weight: 600; }}
table.matrix td {{ border-bottom: 1px solid #EEE; padding: 6px 7px; vertical-align: middle; }}
table.matrix th.mc, table.matrix td.mc {{ text-align: center; padding: 6px 0; }}
table.matrix td.mc {{ font-size: 16px; font-weight: 700; }}
.tier {{ font-weight: 700; }}
.matrix-legend {{ font-size: 12.5px; color: {style.DARK_GREY}; margin: 2px 0 16px; display: flex; flex-wrap: wrap;
  gap: 4px 18px; }}
.gates {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 10px 0 6px; }}
.gate {{ border: 1px solid; border-radius: 4px; padding: 3px 9px; font-size: 13px; background: #fff; }}
.gate-model {{ font-size: 13.5px; color: var(--text); margin-left: 6px; }}
table.log {{ font-size: 14px; min-width: 520px; }}
table.log td:nth-child(2) {{ white-space: nowrap; }}
@media (max-width: 760px) {{ .status-body {{ grid-template-columns: 1fr; }} .meta-label {{ width: 120px; }} }}
"""


def build(d: Data, t: PageText) -> str:
    return MonitoringReport(d, t).build()

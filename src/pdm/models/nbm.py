"""Per-point normal-behavior models.

Base model: all measured hours of the point from install to the training cutoff, less the exclusions
(the days before confirmed conditions, open cases and repair periods, re-baseline windows after repairs
and remounts, flagged baselines), features re-centered per segment; used only when its training hours
cover both summer and winter, and it scores the install segment. Re-baselining: after each repair or
remount the point's model is refit on the 21-day post-repair window when that window is usable and
passes the sanity check, with thresholds from that window's scores; until the window completes, and
while the latest window is flagged, the point is scored by the rule floor only.

Candidates: PCA with T-squared and SPE, Mahalanobis distance, isolation forest, and an autoencoder on
the one-third-octave summary. A separate model per regime with at least the configured training
hours, otherwise one model with the regime as a feature. Thresholds are percentiles of the training
scores; the configured percentile is the operating point and others form the operating curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

VECTOR = [
    "accel_rms_g",
    "vel_rms_mm_s",
    "hf_rms_g",
    "crest_factor",
    "kurtosis",
    "ord_01_g2",
    "ord_02_g2",
    "env_fault_g2",
]
VECTOR_SOURCE = [c for c in VECTOR if c != "env_fault_g2"] + [
    "env_bpfo_1_g2",
    "env_bpfi_1_g2",
    "env_bsf_1_g2",
]
LOG_COLUMNS = {"accel_rms_g", "vel_rms_mm_s", "hf_rms_g", "ord_01_g2", "ord_02_g2", "env_fault_g2"}
REGIMES = ["light", "medium", "heavy"]
CANDIDATES = ("pca_t2_spe", "mahalanobis", "isolation_forest", "autoencoder_third_octave")


class LeakedWindowError(ValueError):
    """Training rows at or after the training cutoff, or inside an excluded interval."""


def env_fault(df: pd.DataFrame) -> pd.Series:
    return df[["env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2"]].fillna(0.0).sum(axis=1)


def toct_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("toct_"))


def features(df: pd.DataFrame, candidate: str) -> tuple[np.ndarray, list[str]]:
    """Log-scaled feature matrix for a candidate (regime columns are added at fit time)."""
    if candidate == "autoencoder_third_octave":
        cols = toct_columns(df)
        return np.log10(df[cols].to_numpy(dtype=float) + 1e-12), cols
    d = df.assign(env_fault_g2=env_fault(df))
    x = np.column_stack(
        [
            np.log10(d[c].to_numpy(dtype=float) + 1e-12) if c in LOG_COLUMNS else d[c].to_numpy(dtype=float)
            for c in VECTOR
        ]
    )
    return x, list(VECTOR)


def segment_ids(hours: pd.Series, segment_starts: list[pd.Timestamp]) -> np.ndarray:
    starts = np.sort(np.asarray(pd.to_datetime(pd.Series(segment_starts), utc=True), dtype="datetime64[ns]"))
    t = np.asarray(pd.to_datetime(hours, utc=True), dtype="datetime64[ns]")
    return np.searchsorted(starts, t, side="right") - 1


def centered(
    df: pd.DataFrame, x: np.ndarray, segment_starts: list[pd.Timestamp], rebaseline: pd.Timedelta
) -> tuple[np.ndarray, np.ndarray]:
    """Features less the median of the segment's first window, per regime where the window has that
    regime, otherwise pooled. Returns the centered matrix and a mask of rows inside a first window."""
    seg = segment_ids(df["hour_utc"], segment_starts)
    starts = pd.to_datetime(pd.Series(segment_starts), utc=True).sort_values().reset_index(drop=True)
    t = np.asarray(pd.to_datetime(df["hour_utc"], utc=True).dt.tz_localize(None), dtype="datetime64[ns]")
    out = np.full_like(x, np.nan)
    in_first = np.zeros(len(df), dtype=bool)
    reg = df["regime"].to_numpy()
    for s in np.unique(seg):
        if s < 0:
            continue
        rows = seg == s
        a = starts.iloc[s].tz_localize(None)
        first = rows & (t >= a.to_datetime64()) & (t < (a + rebaseline).to_datetime64())
        in_first |= first
        pooled = np.nanmedian(x[first], axis=0) if first.any() else np.nanmedian(x[rows], axis=0)
        for r in np.unique(reg[rows]):
            rr = rows & (reg == r)
            fr = first & (reg == r)
            c = np.nanmedian(x[fr], axis=0) if fr.sum() >= 5 else pooled
            out[rr] = x[rr] - c
    return out, in_first


@dataclass
class Scaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        c = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        s = (q75 - q25) / 1.349
        s = np.where(s > 1e-9, s, np.nanstd(x, axis=0))
        return cls(c, np.where(s > 1e-9, s, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.center) / self.scale)


@dataclass
class FittedModel:
    candidate: str
    columns: list[str]
    scaler: Scaler
    model: Any
    thresholds: dict[float, float]
    regime: str | None
    regime_feature: bool
    n_train: int
    extra: dict[str, Any] = field(default_factory=dict)

    def design(self, xc: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        if self.regime_feature:
            xc = np.column_stack([xc, *[(regimes == r).astype(float) for r in REGIMES]])
        return self.scaler.transform(xc)

    def raw_scores(self, xc: np.ndarray, regimes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.design(xc, regimes)
        if self.candidate == "pca_t2_spe":
            p = self.model
            t = p.transform(z)
            t2 = np.sum(t**2 / p.explained_variance_, axis=1)
            spe = np.sum((z - p.inverse_transform(t)) ** 2, axis=1)
            return t2 / self.extra["t2_med"] + spe / self.extra["spe_med"], z**2
        if self.candidate == "mahalanobis":
            return np.einsum("ij,jk,ik->i", z, self.model, z), z**2
        if self.candidate == "isolation_forest":
            return -self.model.score_samples(z), z**2
        rec = self.model.predict(z)
        return np.mean((z - rec) ** 2, axis=1), (z - rec) ** 2

    def top_contributors(self, contrib: np.ndarray, k: int) -> list[str]:
        names = self.columns + ([f"regime_{r}" for r in REGIMES] if self.regime_feature else [])
        idx = np.argsort(-contrib, axis=1)[:, :k]
        return [",".join(names[j] for j in row) for row in idx]


def fit_candidate(
    xc: np.ndarray,
    regimes: np.ndarray,
    cols: list[str],
    candidate: str,
    cfg: dict[str, Any],
    regime: str | None,
    regime_feature: bool,
) -> FittedModel:
    m = cfg["models"]
    seed = m["seed"]
    fm = FittedModel(
        candidate, cols, Scaler(np.zeros(1), np.ones(1)), None, {}, regime, regime_feature, len(xc)
    )
    raw = np.column_stack([xc, *[(regimes == r).astype(float) for r in REGIMES]]) if regime_feature else xc
    fm.scaler = Scaler.fit(raw)
    z = fm.scaler.transform(raw)
    if candidate == "pca_t2_spe":
        model = PCA(n_components=m["nbm"]["pca_variance_kept"], svd_solver="full", random_state=seed).fit(z)
        t = model.transform(z)
        t2 = np.sum(t**2 / model.explained_variance_, axis=1)
        spe = np.sum((z - model.inverse_transform(t)) ** 2, axis=1)
        fm.extra = {"t2_med": float(np.median(t2)) or 1.0, "spe_med": float(np.median(spe)) or 1.0}
    elif candidate == "mahalanobis":
        model = LedoitWolf().fit(z).precision_
    elif candidate == "isolation_forest":
        model = IsolationForest(
            n_estimators=m["nbm"]["isolation_forest"]["n_estimators"], random_state=seed
        ).fit(z)
    else:
        ae = m["nbm"]["autoencoder"]
        model = MLPRegressor(
            hidden_layer_sizes=tuple(ae["hidden"]), max_iter=ae["epochs"], random_state=seed
        ).fit(z, z)
    fm.model = model
    s, _ = fm.raw_scores(xc, regimes)
    pcts = sorted(set(m["nbm"]["sweep_percentiles"]) | {m["nbm"]["threshold_percentile"]})
    fm.thresholds = {float(p): float(np.percentile(s, p)) for p in pcts}
    return fm


def check_training_rows(
    hours: pd.Series, cutoff: pd.Timestamp, excluded: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> None:
    """Refuse rows at or after the training cutoff or inside an excluded interval."""
    t = pd.to_datetime(hours, utc=True)
    if (t >= cutoff).any():
        raise LeakedWindowError("training rows at or after the training cutoff")
    for a, b in excluded:
        if ((t >= a) & (t < b)).any():
            raise LeakedWindowError("training rows inside an excluded interval")


def seasonal_coverage(hours: pd.Series, cfg: dict[str, Any]) -> dict[str, bool]:
    m = pd.to_datetime(hours, utc=True).dt.tz_convert("America/Chicago").dt.month
    cov = cfg["models"]["nbm"]["coverage_months"]
    out = {}
    for season, (a, b) in cov.items():
        months = list(range(a, b + 1)) if a <= b else list(range(a, 13)) + list(range(1, b + 1))
        out[season] = bool(m.isin(months).any())
    return out


def training_mask(
    hours: pd.Series, cutoff: pd.Timestamp, excluded: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> np.ndarray:
    t = pd.to_datetime(hours, utc=True)
    keep = (t < cutoff).to_numpy()
    for a, b in excluded:
        keep &= ~((t >= a) & (t < b)).to_numpy()
    return keep


def fit_point(
    rows: pd.DataFrame,
    candidate: str,
    cfg: dict[str, Any],
    cutoff: pd.Timestamp,
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]],
    segment_starts: list[pd.Timestamp],
) -> tuple[dict[str, FittedModel], dict[str, Any]]:
    """Models for one point keyed by regime ('all' when the regime is a feature), and training info."""
    nb = cfg["models"]["nbm"]
    clean = (
        rows[~rows["implausible"].fillna(False).astype(bool)]
        .dropna(subset=["vel_rms_mm_s"])
        .reset_index(drop=True)
    )
    x, cols = features(clean, candidate)
    xc, _ = centered(clean, x, segment_starts, pd.Timedelta(days=nb["rebaseline_days"]))
    keep = training_mask(clean["hour_utc"], cutoff, excluded) & np.isfinite(xc).all(axis=1)
    train_hours = clean.loc[keep, "hour_utc"]
    check_training_rows(train_hours, cutoff, excluded)
    cov = seasonal_coverage(train_hours, cfg)
    info = {"n_train": int(keep.sum()), **{f"covers_{k}": v for k, v in cov.items()}}
    if not all(cov.values()) or keep.sum() < nb["min_train_hours"]:
        return {}, info
    reg = clean["regime"].to_numpy()
    return fit_models(xc[keep], reg[keep], cols, candidate, cfg), info


def fit_models(
    x: np.ndarray, reg: np.ndarray, cols: list[str], candidate: str, cfg: dict[str, Any]
) -> dict[str, FittedModel]:
    """One model per regime when every regime has the configured hours, otherwise one model with the
    regime as a feature."""
    counts = pd.Series(reg).value_counts()
    big = [r for r, n in counts.items() if n >= cfg["models"]["nbm"]["min_regime_healthy_hours"]]
    if len(big) == len(counts):
        return {r: fit_candidate(x[reg == r], reg[reg == r], cols, candidate, cfg, r, False) for r in big}
    return {"all": fit_candidate(x, reg, cols, candidate, cfg, None, len(counts) > 1)}


def score_point(
    rows: pd.DataFrame,
    models: dict[str, FittedModel],
    cfg: dict[str, Any],
    segment_starts: list[pd.Timestamp],
    top_k: int,
) -> pd.DataFrame:
    """Scores for every measured hour outside re-baseline windows (install window included)."""
    nb = cfg["models"]["nbm"]
    clean = (
        rows[~rows["implausible"].fillna(False).astype(bool)]
        .dropna(subset=["vel_rms_mm_s"])
        .reset_index(drop=True)
    )
    if clean.empty or not models:
        return pd.DataFrame()
    cand = next(iter(models.values())).candidate
    x, _ = features(clean, cand)
    xc, in_first = centered(clean, x, segment_starts, pd.Timedelta(days=nb["rebaseline_days"]))
    seg = segment_ids(clean["hour_utc"], segment_starts)
    scoreable = ~in_first | (seg == 0)
    reg = clean["regime"].to_numpy()
    out = []
    for key, fm in models.items():
        sel = scoreable & np.isfinite(xc).all(axis=1) & ((reg == key) if key != "all" else True)
        if not sel.any():
            continue
        s, contrib = fm.raw_scores(xc[sel], reg[sel])
        d = pd.DataFrame(
            {
                "point_id": clean.loc[sel, "point_id"].to_numpy(),
                "hour_utc": clean.loc[sel, "hour_utc"].to_numpy(),
                "regime": reg[sel],
                "model_regime": key,
                "score": s,
                "top_contributors": fm.top_contributors(contrib, top_k),
            }
        )
        for p, thr in fm.thresholds.items():
            d[f"thr_{p:g}"] = thr
        out.append(d)
    return (
        pd.concat(out, ignore_index=True).sort_values("hour_utc").reset_index(drop=True)
        if out
        else pd.DataFrame()
    )


def _clean(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows[~rows["implausible"].fillna(False).astype(bool)]
        .dropna(subset=["vel_rms_mm_s"])
        .reset_index(drop=True)
    )


def _score(
    clean: pd.DataFrame,
    x: np.ndarray,
    sel: np.ndarray,
    models: dict[str, FittedModel],
    top_k: int,
    label: str,
) -> list[pd.DataFrame]:
    reg = clean["regime"].to_numpy()
    out = []
    for key, fm in models.items():
        m = sel & np.isfinite(x).all(axis=1) & ((reg == key) if key != "all" else True)
        if not m.any():
            continue
        s, contrib = fm.raw_scores(x[m], reg[m])
        d = pd.DataFrame(
            {
                "point_id": clean.loc[m, "point_id"].to_numpy(),
                "hour_utc": clean.loc[m, "hour_utc"].to_numpy(),
                "regime": reg[m],
                "model_regime": key,
                "model_segment": label,
                "score": s,
                "top_contributors": fm.top_contributors(contrib, top_k),
            }
        )
        for p, thr in fm.thresholds.items():
            d[f"thr_{p:g}"] = thr
        out.append(d)
    return out


def segment_plan(
    windows: pd.DataFrame, flagged_ids: set[str], end: pd.Timestamp, rebaseline: pd.Timedelta
) -> pd.DataFrame:
    """Per segment of a point (install, then each repair or remount): whether its re-baseline window is
    usable for a refit. A window is usable when it completed before end, is not excluded for a nearby
    failure and is not flagged by the sanity check."""
    w = windows.sort_values("start").reset_index(drop=True)
    w["window_end"] = w["start"] + rebaseline
    w["complete"] = w["window_end"] <= end
    w["flagged"] = w["window_id"].isin(flagged_ids)
    w["refit"] = w["complete"] & w["usable"].astype(bool) & ~w["flagged"]
    return w


def _recent(t: np.ndarray, events: np.ndarray, gap: np.timedelta64) -> np.ndarray:
    """True where an event occurred in (t - gap, t]."""
    if events.size == 0:
        return np.zeros(t.size, dtype=bool)
    ev = np.sort(events)
    i = np.searchsorted(ev, t, side="right")
    prev = np.where(i > 0, ev[np.clip(i - 1, 0, ev.size - 1)], np.datetime64("NaT"))
    return (i > 0) & (t - prev < gap)


def _naive(ts: Any) -> np.ndarray:
    s = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_localize(None)
    return np.asarray(s, dtype="datetime64[ns]")


def month_starts(a: pd.Timestamp, b: pd.Timestamp) -> list[pd.Timestamp]:
    """Local calendar month starts strictly after a and before b, in UTC."""
    la = a.tz_convert("America/Chicago").tz_localize(None).normalize()
    lb = b.tz_convert("America/Chicago").tz_localize(None)
    ms = pd.date_range(la + pd.offsets.MonthBegin(1), lb, freq="MS")
    return [
        m.tz_localize("America/Chicago").tz_convert("UTC")
        for m in ms
        if a < m.tz_localize("America/Chicago").tz_convert("UTC") < b
    ]


def fit_score_segments(
    rows: pd.DataFrame,
    candidate: str,
    cfg: dict[str, Any],
    cutoff: pd.Timestamp,
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]],
    plan: pd.DataFrame,
    end: pd.Timestamp,
    top_k: int,
    open_work: list[tuple[pd.Timestamp, pd.Timestamp]],
    rule_alerts: list[pd.Timestamp],
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    """Scores before end with one model per segment (install, then each repair or remount).

    A segment that starts before the cutoff is fit on its eligible hours before the cutoff (the base
    rule, with its exclusions); a later segment starts from its 21-day re-baseline window. A segment
    that lacks seasonal coverage at the cutoff, or starts after it, is updated monthly on hours that
    were not inside an open work order, a rule-floor alert episode or its own alert episode at the
    time, until it has coverage. A segment whose re-baseline window is flagged, incomplete or not
    usable is left to the rule floor. Returns scores, per-segment information, training rows and each
    fitted segment's final models.
    """
    nb = cfg["models"]["nbm"]
    gap = np.timedelta64(int(cfg["evaluation"]["catch"]["episode_gap_days"] * 86400), "s")
    op = nb["threshold_percentile"]
    clean = _clean(rows)
    clean = clean[pd.to_datetime(clean["hour_utc"], utc=True) < end].reset_index(drop=True)
    if clean.empty:
        return pd.DataFrame(), [], pd.DataFrame(), []
    t = _naive(clean["hour_utc"])
    seg = segment_ids(clean["hour_utc"], list(plan["start"]))
    reg = clean["regime"].to_numpy()
    x, cols = features(clean, candidate)
    finite = np.isfinite(x).all(axis=1)
    eligible = training_mask(clean["hour_utc"], cutoff, excluded) & finite
    blocked_work = np.zeros(t.size, dtype=bool)
    for wa, wb in open_work:
        blocked_work |= (t >= _naive([wa])[0]) & (t < _naive([wb])[0])
    rule_recent = _recent(
        t, _naive(rule_alerts) if len(rule_alerts) else np.array([], dtype="datetime64[ns]"), gap
    )
    cut = _naive([cutoff])[0]
    out: list[pd.DataFrame] = []
    infos: list[dict[str, Any]] = []
    train_rows: list[pd.DataFrame] = []
    seg_models: list[dict[str, Any]] = []
    for k, w in plan.iterrows():
        rows_k = seg == k
        seg_end = plan["start"].iat[k + 1] if k + 1 < len(plan) else end
        seg_end = min(seg_end, end)
        info = {
            "point_id": w["point_id"],
            "segment": int(k),
            "segment_start": w["start"],
            "reason": w["reason"],
        }
        if k > 0 and not w["refit"]:
            why = (
                "flagged"
                if w["flagged"]
                else ("window incomplete" if not w["complete"] else "window not usable")
            )
            infos.append({**info, "model": f"rule floor only ({why})", "n_train": 0})
            continue
        ws, we = _naive([w["start"]])[0], _naive([w["window_end"]])[0]
        win = rows_k & finite & (t >= ws) & (t < we)
        if w["start"] < cutoff:
            train = rows_k & eligible & (t < min(cut, _naive([seg_end])[0]))
            if train.sum() < nb["min_train_hours"] and k > 0:
                train = win
            kind = "segment fit"
        else:
            train = win
            kind = "window refit"
        if train.sum() < nb["min_train_hours"]:
            infos.append({**info, "model": "rule floor only (too few hours)", "n_train": int(train.sum())})
            continue
        covered = all(seasonal_coverage(clean.loc[train, "hour_utc"], cfg).values())
        if w["start"] < cutoff and not covered and seg_end <= cutoff:
            infos.append(
                {**info, "model": "rule floor only (no seasonal coverage)", "n_train": int(train.sum())}
            )
            continue
        n0 = int(train.sum())
        models = fit_models(x[train], reg[train], cols, candidate, cfg)
        score_from = we
        updates = month_starts(max(cutoff, w["window_end"]), seg_end) if not covered else []
        bounds = [score_from, *[_naive([m])[0] for m in updates], _naive([seg_end])[0]]
        seg_scores: list[pd.DataFrame] = []
        n_updates = 0
        last = max(cut, we)
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            part = _score(clean, x, rows_k & (t >= a) & (t < b), models, top_k, f"segment {k}")
            seg_scores += part
            if i == len(bounds) - 2 or covered:
                continue
            done = pd.concat(seg_scores, ignore_index=True) if seg_scores else pd.DataFrame()
            own = exceedance_hours(done, op, nb["persistence_hours"]) if len(done) else pd.DataFrame()
            own_recent = _recent(
                t, _naive(own["ts"]) if len(own) else np.array([], dtype="datetime64[ns]"), gap
            )
            add = rows_k & finite & (t >= last) & (t < b) & ~blocked_work & ~rule_recent & ~own_recent
            train = train | add
            last = b
            models = fit_models(x[train], reg[train], cols, candidate, cfg)
            n_updates += 1
            covered = all(seasonal_coverage(clean.loc[train, "hour_utc"], cfg).values())
        out += seg_scores
        infos.append(
            {
                **info,
                "model": kind,
                "n_train": n0,
                "n_train_final": int(train.sum()),
                "monthly_updates": n_updates,
                "seasonal_coverage": covered,
                "models": ",".join(models),
            }
        )
        train_rows.append(clean[train].assign(segment=int(k), segment_start=w["start"]))
        seg_models.append({"segment": int(k), "start": w["start"], "reason": w["reason"], "models": models})
    scores = (
        pd.concat(out, ignore_index=True).sort_values("hour_utc").reset_index(drop=True)
        if out
        else pd.DataFrame()
    )
    tr = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    return scores, infos, tr, seg_models


def exceedance_hours(scores: pd.DataFrame, percentile: float, persistence: int) -> pd.DataFrame:
    """Hours at which the score has exceeded the percentile threshold on persistence consecutive
    measured hours (the persisted exceedances that alert episodes are built from)."""
    col = f"thr_{percentile:g}"
    rows = []
    for pid, g in scores.sort_values("hour_utc").groupby("point_id"):
        over = (g["score"] > g[col]).to_numpy()
        run = np.zeros(len(over), dtype=int)
        c = 0
        for i, o in enumerate(over):
            c = c + 1 if o else 0
            run[i] = c
        hit = run >= persistence
        if hit.any():
            rows.append(
                pd.DataFrame(
                    {
                        "point_id": pid,
                        "ts": g["hour_utc"].to_numpy()[hit],
                        "ratio": (g["score"] / g[col]).to_numpy()[hit],
                        "top_contributors": g["top_contributors"].to_numpy()[hit],
                    }
                )
            )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["point_id", "ts", "ratio", "top_contributors"])
    )


def season_of(hours: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    m = pd.to_datetime(hours, utc=True).dt.tz_convert("America/Chicago").dt.month
    out = pd.Series("shoulder", index=hours.index, dtype=object)
    for season, (a, b) in cfg["models"]["nbm"]["coverage_months"].items():
        months = list(range(a, b + 1)) if a <= b else list(range(a, 13)) + list(range(1, b + 1))
        out[m.isin(months).to_numpy()] = season
    return out


def exceedance_diagnosis(
    scores: pd.DataFrame,
    segments: pd.DataFrame,
    percentile: float,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Share of scored hours above the percentile threshold per point, segment (install, then each
    repair or remount), period, regime and season."""
    col = f"thr_{percentile:g}"
    s = scores.copy()
    s["hour_utc"] = pd.to_datetime(s["hour_utc"], utc=True)
    s["over"] = s["score"] > s[col]
    s["season"] = season_of(s["hour_utc"], cfg).to_numpy()
    s["period"] = "other"
    for name, (a, b) in periods.items():
        s.loc[(s["hour_utc"] >= a) & (s["hour_utc"] < b), "period"] = name
    parts = []
    for pid, g in s.groupby("point_id"):
        seg = segments[segments["point_id"] == pid].sort_values("start").reset_index(drop=True)
        idx = segment_ids(g["hour_utc"], list(seg["start"]))
        g = g.assign(segment=idx)
        g["segment_start"] = [seg["start"].iat[i] if i >= 0 else pd.NaT for i in idx]
        g["segment_reason"] = [seg["reason"].iat[i] if i >= 0 else "" for i in idx]
        parts.append(
            g.groupby(
                ["point_id", "segment", "segment_start", "segment_reason", "period", "regime", "season"]
            )
            .agg(hours=("over", "size"), exceed_share=("over", "mean"))
            .reset_index()
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

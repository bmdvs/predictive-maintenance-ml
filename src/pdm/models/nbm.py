"""Per-point normal-behavior models.

Training data: all measured hours of the point from install to the training cutoff, less the
exclusions (the days before confirmed conditions, open cases and repair periods, re-baseline windows
after repairs and remounts, flagged baselines). A point's model is used only when its training hours
cover both summer and winter. Features are re-centered per segment (install, then each repair or
remount) on the segment's first window, so a repair re-baselines the point without refitting.

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
    if not all(cov.values()) or keep.sum() < 50:
        return {}, info
    reg = clean["regime"].to_numpy()
    counts = pd.Series(reg[keep]).value_counts()
    big = [r for r, n in counts.items() if n >= nb["min_regime_healthy_hours"]]
    if len(big) == len(counts):
        models = {
            r: fit_candidate(xc[keep & (reg == r)], reg[keep & (reg == r)], cols, candidate, cfg, r, False)
            for r in big
        }
    else:
        models = {"all": fit_candidate(xc[keep], reg[keep], cols, candidate, cfg, None, len(counts) > 1)}
    return models, info


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

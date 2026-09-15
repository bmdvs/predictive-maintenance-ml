"""Per-point normal-behavior models trained on approved healthy windows.

Candidates: PCA with T-squared and SPE, Mahalanobis distance, isolation forest, and an autoencoder on
the one-third-octave summary. A separate model per regime where the regime has the configured number
of healthy hours in the window, otherwise one model with the regime as a feature. Thresholds are the
configured percentile of the healthy-window scores; alerts need the configured persistence and are
suppressed for the configured days. Inputs and outputs are pandas frames for one point.
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

from pdm.rules.persistence import ConsecutiveCounter, Suppressor

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
    """Training rows outside the approved healthy windows or past the training cutoff."""


def env_fault(df: pd.DataFrame) -> pd.Series:
    return df[["env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2"]].fillna(0.0).sum(axis=1)


def toct_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("toct_"))


def design(df: pd.DataFrame, kind: str, regime_feature: bool) -> tuple[np.ndarray, list[str]]:
    if kind == "autoencoder_third_octave":
        cols = toct_columns(df)
        return np.log10(df[cols].to_numpy(dtype=float) + 1e-12), cols
    d = df.copy()
    d["env_fault_g2"] = env_fault(d)
    cols = list(VECTOR)
    x = np.column_stack(
        [
            np.log10(d[c].to_numpy(dtype=float) + 1e-12) if c in LOG_COLUMNS else d[c].to_numpy(dtype=float)
            for c in cols
        ]
    )
    if regime_feature:
        reg = np.column_stack([(d["regime"] == r).to_numpy(dtype=float) for r in REGIMES])
        x = np.column_stack([x, reg])
        cols = cols + [f"regime_{r}" for r in REGIMES]
    return x, cols


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
    threshold: float
    regime: str | None
    regime_feature: bool
    n_train: int
    extra: dict[str, Any] = field(default_factory=dict)

    def raw_scores(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x, _ = design(df, self.candidate, self.regime_feature)
        z = self.scaler.transform(x)
        if self.candidate == "pca_t2_spe":
            p = self.model
            t = p.transform(z)
            t2 = np.sum(t**2 / p.explained_variance_, axis=1)
            rec = p.inverse_transform(t)
            spe = np.sum((z - rec) ** 2, axis=1)
            score = t2 / self.extra["t2_med"] + spe / self.extra["spe_med"]
            return score, z**2
        if self.candidate == "mahalanobis":
            prec = self.model
            score = np.einsum("ij,jk,ik->i", z, prec, z)
            return score, z**2
        if self.candidate == "isolation_forest":
            return -self.model.score_samples(z), z**2
        rec = self.model.predict(z)
        return np.mean((z - rec) ** 2, axis=1), (z - rec) ** 2

    def top_contributors(self, contrib: np.ndarray, k: int) -> list[str]:
        idx = np.argsort(-contrib, axis=1)[:, :k]
        return [",".join(self.columns[j] for j in row) for row in idx]


def fit_candidate(
    train: pd.DataFrame, candidate: str, cfg: dict[str, Any], regime: str | None, regime_feature: bool
) -> FittedModel:
    m = cfg["models"]
    seed = m["seed"]
    x, cols = design(train, candidate, regime_feature)
    sc = Scaler.fit(x)
    z = sc.transform(x)
    extra: dict[str, Any] = {}
    if candidate == "pca_t2_spe":
        model = PCA(n_components=m["nbm"]["pca_variance_kept"], svd_solver="full", random_state=seed).fit(z)
        t = model.transform(z)
        t2 = np.sum(t**2 / model.explained_variance_, axis=1)
        spe = np.sum((z - model.inverse_transform(t)) ** 2, axis=1)
        extra = {"t2_med": float(np.median(t2)) or 1.0, "spe_med": float(np.median(spe)) or 1.0}
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
    fm = FittedModel(candidate, cols, sc, model, np.inf, regime, regime_feature, len(train), extra)
    s, _ = fm.raw_scores(train)
    fm.threshold = float(np.percentile(s, m["nbm"]["threshold_percentile"]))
    return fm


def check_training_rows(
    train: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp], cutoff: pd.Timestamp | None
) -> None:
    """Refuse rows outside the approved window or at or after the training cutoff."""
    t = pd.to_datetime(train["hour_utc"])
    if ((t < window[0]) | (t >= window[1])).any():
        raise LeakedWindowError("training rows outside the approved healthy window")
    if cutoff is not None and (t >= cutoff).any():
        raise LeakedWindowError("training rows at or after the training cutoff")


def fit_point_window(
    train: pd.DataFrame,
    candidate: str,
    cfg: dict[str, Any],
    window: tuple[pd.Timestamp, pd.Timestamp],
    cutoff: pd.Timestamp | None = None,
) -> dict[str, FittedModel]:
    """Models for one point and one healthy window, keyed by regime ('all' when regime is a feature)."""
    check_training_rows(train, window, cutoff)
    clean = train[~train["implausible"].fillna(False).astype(bool)].dropna(subset=["vel_rms_mm_s"])
    need = cfg["models"]["nbm"]["min_regime_healthy_hours"]
    counts = clean["regime"].value_counts()
    big = [r for r, n in counts.items() if n >= need]
    if len(big) == len(counts) and len(big) > 0:
        return {r: fit_candidate(clean[clean["regime"] == r], candidate, cfg, r, False) for r in big}
    multi = clean["regime"].nunique() > 1
    return {"all": fit_candidate(clean, candidate, cfg, None, multi)}


def score_rows(rows: pd.DataFrame, models: dict[str, FittedModel], top_k: int) -> pd.DataFrame:
    out = []
    for key, fm in models.items():
        sel = rows if key == "all" else rows[rows["regime"] == key]
        if sel.empty:
            continue
        s, contrib = fm.raw_scores(sel)
        out.append(
            pd.DataFrame(
                {
                    "point_id": sel["point_id"].to_numpy(),
                    "hour_utc": sel["hour_utc"].to_numpy(),
                    "regime": sel["regime"].to_numpy(),
                    "score": s,
                    "threshold": fm.threshold,
                    "score_ratio": s / fm.threshold if fm.threshold > 0 else np.nan,
                    "top_contributors": fm.top_contributors(contrib, top_k),
                    "model_regime": key,
                }
            )
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def alerts_from_scores(scores: pd.DataFrame, persistence: int, suppression_days: float) -> pd.DataFrame:
    """Persisted exceedances with re-alert suppression, per point in time order."""
    rows = []
    for pid, g in scores.sort_values("hour_utc").groupby("point_id"):
        counter, sup = ConsecutiveCounter(persistence), Suppressor(suppression_days)
        for r in g.itertuples():
            if counter.update(bool(r.score_ratio > 1.0)) and sup.allow(pd.Timestamp(r.hour_utc)):
                rows.append(
                    {
                        "point_id": pid,
                        "alert_ts": r.hour_utc,
                        "score_ratio": r.score_ratio,
                        "top_contributors": r.top_contributors,
                    }
                )
    return pd.DataFrame(rows, columns=["point_id", "alert_ts", "score_ratio", "top_contributors"])

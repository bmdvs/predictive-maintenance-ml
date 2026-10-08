"""ML Model: one normal-behavior model per sensor point over all of the point's training segments.

Each segment (install, then each repair or remount) is expressed relative to its own level, set from
its first days (between recenter_days[0] and recenter_days[1], at least recenter_min_hours measured
hours); the model cannot alert while a level forms. One model per point is fit on its healthy training
hours across every segment and frozen; thresholds are percentiles of its training scores, one per
spindle speed band or one for the point, as chosen in selection. A segment that starts after the
training cutoff is re-centered only: when its level on velocity, HF acceleration or an envelope fault
band is above the point's highest earlier level by more than the plausibility factor, the segment is
flagged (WATCH) and the model stays silent on it. Model types: PCA with T-squared and SPE, Mahalanobis
distance, isolation forest and an autoencoder. Input sets: the indicator summary (RMS values, crest
factor, kurtosis, 1x and 2x orders, summed envelope fault energy), the one-third-octave bands alone, and
the extended set (one-third-octave bands, each envelope fault band, HF acceleration and the temperature
residual). Selection is between model types on the extended set; the other input sets are ablations.
(Revision 2. Originally one model per segment, refit on the 21-day re-baseline window after each
repair and updated monthly until it had seasonal coverage.)
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

from pdm.models import temperature

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
ENV_BANDS = ["env_bpfo_1_g2", "env_bpfi_1_g2", "env_bsf_1_g2"]
VECTOR_SOURCE = [c for c in VECTOR if c != "env_fault_g2"] + ENV_BANDS
LOG_COLUMNS = {
    "accel_rms_g",
    "vel_rms_mm_s",
    "hf_rms_g",
    "ord_01_g2",
    "ord_02_g2",
    "env_fault_g2",
    *ENV_BANDS,
}
PLAUSIBILITY = ["vel_rms_mm_s", "hf_rms_g", *ENV_BANDS]
REGIMES = ["light", "medium", "heavy", "running"]
BANDS = ["low", "mid", "high"]
MODEL_PREFIXES = ("pca_t2_spe", "mahalanobis", "isolation_forest", "autoencoder")
# Input sets (checkpoint (b) diagnosis decisions 2026-10-30, item 4): (a) extended, the 37 inputs as before;
# (b) vendor, the Vendor Rules' measurements plus crest factor, kurtosis and the 1x and 2x order energies;
# (c) vendor_octave, set (b) plus the 32 one-third-octave bands merged into 8 bands of 4.
INPUT_SETS = ("extended", "vendor", "vendor_octave")
CANDIDATES = (
    "pca_t2_spe",
    "mahalanobis",
    "isolation_forest",
    "autoencoder_third_octave",
    *(f"{m}_{i}" for i in INPUT_SETS for m in MODEL_PREFIXES),
)
EXTENDED = tuple(f"{m}_extended" for m in MODEL_PREFIXES)
AUTOENCODERS = ("autoencoder_third_octave", *(f"autoencoder_{i}" for i in INPUT_SETS))
MODEL_TYPES = ("pca", "mahalanobis", "isolation_forest", "autoencoder")
VENDOR_LOG = [
    "vel_rms_mm_s",
    "hf_rms_g",
    "env_bpfo_1_g2",
    "env_bpfo_2_g2",
    "env_bpfi_1_g2",
    "env_bpfi_2_g2",
    "env_bsf_1_g2",
    "env_bsf_2_g2",
    "ord_01_g2",
    "ord_02_g2",
]
VENDOR_LINEAR = ["crest_factor", "kurtosis"]
OCTAVE_GROUP = 4  # 32 bands over 10 Hz to 10 kHz (about 3.2 per octave) merged 4 at a time into 8 bands


def model_type(candidate: str) -> str:
    """The model family of a candidate, whatever its input set."""
    if candidate.startswith("pca"):
        return "pca"
    if candidate.startswith("mahalanobis"):
        return "mahalanobis"
    if candidate.startswith("isolation_forest"):
        return "isolation_forest"
    return "autoencoder"


def input_set(candidate: str) -> str:
    for i in sorted(INPUT_SETS, key=len, reverse=True):
        if candidate.endswith(f"_{i}"):
            return i
    return "third_octave" if candidate == "autoencoder_third_octave" else "summary"


TEMP_RESID = "temp_resid_c"
EPS = 1e-12


class LeakedWindowError(ValueError):
    """Training rows at or after the training cutoff, or inside an excluded interval."""


def env_fault(df: pd.DataFrame) -> pd.Series:
    return df[ENV_BANDS].fillna(0.0).sum(axis=1)


def toct_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("toct_"))


def _log(v: np.ndarray) -> np.ndarray:
    return np.log10(np.where(np.isfinite(v), v, 0.0) + EPS)


def octave_bands(df: pd.DataFrame) -> tuple[list[np.ndarray], list[str]]:
    """The one-third-octave band energies summed 4 at a time (8 bands), on a log scale."""
    cols = toct_columns(df)
    parts, names = [], []
    for k in range(0, len(cols), OCTAVE_GROUP):
        grp = cols[k : k + OCTAVE_GROUP]
        e = sum(np.where(np.isfinite(v), v, 0.0) for v in (df[c].to_numpy(dtype=float) for c in grp))
        parts.append(_log(e))
        names.append(f"oct_{k // OCTAVE_GROUP + 1:02d}_g2")
    return parts, names


def features(df: pd.DataFrame, candidate: str) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for a candidate: energies and RMS values on a log scale; crest factor, kurtosis and
    the temperature residual (degrees C) linear."""
    iset = input_set(candidate)
    if iset in ("vendor", "vendor_octave"):
        parts = [_log(df[c].to_numpy(dtype=float)) for c in VENDOR_LOG]
        parts += [df[c].to_numpy(dtype=float) for c in VENDOR_LINEAR]
        parts.append(df[TEMP_RESID].to_numpy(dtype=float))
        cols = [*VENDOR_LOG, *VENDOR_LINEAR, TEMP_RESID]
        if iset == "vendor_octave":
            op, on = octave_bands(df)
            parts += op
            cols += on
        return np.column_stack(parts), cols
    if candidate in AUTOENCODERS or candidate in EXTENDED:
        cols = toct_columns(df)
        parts = [_log(df[c].to_numpy(dtype=float)) for c in cols]
        if candidate in EXTENDED:
            extra = [*ENV_BANDS, "hf_rms_g"]
            parts += [_log(df[c].to_numpy(dtype=float)) for c in extra]
            parts.append(df[TEMP_RESID].to_numpy(dtype=float))
            cols = [*cols, *extra, TEMP_RESID]
        return np.column_stack(parts), cols
    d = df.assign(env_fault_g2=env_fault(df))
    x = np.column_stack(
        [_log(d[c].to_numpy(dtype=float)) if c in LOG_COLUMNS else d[c].to_numpy(dtype=float) for c in VECTOR]
    )
    return x, list(VECTOR)


def speed_band(rows: pd.DataFrame, rated_rpm: float | None, cfg: dict[str, Any]) -> np.ndarray:
    """Spindle speed band (low, mid, high) from the configured fractions of rated speed; 'all' on points
    without a spindle speed."""
    if not rated_rpm or not str(rows["point_class"].iat[0]).startswith("spindle"):
        return np.full(len(rows), "all", dtype=object)
    edges = cfg["regime"]["speed_bands_frac_rated"]
    f = rows["speed_rpm_at_measurement"].to_numpy(dtype=float) / float(rated_rpm)
    out = np.full(len(rows), BANDS[0], dtype=object)
    for i, e in enumerate(edges[1:-1], start=1):
        out[f >= e] = BANDS[i]
    return out


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
class Settings:
    """Candidate settings chosen in selection."""

    candidate: str
    hidden: tuple[int, ...] = ()
    epochs: int = 0
    threshold_mode: str = "single"

    @property
    def key(self) -> str:
        if self.candidate in AUTOENCODERS:  # every autoencoder candidate carries its network and length
            net = "-".join(str(h) for h in self.hidden)
            return f"{self.candidate} [{net}] {self.epochs} ep, {self.threshold_mode}"
        return f"{self.candidate}, {self.threshold_mode}"


def sweep_percentiles(cfg: dict[str, Any]) -> list[float]:
    """Every percentile a model keeps a threshold for: the operating-curve sweep, its extension below the
    99th percentile and the operating percentile."""
    nb = cfg["models"]["nbm"]
    return sorted(
        {float(p) for p in nb["sweep_percentiles"]}
        | {float(p) for p in nb.get("sweep_extension_percentiles", [])}
        | {float(nb["threshold_percentile"])}
    )


def settings_grid(cfg: dict[str, Any]) -> list[Settings]:
    """Every candidate with each autoencoder network size and training length in the selection grid and
    each threshold mode, simplest first."""
    nb = cfg["models"]["nbm"]
    grid = nb["selection_grid"]
    out = []
    for cand in nb["candidates"]:
        for mode in grid["threshold_mode"]:
            if cand in AUTOENCODERS:
                for h in grid["autoencoder_hidden"]:
                    for e in grid["autoencoder_epochs"]:
                        out.append(Settings(cand, tuple(h), int(e), mode))
            else:
                out.append(Settings(cand, (), 0, mode))
    return out


def complexity(s: Settings) -> tuple[int, int, int, int]:
    return (
        MODEL_TYPES.index(model_type(s.candidate)),
        sum(s.hidden),
        s.epochs,
        0 if s.threshold_mode == "single" else 1,
    )


@dataclass
class Segment:
    start: pd.Timestamp
    reason: str
    ready: pd.Timestamp | None
    level: np.ndarray | None
    plaus: dict[str, float]
    n_level_hours: int
    flagged: bool = False


def _naive(ts: Any) -> np.ndarray:
    s = pd.to_datetime(pd.Series(ts), utc=True).dt.tz_localize(None)
    return np.asarray(s, dtype="datetime64[ns]")


def segment_ids(hours: Any, starts: list[pd.Timestamp]) -> np.ndarray:
    st = np.sort(_naive(starts))
    return np.searchsorted(st, _naive(hours), side="right") - 1


def _plaus_levels(rows: pd.DataFrame, m: np.ndarray) -> dict[str, float]:
    out = {}
    for c in PLAUSIBILITY:
        v = rows.loc[m, c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        out[c] = float(np.log10(np.median(v) + EPS)) if len(v) else np.nan
    return out


def segment_levels(
    rows: pd.DataFrame,
    x: np.ndarray,
    keys: np.ndarray,
    windows: pd.DataFrame,
    offsets: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> list[Segment]:
    """Each segment's level: the median over its re-centering window of the features less the point's
    operating-key offsets. The window ends once recenter_days[0] have passed and recenter_min_hours
    measured hours have accumulated, at most recenter_days[1] after the start; a segment with fewer than
    recenter_floor_hours by then has no level and is not scored."""
    nb = cfg["models"]["nbm"]
    lo, hi = (pd.Timedelta(days=d) for d in nb["recenter_days"])
    t = _naive(rows["hour_utc"])
    w = windows.sort_values("start").reset_index(drop=True)
    seg = segment_ids(rows["hour_utc"], list(w["start"]))
    adj = (
        x - np.vstack([offsets.get(k, offsets.get("_all", np.zeros(x.shape[1]))) for k in keys])
        if len(x)
        else x
    )
    out = []
    for k, r in w.iterrows():
        a = _naive([r["start"]])[0]
        mine = np.flatnonzero((seg == k) & (t >= a) & (t < a + np.timedelta64(hi)))
        ready = None
        if len(mine) >= nb["recenter_min_hours"]:
            nth = t[mine[nb["recenter_min_hours"] - 1]]
            ready = max(a + np.timedelta64(lo), nth + np.timedelta64(1, "h"))
        if ready is None or ready > a + np.timedelta64(hi):
            ready = a + np.timedelta64(hi) if len(mine) >= nb["recenter_floor_hours"] else None
        if ready is None:
            out.append(Segment(r["start"], r["reason"], None, None, {}, len(mine)))
            continue
        m = (seg == k) & (t >= a) & (t < ready)
        out.append(
            Segment(
                r["start"],
                r["reason"],
                pd.Timestamp(ready).tz_localize("UTC"),
                np.nanmedian(adj[m], axis=0),
                _plaus_levels(rows, m),
                int(m.sum()),
            )
        )
    return out


def operating_key(rows: pd.DataFrame, bands: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{r}|{b}" for r, b in zip(rows["regime"].to_numpy(), bands, strict=True)], dtype=object
    )


def key_offsets(
    x: np.ndarray, keys: np.ndarray, seg_first: np.ndarray, min_rows: int
) -> dict[str, np.ndarray]:
    """Median of features less each row's segment first-window median, per operating key (load regime and
    speed band); '_all' pools every key."""
    d = x - seg_first
    out = {"_all": np.nanmedian(d, axis=0)}
    for k in np.unique(keys):
        m = keys == k
        if m.sum() >= min_rows:
            out[str(k)] = np.nanmedian(d[m], axis=0)
    return out


def _one_hot(rows: pd.DataFrame, bands: np.ndarray) -> np.ndarray:
    reg = rows["regime"].to_numpy()
    return np.column_stack(
        [(reg == r).astype(float) for r in REGIMES] + [(bands == b).astype(float) for b in BANDS]
    )


@dataclass
class PointModel:
    point_id: str
    point_class: str
    settings: Settings
    columns: list[str]
    rated_rpm: float | None
    temp_fit: temperature.TempFit
    offsets: dict[str, np.ndarray]
    scaler: Scaler
    model: Any
    extra: dict[str, Any]
    thresholds: dict[str, dict[str, dict[float, float]]]
    train_levels: list[dict[str, float]]
    n_train: int
    segments_trained: int
    info: dict[str, Any] = field(default_factory=dict)

    def raw_scores(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cand = model_type(self.settings.candidate)
        if cand == "pca":
            p = self.model
            t = p.transform(z)
            t2 = np.sum(t**2 / p.explained_variance_, axis=1)
            spe = np.sum((z - p.inverse_transform(t)) ** 2, axis=1)
            return t2 / self.extra["t2_med"] + spe / self.extra["spe_med"], z**2
        if cand == "mahalanobis":
            return np.einsum("ij,jk,ik->i", z, self.model, z), z**2
        if cand == "isolation_forest":
            return -self.model.score_samples(z), z**2
        rec = self.model.predict(z)
        return np.mean((z - rec) ** 2, axis=1), (z - rec) ** 2

    def contributors(self, contrib: np.ndarray, k: int) -> list[str]:
        names = self.columns + [f"regime_{r}" for r in REGIMES] + [f"band_{b}" for b in BANDS]
        idx = np.argsort(-contrib, axis=1)[:, :k]
        return [",".join(names[j] for j in row) for row in idx]

    def threshold_for(self, bands: np.ndarray, pct: float, mode: str | None = None) -> np.ndarray:
        mode = mode or self.settings.threshold_mode
        tb = self.thresholds[mode]
        return np.asarray([tb.get(b, tb["_point"])[float(pct)] for b in bands], dtype=float)


def prepare(rows: pd.DataFrame, ambient: pd.Series | None) -> pd.DataFrame:
    """Measured, plausible rows in time order, with the compressor-room ambient joined by hour."""
    clean = (
        rows[~rows["implausible"].fillna(False).astype(bool)]
        .dropna(subset=["vel_rms_mm_s"])
        .sort_values("hour_utc", kind="stable")
        .reset_index(drop=True)
    )
    if ambient is not None and len(clean):
        h = pd.to_datetime(clean["hour_utc"], utc=True)
        clean[temperature.AMBIENT] = h.map(ambient).to_numpy(dtype=float)
    return clean


def training_mask(
    hours: pd.Series,
    cutoff: pd.Timestamp,
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]],
    start: pd.Timestamp | None = None,
) -> np.ndarray:
    t = pd.to_datetime(hours, utc=True)
    keep = (t < cutoff).to_numpy()
    if start is not None:
        keep &= (t >= start).to_numpy()
    for a, b in excluded:
        keep &= ~((t >= a) & (t < b)).to_numpy()
    return keep


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


def _fit_estimator(z: np.ndarray, s: Settings, cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    m = cfg["models"]
    seed = m["seed"]
    kind = model_type(s.candidate)
    if kind == "pca":
        model = PCA(n_components=m["nbm"]["pca_variance_kept"], svd_solver="full", random_state=seed).fit(z)
        t = model.transform(z)
        t2 = np.sum(t**2 / model.explained_variance_, axis=1)
        spe = np.sum((z - model.inverse_transform(t)) ** 2, axis=1)
        return model, {"t2_med": float(np.median(t2)) or 1.0, "spe_med": float(np.median(spe)) or 1.0}
    if kind == "mahalanobis":
        return LedoitWolf().fit(z).precision_, {}
    if kind == "isolation_forest":
        return IsolationForest(
            n_estimators=m["nbm"]["isolation_forest"]["n_estimators"], random_state=seed
        ).fit(z), {}
    return MLPRegressor(hidden_layer_sizes=tuple(s.hidden), max_iter=int(s.epochs), random_state=seed).fit(
        z, z
    ), {}


def _thresholds(
    scores: np.ndarray, bands: np.ndarray, pcts: list[float], min_hours: int
) -> dict[str, dict[str, dict[float, float]]]:
    point = {float(p): float(np.percentile(scores, p)) for p in pcts}
    by_band: dict[str, dict[float, float]] = {"_point": point}
    for b in np.unique(bands):
        m = bands == b
        if m.sum() >= min_hours:
            by_band[str(b)] = {float(p): float(np.percentile(scores[m], p)) for p in pcts}
    return {"single": {"_point": point}, "speed_band": by_band}


def fit_point(
    rows: pd.DataFrame,
    windows: pd.DataFrame,
    settings_list: list[Settings],
    cfg: dict[str, Any],
    fit_start: pd.Timestamp,
    cutoff: pd.Timestamp,
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]],
    rated_rpm: float | None,
    ambient: pd.Series | None,
) -> tuple[dict[str, PointModel], dict[str, Any]]:
    """One frozen model per settings for a point, fit on its healthy hours from fit_start to the cutoff
    across every segment that starts before the cutoff. Returns models keyed by Settings.key and the
    point's training information."""
    nb = cfg["models"]["nbm"]
    pid = str(rows["point_id"].iat[0])
    pclass = str(rows["point_class"].iat[0])
    clean = prepare(rows, ambient)
    info: dict[str, Any] = {"point_id": pid, "point_class": pclass}
    if clean.empty:
        return {}, {**info, "n_train": 0, "reason": "no measured hours"}
    win = (
        windows[pd.to_datetime(windows["start"], utc=True) < cutoff]
        .sort_values("start")
        .reset_index(drop=True)
    )
    if win.empty:
        return {}, {**info, "n_train": 0, "reason": "no segment before the cutoff"}
    eligible = training_mask(clean["hour_utc"], cutoff, excluded, fit_start)
    tf = temperature.fit(clean[eligible], pclass)
    clean[TEMP_RESID] = tf.residual(clean)
    bands = speed_band(clean, rated_rpm, cfg)
    keys = operating_key(clean, bands)
    seg = segment_ids(clean["hour_utc"], list(win["start"]))
    t = _naive(clean["hour_utc"])
    models: dict[str, PointModel] = {}
    by_candidate: dict[str, tuple[np.ndarray, list[str]]] = {}
    for s in settings_list:
        if s.candidate not in by_candidate:
            by_candidate[s.candidate] = features(clean, s.candidate)
    shared: dict[str, Any] = {}
    for cand, (x, cols) in by_candidate.items():
        first = np.full_like(x, np.nan)
        zero = {"_all": np.zeros(x.shape[1])}
        segs0 = segment_levels(clean, x, keys, win, zero, cfg)
        for k, sg in enumerate(segs0):
            if sg.level is not None:
                first[seg == k] = sg.level
        in_win = np.zeros(len(clean), dtype=bool)
        for k, sg in enumerate(segs0):
            a = _naive([sg.start])[0]
            b = _naive([sg.ready])[0] if sg.ready is not None else np.datetime64("NaT")
            in_win |= (seg == k) & (t >= a) & ((t < b) if sg.ready is not None else True)
        ok = eligible & ~in_win & np.isfinite(first).all(axis=1) & np.isfinite(x).all(axis=1) & (seg >= 0)
        offsets = key_offsets(x[ok], keys[ok], first[ok], nb["min_key_hours"]) if ok.any() else zero
        segs = segment_levels(clean, x, keys, win, offsets, cfg)
        lev = np.full_like(x, np.nan)
        for k, sg in enumerate(segs):
            if sg.level is not None:
                lev[seg == k] = sg.level
        xc = x - lev
        train = ok & np.isfinite(xc).all(axis=1)
        shared[cand] = (xc, train, offsets, segs, cols)
    info.update(
        {
            "segments_before_cutoff": len(win),
            "temp_fit": tf.to_dict(),
        }
    )
    pcts = sweep_percentiles(cfg)
    oh = _one_hot(clean, bands)
    fitted: dict[tuple[str, tuple[int, ...], int], tuple[Scaler, Any, dict[str, Any], np.ndarray]] = {}
    for s in settings_list:
        xc, train, offsets, segs, cols = shared[s.candidate]
        n = int(train.sum())
        if n < nb["min_train_hours"]:
            info.setdefault("too_few", []).append(s.key)
            continue
        arch = (s.candidate, tuple(s.hidden), int(s.epochs))
        if arch not in fitted:
            check_training_rows(clean.loc[train, "hour_utc"], cutoff, excluded)
            raw = np.column_stack([xc[train], oh[train]])
            sc0 = Scaler.fit(raw)
            z0 = sc0.transform(raw)
            est0, extra0 = _fit_estimator(z0, s, cfg)
            keep = np.ones(len(raw), dtype=bool)
            trim = float(nb.get("robust_trim_share") or 0.0)
            if trim > 0:
                # robust fit (checkpoint (b) diagnosis decisions 2026-10-30, item 3.2): drop the training
                # hours whose first-fit score is above the (1 - trim) quantile and refit once; the
                # thresholds come from the kept hours (was a single fit on every training hour)
                probe = PointModel(
                    pid, pclass, s, cols, rated_rpm, tf, offsets, sc0, est0, extra0, {}, [], 0, 0
                )
                sc_first, _ = probe.raw_scores(z0)
                keep = sc_first <= np.quantile(sc_first, 1.0 - trim)
                if keep.sum() >= nb["min_train_hours"]:
                    sc0 = Scaler.fit(raw[keep])
                    z0 = sc0.transform(raw[keep])
                    est0, extra0 = _fit_estimator(z0, s, cfg)
                else:
                    keep = np.ones(len(raw), dtype=bool)
            fitted[arch] = (sc0, est0, extra0, z0, keep)
        sc, est, extra, z, keep = fitted[arch]
        pm = PointModel(
            pid,
            pclass,
            s,
            cols,
            rated_rpm,
            tf,
            offsets,
            sc,
            est,
            extra,
            {},
            [g.plaus for g in segs if g.level is not None],
            n,
            sum(g.level is not None for g in segs),
        )
        scores, _ = pm.raw_scores(z)
        tb = bands[train][keep]
        pm.thresholds = _thresholds(scores, tb, pcts, nb["min_regime_healthy_hours"])
        pm.n_train = int(keep.sum())
        pm.info = {
            "n_train": int(keep.sum()),
            "n_before_trim": n,
            "cutoff": cutoff,
            "fit_start": fit_start,
            "train_bands": pd.Series(tb).value_counts().to_dict(),
        }
        models[s.key] = pm
    info["n_train"] = max((m.n_train for m in models.values()), default=0)
    return models, info


def plausibility_factor(models: list[PointModel], quantile: float, fallback: float) -> tuple[float, int]:
    """The quantile of the high-side ratio between consecutive healthy segment levels across all points
    in training, on velocity, HF acceleration and the envelope fault bands. Returns the factor and the
    number of ratios it rests on; with fewer than 5 ratios the fallback is used."""
    ratios = []
    for m in models:
        lv = m.train_levels
        for a, b in zip(lv[:-1], lv[1:], strict=False):
            for c in PLAUSIBILITY:
                if np.isfinite(a.get(c, np.nan)) and np.isfinite(b.get(c, np.nan)):
                    ratios.append(10 ** (b[c] - a[c]))
    if len(ratios) < 5:
        return fallback, len(ratios)
    return float(np.quantile(ratios, quantile)), len(ratios)


def score_point(
    model: PointModel,
    rows: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    plaus_factor: float,
    top_k: int,
    ambient: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores from start to end with the frozen model. Every segment is re-centered on its own window
    (no score while the level forms); a segment starting at or after the model's training cutoff is
    checked for plausibility and, when flagged, left silent. Returns hourly rows (state 'scored',
    'recentering', 'plausibility' or 'no level') and one row per segment."""
    nb = cfg["models"]["nbm"]
    clean = prepare(rows, ambient)
    if clean.empty:
        return pd.DataFrame(), pd.DataFrame()
    clean[TEMP_RESID] = model.temp_fit.residual(clean)
    win = windows.sort_values("start").reset_index(drop=True)
    win = win[pd.to_datetime(win["start"], utc=True) < end].reset_index(drop=True)
    if win.empty:
        return pd.DataFrame(), pd.DataFrame()
    bands = speed_band(clean, model.rated_rpm, cfg)
    keys = operating_key(clean, bands)
    x, _ = features(clean, model.settings.candidate)
    segs = segment_levels(clean, x, keys, win, model.offsets, cfg)
    seg = segment_ids(clean["hour_utc"], list(win["start"]))
    t = _naive(clean["hour_utc"])
    cutoff = model.info.get("cutoff")
    lvl = np.full_like(x, np.nan)
    state = np.full(len(clean), "no level", dtype=object)
    log_factor = float(np.log10(plaus_factor))
    hist = list(model.train_levels)
    seg_rows = []
    for k, sg in enumerate(segs):
        rows_k = seg == k
        trained = cutoff is not None and sg.start < cutoff
        if sg.level is not None and not trained:
            prior = [h for h in hist if h]
            highs = {
                c: max((h[c] for h in prior if np.isfinite(h.get(c, np.nan))), default=np.nan)
                for c in PLAUSIBILITY
            }
            over = [
                c
                for c in PLAUSIBILITY
                if np.isfinite(sg.plaus.get(c, np.nan))
                and np.isfinite(highs[c])
                and sg.plaus[c] - highs[c] > log_factor
            ]
            sg.flagged = bool(over)
            if not sg.flagged:
                hist.append(sg.plaus)
        else:
            over = []
        a = _naive([sg.start])[0]
        if sg.level is None:
            state[rows_k] = "no level"
        else:
            r = _naive([sg.ready])[0]
            state[rows_k & (t >= a) & (t < r)] = "recentering"
            after = rows_k & (t >= r)
            state[after] = "plausibility" if sg.flagged else "scored"
            lvl[after] = sg.level
        seg_rows.append(
            {
                "point_id": model.point_id,
                "segment": k,
                "segment_start": sg.start,
                "reason": sg.reason,
                "level_ready": sg.ready,
                "level_hours": sg.n_level_hours,
                "trained": bool(trained),
                "plausibility_flag": bool(sg.flagged),
                "plausibility_over": ",".join(over),
                **{f"level_log10_{c}": sg.plaus.get(c, np.nan) for c in PLAUSIBILITY},
            }
        )
    hr = pd.to_datetime(clean["hour_utc"], utc=True)
    span = ((hr >= start) & (hr < end)).to_numpy()
    sc = (state == "scored") & span & np.isfinite(lvl).all(axis=1) & np.isfinite(x).all(axis=1)
    score = np.full(len(clean), np.nan)
    contrib_txt = np.full(len(clean), "", dtype=object)
    if sc.any():
        raw = np.column_stack([x[sc] - lvl[sc], _one_hot(clean[sc], bands[sc])])
        s, contrib = model.raw_scores(model.scaler.transform(raw))
        score[sc] = s
        contrib_txt[sc] = model.contributors(contrib, top_k)
    out = pd.DataFrame(
        {
            "point_id": model.point_id,
            "hour_utc": hr.to_numpy(),
            "segment": seg,
            "regime": clean["regime"].to_numpy(),
            "speed_band": bands,
            "state": state,
            "score": score,
            "temp_resid_c": clean[TEMP_RESID].to_numpy(dtype=float),
            "top_contributors": contrib_txt,
        }
    )[span]
    pcts = sweep_percentiles(cfg)
    b = out["speed_band"].to_numpy()
    for p in pcts:
        out[f"thr_{p:g}"] = model.threshold_for(b, p)
    out["threshold"] = model.threshold_for(b, nb["threshold_percentile"])
    return out.reset_index(drop=True), pd.DataFrame(seg_rows)


def exceedance_hours(scores: pd.DataFrame, threshold_col: str, persistence: int) -> pd.DataFrame:
    """Hours at which the score has exceeded the threshold column on persistence consecutive scored hours
    (the persisted exceedances that ALERT episodes are built from)."""
    rows = []
    s = scores[scores["state"] == "scored"] if "state" in scores else scores
    for pid, g in s.sort_values("hour_utc").groupby("point_id"):
        over = (g["score"] > g[threshold_col]).to_numpy()
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
                        "ratio": (g["score"] / g[threshold_col]).to_numpy()[hit],
                        "top_contributors": g["top_contributors"].to_numpy()[hit],
                    }
                )
            )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["point_id", "ts", "ratio", "top_contributors"])
    )

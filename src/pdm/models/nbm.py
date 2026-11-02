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
# Model fixes after the first evaluation-year scoring (2026-11-01), each behind a switch in
# models.nbm.fixes; all off by default (the behavior before the fixes)
FIX_DEFAULTS: dict[str, Any] = {
    "f1_hold_level": False,
    "f2_post_repair_window": False,
    "f2_days": [3, 14],
    "f2_min_hours": 24,
    "f2_train_factor": None,
    "f3_spindle_temp_regime": False,
    "f4_high_ratio": False,
    "f4_ratio": 3.0,
    "f4_hours": 2,
    "f1_scope": False,
    "f4_exclude_pump_low_pressure": False,
    "t1_vendor_exclusion": False,
    "t2_season_min_months": None,
    "t3_min_train_hours": None,
    "s1_on_component_window": False,
    "s1_max_days": 60,
    "c1_wear_stretch": False,
    "c2_season_clamp": False,
    "c3_flat_rearm": False,
    "c4_other_repair_window": False,
    "c3_lookback_days": 150,
}
# F1-scope (2026-11-01): parts that mark a repair on the component a point measures, by point kind
COMPONENT_PARTS = {
    "spindle": ("SPINDLE-CARTRIDGE", "SPINDLE-REBUILD"),
    "pump": ("PUMP-UNIT", "6205", "6309"),
    "compressor": ("6316", "NU2210"),
}


def point_kind(point_id: str) -> str:
    if str(point_id).startswith("CMP-"):
        return "compressor"
    return "spindle" if "-SP-" in str(point_id) else "pump"


def component_repairs(windows: pd.DataFrame, wo: pd.DataFrame) -> pd.DataFrame:
    """The windows with on_component: the repair that opened the segment replaced parts of the component the
    point measures (a spindle cartridge or rebuild on spindle points; a pump unit or pump bearing on pump
    points; motor or airend bearings on compressor points), from the work order's parts."""
    parts = wo.set_index("wo_id")["parts"].astype(str).to_dict()
    out = windows.copy()
    flags = []
    for pid, reason in zip(out["point_id"], out["reason"], strict=True):
        if not is_repair(reason):
            flags.append(False)
            continue
        wid = str(reason).split(" ", 1)[1] if " " in str(reason) else ""
        have = parts.get(wid, "")
        flags.append(any(k in have for k in COMPONENT_PARTS[point_kind(pid)]))
    out["on_component"] = flags
    return out


def fixes(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**FIX_DEFAULTS, **(cfg["models"]["nbm"].get("fixes") or {})}


def is_repair(reason: Any) -> bool:
    return str(reason).startswith("repair")


def is_remount(reason: Any) -> bool:
    return str(reason).startswith("remount")


REGIMES = ["light", "medium", "heavy", "running"]
BANDS = ["low", "mid", "high"]
MODEL_PREFIXES = ("pca_t2_spe", "mahalanobis", "isolation_forest", "autoencoder")
# Input sets (checkpoint (b) diagnosis decisions 2026-10-30, item 4): (a) extended, the 37 inputs as before;
# (b) vendor, the Vendor Rules' measurements plus crest factor, kurtosis and the 1x and 2x order energies;
# (c) vendor_octave, set (b) plus the 32 one-third-octave bands merged into 8 bands of 4.
INPUT_SETS = ("extended", "vendor", "vendor_octave", "full", "full_env")
# (e) full_env (2026-10-31): full plus the envelope 3rd harmonics of BPFO, BPFI and BSF (the export carries
# harmonics 1 to 3; set (b) holds the 1st and 2nd) and the sideband energies (BPFI +/- 1x, BSF +/- FTF);
# 52 inputs
FULL_ENV_EXTRA = [
    "env_bpfo_3_g2",
    "env_bpfi_3_g2",
    "env_bsf_3_g2",
    "env_bpfi_sb_lo_g2",
    "env_bpfi_sb_hi_g2",
    "env_bsf_sb_lo_g2",
    "env_bsf_sb_hi_g2",
]
# (d) full (2026-10-31): the 32 one-third-octave bands plus the 13 inputs of set (b), 45 inputs; no input
# counted twice (set (b) carries no third-octave band)
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
    if iset in ("vendor", "vendor_octave", "full", "full_env"):
        parts = [_log(df[c].to_numpy(dtype=float)) for c in VENDOR_LOG]
        parts += [df[c].to_numpy(dtype=float) for c in VENDOR_LINEAR]
        parts.append(df[TEMP_RESID].to_numpy(dtype=float))
        cols = [*VENDOR_LOG, *VENDOR_LINEAR, TEMP_RESID]
        if iset in ("full", "full_env"):
            tc = toct_columns(df)
            parts = [_log(df[c].to_numpy(dtype=float)) for c in tc] + parts
            cols = [*tc, *cols]
        if iset == "full_env":
            parts += [_log(df[c].to_numpy(dtype=float)) for c in FULL_ENV_EXTRA]
            cols += FULL_ENV_EXTRA
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
        """Median and robust scale per column. The robust scale (IQR / 1.349) is floored at the column's
        standard deviation (parity failure decisions, 2026-10-31: on HMC-08-CP the IQR of HF acceleration
        was 43 times below its standard deviation, so one input took 99 percent of the variance); a
        constant column takes 1. (Was the IQR / 1.349, the standard deviation only where the IQR is zero.)"""
        c = np.nanmedian(x, axis=0)
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        s = np.maximum((q75 - q25) / 1.349, np.nanstd(x, axis=0))
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
    # final tuning round (2026-10-31): the autoencoder's L2 weight decay (None: the library default 1e-4)
    # and early stopping on a time-ordered validation slice in place of a fixed number of epochs
    weight_decay: float | None = None
    early_stopping: bool = False

    @property
    def key(self) -> str:
        if self.candidate in AUTOENCODERS:  # every autoencoder candidate carries its network and length
            net = "-".join(str(h) for h in self.hidden)
            length = "early stopping" if self.early_stopping else f"{self.epochs} ep"
            wd = "" if self.weight_decay is None else f", wd {self.weight_decay:g}"
            return f"{self.candidate} [{net}] {length}{wd}, {self.threshold_mode}"
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
    s1_after: pd.Timestamp | None = None,
) -> list[Segment]:
    """Each segment's level: the median over its re-centering window of the features less the point's
    operating-key offsets. The window ends once recenter_days[0] have passed and recenter_min_hours
    measured hours have accumulated, at most recenter_days[1] after the start; a segment with fewer than
    recenter_floor_hours by then has no level and is not scored."""
    nb = cfg["models"]["nbm"]
    lo, hi = (pd.Timedelta(days=d) for d in nb["recenter_days"])
    fx = fixes(cfg)
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
        if (
            fx["s1_on_component_window"]
            and s1_after is not None
            and pd.Timestamp(r["start"]) >= s1_after
            and is_repair(r["reason"])
            and bool(r.get("on_component", False))
        ):
            # S1 (2026-11-01, switch models.nbm.fixes.s1_on_component_window; scoring only): after a repair
            # on the measured component the level window runs from recenter_days[0] until
            # recenter_min_hours measured hours exist, at most s1_max_days; no level until then (WATCH)
            span_h = np.timedelta64(int(fx["s1_max_days"]) * 24, "h")
            mine = np.flatnonzero((seg == k) & (t >= a) & (t < a + span_h))
            if len(mine) < nb["recenter_min_hours"]:
                out.append(Segment(r["start"], r["reason"], None, None, {}, len(mine)))
                continue
            nth = t[mine[nb["recenter_min_hours"] - 1]]
            ready = max(a + np.timedelta64(lo), nth + np.timedelta64(1, "h"))
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
            continue
        c4 = fx["c4_other_repair_window"] and is_repair(r["reason"]) and not bool(r.get("on_component", True))
        if (fx["f2_post_repair_window"] and is_repair(r["reason"])) or c4:
            # F2 (2026-11-01, switch models.nbm.fixes.f2_post_repair_window): after a repair the level is the
            # median over days f2_days[0] to f2_days[1] (at least f2_min_hours measured hours); the segment
            # is scored against the previous level until then. Fewer hours: no level (held, F1).
            d0, d1 = (np.timedelta64(int(float(d) * 24), "h") for d in fx["f2_days"])
            m = (seg == k) & (t >= a + d0) & (t < a + d1)
            if int(m.sum()) < int(fx["f2_min_hours"]):
                out.append(Segment(r["start"], r["reason"], None, None, {}, int(m.sum())))
                continue
            out.append(
                Segment(
                    r["start"],
                    r["reason"],
                    pd.Timestamp(a + d1).tz_localize("UTC"),
                    np.nanmedian(adj[m], axis=0),
                    _plaus_levels(rows, m),
                    int(m.sum()),
                )
            )
            continue
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
    # per-input limits (fold diagnosis round 2 decision, 2026-10-31): each standardized input's high-side
    # limit, the configured percentile of its values on the point's healthy training hours; empty = off
    input_limits: dict[str, float] = field(default_factory=dict)
    # base-model decision 2026-10-31: the indicator columns kept (those that vary in the point's training);
    # None keeps all seven (the earlier models)
    indicator_keep: np.ndarray | None = None

    def indicators(self, rows: pd.DataFrame, bands: np.ndarray) -> np.ndarray:
        oh = _one_hot(rows, bands)
        return oh if self.indicator_keep is None else oh[:, self.indicator_keep]

    def raw_scores(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cand = model_type(self.settings.candidate)
        if cand == "pca":
            p = self.model
            t = p.transform(z)
            t2 = np.sum(t**2 / p.explained_variance_, axis=1)
            if self.extra.get("spe_med") is None:  # no residual subspace: T2 alone (2026-10-31)
                return t2 / self.extra["t2_med"], z**2
            spe = np.sum((z - p.inverse_transform(t)) ** 2, axis=1)
            return t2 / self.extra["t2_med"] + spe / self.extra["spe_med"], z**2
        if cand == "mahalanobis":
            loc = self.extra.get("location")
            zc = z - loc if loc is not None else z  # models fitted before the fix carry no location
            return np.einsum("ij,jk,ik->i", zc, self.model, zc), zc**2
        if cand == "isolation_forest":
            return -self.model.score_samples(z), z**2
        rec = self.model.predict(z)
        return np.mean((z - rec) ** 2, axis=1), (z - rec) ** 2

    def contributors(self, contrib: np.ndarray, k: int) -> list[str]:
        ind = [f"regime_{r}" for r in REGIMES] + [f"band_{b}" for b in BANDS]
        if self.indicator_keep is not None:
            ind = [n for n, kp in zip(ind, self.indicator_keep, strict=True) if kp]
        names = self.columns + ind
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
        # parity failure decisions (2026-10-31): where the PCA keeps every component of the training
        # matrix's rank there is no residual subspace and SPE is undefined (rounding only); the score is T2
        # alone (was T2 plus SPE, each over its training median, on every point)
        rank = int(np.linalg.matrix_rank(z))
        if model.n_components_ >= rank:
            return model, {"t2_med": float(np.median(t2)) or 1.0, "spe_med": None, "rank": rank}
        spe = np.sum((z - model.inverse_transform(t)) ** 2, axis=1)
        return model, {
            "t2_med": float(np.median(t2)) or 1.0,
            "spe_med": float(np.median(spe)) or 1.0,
            "rank": rank,
        }
    if kind == "mahalanobis":
        # 2026-10-31 (centering defect): the Ledoit-Wolf location (the training mean of z) is kept and
        # subtracted at score time, so the score is the squared distance from the fitted location (was the
        # precision alone, scored around the scaler's median: an offset m'Pm plus a cross term 2(z - m)'Pm)
        lw = LedoitWolf().fit(z)
        return lw.precision_, {"location": lw.location_.astype(float)}
    if kind == "isolation_forest":
        return IsolationForest(
            n_estimators=m["nbm"]["isolation_forest"]["n_estimators"], random_state=seed
        ).fit(z), {}
    alpha = 1e-4 if s.weight_decay is None else float(s.weight_decay)
    if not s.early_stopping:
        return MLPRegressor(
            hidden_layer_sizes=tuple(s.hidden), max_iter=int(s.epochs), alpha=alpha, random_state=seed
        ).fit(z, z), {}
    # early stopping (2026-10-31): the last share of the training hours by time is the validation slice;
    # train epoch by epoch on the rest, stop after patience epochs without a lower validation rebuild error
    # (at most max_epochs) and keep the best weights
    es = {
        "validation_share": 0.15,
        "patience": 20,
        "max_epochs": 1000,
        **(m["nbm"].get("autoencoder", {}).get("early_stopping") or {}),
    }
    n_val = max(1, int(round(float(es["validation_share"]) * len(z))))
    fit_z, val_z = z[: len(z) - n_val], z[len(z) - n_val :]
    model = MLPRegressor(hidden_layer_sizes=tuple(s.hidden), alpha=alpha, random_state=seed, max_iter=1)
    best, best_epoch, since, state = np.inf, 0, 0, None
    epoch = 0
    for epoch in range(1, int(es["max_epochs"]) + 1):
        model.partial_fit(fit_z, fit_z)
        err = float(np.mean((val_z - model.predict(val_z)) ** 2))
        if err < best:
            best, best_epoch, since = err, epoch, 0
            state = ([w.copy() for w in model.coefs_], [b.copy() for b in model.intercepts_])
        else:
            since += 1
            if since >= int(es["patience"]):
                break
    model.coefs_, model.intercepts_ = state
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "validation_error": best}


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


def f2_train_held(segs: list[Segment], cfg: dict[str, Any]) -> set[int]:
    """F2 in training: post-repair segments whose level is missing or differs from the last accepted level
    by more than the factor in either direction on any plausibility indicator. Their hours are left out of
    training and the segment is held at the last accepted level when scored (F1)."""
    fx = fixes(cfg)
    if not fx["f2_post_repair_window"]:
        return set()
    pl = cfg["models"]["nbm"]["plausibility"]
    lf = float(np.log10(fx["f2_train_factor"] or pl.get("factor") or 1.45))
    held: set[int] = set()
    last: dict[str, float] | None = None
    for k, sg in enumerate(segs):
        if sg.level is None:
            if is_repair(sg.reason) and last is not None:
                held.add(k)
            continue
        if (
            is_repair(sg.reason)
            and last is not None
            and any(
                np.isfinite(sg.plaus.get(c, np.nan))
                and np.isfinite(last.get(c, np.nan))
                and abs(sg.plaus[c] - last[c]) > lf
                for c in PLAUSIBILITY
            )
        ):
            held.add(k)
            continue
        last = sg.plaus
    return held


def segment_plaus(
    rows: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: dict[str, Any],
    cutoff: pd.Timestamp,
    ambient: pd.Series | None,
) -> list[dict[str, float]]:
    """The plausibility levels of a point's training segments (no model needed), for the F2 factor."""
    clean = prepare(rows, ambient)
    win = (
        windows[pd.to_datetime(windows["start"], utc=True) < cutoff]
        .sort_values("start")
        .reset_index(drop=True)
    )
    if clean.empty or win.empty:
        return []
    n = len(clean)
    segs = segment_levels(
        clean, np.zeros((n, 1)), np.full(n, "_", dtype=object), win, {"_all": np.zeros(1)}, cfg
    )
    return [g.plaus for g in segs if g.level is not None]


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
    tf = temperature.fit(
        clean[eligible],
        pclass,
        regime=bool(fixes(cfg)["f3_spindle_temp_regime"]),
        season_min_months=fixes(cfg)["t2_season_min_months"],
        season_clamp=bool(fixes(cfg)["c2_season_clamp"]),
    )
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
        held = f2_train_held(segs0, cfg)
        for k, sg in enumerate(segs0):
            if sg.level is not None and k not in held:
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
            if sg.level is not None and k not in held:
                lev[seg == k] = sg.level
        xc = x - lev
        train = ok & np.isfinite(xc).all(axis=1)
        shared[cand] = (xc, train, offsets, segs, cols, held)
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
        xc, train, offsets, segs, cols, held = shared[s.candidate]
        n = int(train.sum())
        if n < nb["min_train_hours"]:
            info.setdefault("too_few", []).append(s.key)
            continue
        arch = (s.candidate, tuple(s.hidden), int(s.epochs), s.weight_decay, s.early_stopping)
        if arch not in fitted:
            check_training_rows(clean.loc[train, "hour_utc"], cutoff, excluded)
            # base-model decision 2026-10-31: indicators that never vary in the point's training are dropped
            # before fitting (was all seven on every point)
            ind_keep = (
                oh[train].std(axis=0) > 0
                if nb.get("drop_constant_indicators")
                else np.ones(oh.shape[1], dtype=bool)
            )
            raw = np.column_stack([xc[train], oh[train][:, ind_keep]])
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
            fitted[arch] = (sc0, est0, extra0, z0, keep, ind_keep)
        sc, est, extra, z, keep, ind_keep = fitted[arch]
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
            [g.plaus for k, g in enumerate(segs) if g.level is not None and k not in held],
            n,
            sum(g.level is not None and k not in held for k, g in enumerate(segs)),
        )
        pm.indicator_keep = None if ind_keep.all() else ind_keep
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
            **({"f2_held_starts": [str(pd.Timestamp(segs[k].start)) for k in sorted(held)]} if held else {}),
        }
        lim_p = nb.get("input_limit_percentile")
        if lim_p:
            pm.input_limits = input_limits(
                pm, rows, windows, cfg, fit_start, cutoff, excluded, ambient, float(lim_p)
            )
        models[s.key] = pm
    info["n_train"] = max((m.n_train for m in models.values()), default=0)
    return models, info


def input_limits(
    pm: PointModel,
    rows: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: dict[str, Any],
    fit_start: pd.Timestamp,
    cutoff: pd.Timestamp,
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]],
    ambient: pd.Series | None,
    pct: float,
) -> dict[str, float]:
    """Each standardized input's limit: the pct percentile of its values on the point's healthy training
    hours (scored hours from fit_start to the cutoff outside the excluded intervals), high side. Scores the
    training span with the fitted model, as the scoring run does, so the values are those the model sees."""
    s, _ = score_point(pm, rows, windows, cfg, fit_start, cutoff, 1.0, 0, ambient, keep_inputs=True)
    if s.empty:
        return {}
    healthy = (s["state"] == "scored").to_numpy() & training_mask(s["hour_utc"], cutoff, excluded, fit_start)
    out = {}
    for c in pm.columns:
        v = s.loc[healthy, f"z:{c}"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            out[c] = float(np.quantile(v, pct / 100.0))
    return out


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
    keep_inputs: bool = False,
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
    segs = segment_levels(clean, x, keys, win, model.offsets, cfg, s1_after=model.info.get("cutoff"))
    seg = segment_ids(clean["hour_utc"], list(win["start"]))
    t = _naive(clean["hour_utc"])
    cutoff = model.info.get("cutoff")
    lvl = np.full_like(x, np.nan)
    state = np.full(len(clean), "no level", dtype=object)
    watch = np.zeros(len(clean), dtype=bool)
    log_factor = float(np.log10(plaus_factor))
    hist = list(model.train_levels)
    fx = fixes(cfg)
    f1, f2 = bool(fx["f1_hold_level"]), bool(fx["f2_post_repair_window"])
    train_held = {str(pd.Timestamp(s)) for s in (model.info or {}).get("f2_held_starts", [])}
    on_comp = (
        {str(pd.Timestamp(s)): bool(v) for s, v in zip(win["start"], win["on_component"], strict=True)}
        if (fx["f1_scope"] or fx["s1_on_component_window"] or fx["c4_other_repair_window"])
        and "on_component" in win
        else {}
    )
    last_level: np.ndarray | None = None
    last_plaus: dict[str, float] | None = None
    seg_rows = []
    for k, sg in enumerate(segs):
        rows_k = seg == k
        trained = cutoff is not None and sg.start < cutoff
        repair_f2 = f2 and is_repair(sg.reason)
        # C4: repair to another component: previous level while the day 3 to 14 level forms (no two-way check)
        repair_c4 = (
            bool(fx["c4_other_repair_window"])
            and is_repair(sg.reason)
            and str(pd.Timestamp(sg.start)) in on_comp
            and not on_comp[str(pd.Timestamp(sg.start))]
        )
        # S1: no held level after a repair on the measured component (WATCH until its own level forms)
        s1_on = (
            bool(fx["s1_on_component_window"])
            and is_repair(sg.reason)
            and on_comp.get(str(pd.Timestamp(sg.start)), False)
        )
        hold_ok = (f1 or repair_f2) and last_level is not None and not s1_on
        over: list[str] = []
        if sg.level is not None and not trained:
            if f1 and is_remount(sg.reason):
                # F1: a remount accepts its new level without the plausibility check
                sg.flagged = False
            elif repair_f2 and last_plaus is not None:
                # F2: the post-repair level against the last healthy pre-repair level, both directions
                over = [
                    c
                    for c in PLAUSIBILITY
                    if np.isfinite(sg.plaus.get(c, np.nan))
                    and np.isfinite(last_plaus.get(c, np.nan))
                    and abs(sg.plaus[c] - last_plaus[c]) > log_factor
                ]
                sg.flagged = bool(over)
            else:
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
            if sg.flagged and (fx["f1_scope"] or s1_on) and on_comp.get(str(pd.Timestamp(sg.start)), False):
                # F1-scope: the repair replaced the measured component; its new level is accepted
                sg.flagged = False
                over = [f"accepted on component ({','.join(over)})"]
            if not sg.flagged:
                hist.append(sg.plaus)
        elif sg.level is not None and trained and str(pd.Timestamp(sg.start)) in train_held:
            sg.flagged = True  # held in training by the F2 check
        a = _naive([sg.start])[0]
        if sg.level is None:
            if hold_ok:
                # F1: no level could be formed; scored against the last accepted level, on WATCH
                state[rows_k] = "scored"
                lvl[rows_k] = last_level
                watch[rows_k] = True
            else:
                state[rows_k] = "no level"
        else:
            r = _naive([sg.ready])[0]
            pre = rows_k & (t >= a) & (t < r)
            if (repair_f2 or repair_c4) and last_level is not None:
                # F2 and C4: scored against the previous level while the new level forms
                state[pre] = "scored"
                lvl[pre] = last_level
            else:
                state[pre] = "recentering"
            after = rows_k & (t >= r)
            if sg.flagged and hold_ok:
                # F1: flagged level; scored against the last accepted level, on WATCH
                state[after] = "scored"
                lvl[after] = last_level
                watch[after] = True
            else:
                state[after] = "plausibility" if sg.flagged else "scored"
                lvl[after] = sg.level
                if not sg.flagged:
                    last_level, last_plaus = sg.level, sg.plaus
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
                **({"held": bool(hold_ok and (sg.level is None or sg.flagged))} if (f1 or f2) else {}),
                **{f"level_log10_{c}": sg.plaus.get(c, np.nan) for c in PLAUSIBILITY},
            }
        )
    hr = pd.to_datetime(clean["hour_utc"], utc=True)
    span = ((hr >= start) & (hr < end)).to_numpy()
    sc = (state == "scored") & span & np.isfinite(lvl).all(axis=1) & np.isfinite(x).all(axis=1)
    score = np.full(len(clean), np.nan)
    contrib_txt = np.full(len(clean), "", dtype=object)
    limit_txt = np.full(len(clean), "", dtype=object)
    zin = np.full((len(clean), len(model.columns)), np.nan)
    if sc.any():
        raw = np.column_stack([x[sc] - lvl[sc], model.indicators(clean[sc], bands[sc])])
        zs = model.scaler.transform(raw)
        s, contrib = model.raw_scores(zs)
        score[sc] = s
        contrib_txt[sc] = model.contributors(contrib, top_k)
        zin[sc] = zs[:, : len(model.columns)]
        if model.input_limits:
            lim = np.array([model.input_limits.get(c, np.inf) for c in model.columns], dtype=float)
            over = np.nan_to_num(zs[:, : len(model.columns)], nan=-np.inf) > lim
            limit_txt[sc] = [",".join(c for c, o in zip(model.columns, r, strict=True) if o) for r in over]
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
            "limit_inputs": limit_txt,
            **({"watch": watch} if (f1 or f2) else {}),
            **(
                {"alert_blocked": np.full(len(clean), model.n_train < int(fx["t3_min_train_hours"]))}
                if fx["t3_min_train_hours"]
                else {}
            ),
            **({f"z:{c}": zin[:, j] for j, c in enumerate(model.columns)} if keep_inputs else {}),
        }
    )[span]
    pcts = sweep_percentiles(cfg)
    b = out["speed_band"].to_numpy()
    for p in pcts:
        out[f"thr_{p:g}"] = model.threshold_for(b, p)
    out["threshold"] = model.threshold_for(b, nb["threshold_percentile"])
    return out.reset_index(drop=True), pd.DataFrame(seg_rows)


def pelt_mean(x: np.ndarray, penalty: float | None = None, min_size: int = 7) -> list[int]:
    """Change points in the mean of a series by PELT (Killick et al. 2012): Gaussian cost scaled by a robust
    noise variance (median absolute first difference), BIC penalty 2 ln n per change. Returns the indices
    where a new segment starts."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * min_size:
        return []
    d = np.diff(x)
    sig = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.std(x)) or 1.0
    z = x / sig
    s1 = np.r_[0.0, np.cumsum(z)]
    s2 = np.r_[0.0, np.cumsum(z * z)]
    pen = 2.0 * np.log(n) if penalty is None else penalty

    def cost(a: int, b: int) -> float:
        m = b - a
        return float(s2[b] - s2[a] - (s1[b] - s1[a]) ** 2 / m)

    f = np.full(n + 1, np.inf)
    f[0] = -pen
    last = np.zeros(n + 1, dtype=int)
    cands = [0]
    for t_ in range(min_size, n + 1):
        vals = [(f[s] + cost(s, t_) + pen, s) for s in cands if t_ - s >= min_size]
        if not vals:
            continue
        best, arg = min(vals)
        f[t_] = best
        last[t_] = arg
        cands = [s for s in cands if t_ - s < min_size or f[s] + cost(s, t_) <= f[t_]] + [t_ - min_size + 1]
    cps = []
    t_ = n
    while t_ > 0:
        s = last[t_]
        if s > 0:
            cps.append(s)
        t_ = s
    return sorted(cps)


def flat_rearm(
    scores: pd.DataFrame,
    segs: pd.DataFrame,
    conds: pd.DataFrame,
    wo: pd.DataFrame,
    days: int = 30,
    pct: float = 99.5,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """C3 (2026-11-01, switch models.nbm.fixes.c3_flat_rearm; scoring): a point above its threshold for at
    least half its scored hours over `days` consecutive days, with no confirmed condition and no open work
    order on the asset in them, and no upward trend in its daily-median score/threshold (Mann-Kendall
    p >= 0.05 or Theil-Sen slope <= 0), has its threshold reset to the pct percentile of those days' scores
    until its next segment start (repair or re-centering). Returns the scores and the re-arm log."""
    from scipy import stats

    out = scores.copy()
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True)
    log: list[dict[str, Any]] = []
    span = pd.Timedelta(days=days)
    for pid, g in out[out["state"] == "scored"].sort_values("hour_utc").groupby("point_id"):
        asset = str(g["asset_id"].iat[0])
        starts = (
            pd.to_datetime(segs.loc[segs["point_id"] == pid, "segment_start"], utc=True)
            .sort_values()
            .tolist()
        )
        cts = pd.to_datetime(conds.loc[conds["asset_key"] == asset, "credit_ts"], utc=True)
        w = wo[wo["asset_id"].astype(str).str.startswith(asset)]
        wo_iv = list(
            zip(pd.to_datetime(w["opened"], utc=True), pd.to_datetime(w["closed"], utc=True), strict=True)
        )
        thr = g["threshold"].to_numpy(dtype=float).copy()
        hrs = g["hour_utc"].reset_index(drop=True)
        sc = g["score"].to_numpy(dtype=float)
        day = hrs.dt.floor("D")
        days_u = day.drop_duplicates().tolist()
        i = 0
        while i < len(days_u):
            a = days_u[i]
            b = a + span
            m = ((hrs >= a) & (hrs < b)).to_numpy()
            if hrs.iloc[-1] < b - pd.Timedelta(days=1) or not m.any():
                break
            over = sc[m] > thr[m]
            busy = bool(((cts >= a) & (cts < b)).any()) or any(
                (o < b) and (pd.isna(c) or c > a) for o, c in wo_iv
            )
            if over.mean() >= 0.5 and not busy:
                ratio = pd.Series(sc[m] / thr[m], index=hrs[m].to_numpy()).groupby(day[m].to_numpy()).median()
                tt = np.arange(len(ratio), dtype=float)
                tau, p = stats.kendalltau(tt, ratio.to_numpy())
                slope = stats.theilslopes(ratio.to_numpy(), tt)[0]
                if (not np.isfinite(p)) or p >= 0.05 or slope <= 0:
                    new = float(np.percentile(sc[m], pct))
                    nxt = next((s for s in starts if s > b), None)
                    k = ((hrs >= b) & ((hrs < nxt) if nxt is not None else True)).to_numpy()
                    thr[k] = new
                    log.append(
                        {
                            "point_id": pid,
                            "window_start": a,
                            "rearm_from": b,
                            "until": nxt,
                            "new_threshold": new,
                            "share_above": float(over.mean()),
                            "mk_p": float(p),
                            "theil_sen_slope": float(slope),
                        }
                    )
                    j = next((n for n, d in enumerate(days_u) if d >= b), len(days_u))
                    i = j
                    continue
            i += 1
        scale = thr / g["threshold"].to_numpy(dtype=float)
        for col in [c for c in out.columns if c.startswith("thr_")]:
            out.loc[g.index, col] = g[col].to_numpy(dtype=float) * scale  # the sweep moves with the re-arm
        out.loc[g.index, "threshold"] = thr
    return out, log


def _runs(over: np.ndarray) -> np.ndarray:
    run = np.zeros(len(over), dtype=int)
    c = 0
    for i, o in enumerate(over):
        c = c + 1 if o else 0
        run[i] = c
    return run


EXCEED_COLS = ["point_id", "ts", "ratio", "top_contributors", "source"]


def exceedance_hours(
    scores: pd.DataFrame,
    threshold_col: str,
    persistence: int,
    high_ratio: tuple[float, int] | None = None,
    exclude_low_pressure: bool = False,
) -> pd.DataFrame:
    """Hours at which the score has exceeded the threshold column on persistence consecutive scored hours,
    or (fold diagnosis round 2 decision, 2026-10-31) any single standardized input has exceeded its own
    limit on persistence consecutive scored hours (column limit_inputs): the persisted exceedances that
    ALERT episodes are built from. source names what persisted at each hour: 'pca' and 'limit:<input>'.
    (Was the score alone.)"""
    rows = []
    s = scores[scores["state"] == "scored"] if "state" in scores else scores
    for pid, g in s.sort_values("hour_utc").groupby("point_id"):
        pca = _runs((g["score"] > g[threshold_col]).to_numpy()) >= persistence
        hit = pca.copy()
        src = [["pca"] if p else [] for p in pca]
        if high_ratio is not None:
            # F4 (2026-11-01, test only; switch models.nbm.fixes.f4_high_ratio): a score at high_ratio[0]
            # times the threshold or more on high_ratio[1] consecutive scored hours counts without the
            # full persistence
            high = (g["score"] >= high_ratio[0] * g[threshold_col]).to_numpy()
            if exclude_low_pressure and point_kind(pid) == "pump" and "low_pressure" in g:
                # F4-tight: hours on pump points during a controller low-pressure alarm (column
                # low_pressure) do not count toward the high-ratio run
                high &= ~g["low_pressure"].fillna(False).astype(bool).to_numpy()
            hr = _runs(high) >= int(high_ratio[1])
            hit |= hr
            for i in np.flatnonzero(hr):
                src[i].append("high_ratio")
        if "limit_inputs" in g and g["limit_inputs"].fillna("").astype(str).str.len().any():
            lists = [set(v.split(",")) if v else set() for v in g["limit_inputs"].fillna("").astype(str)]
            for name in sorted(set().union(*lists)):
                m = _runs(np.array([name in x for x in lists])) >= persistence
                hit |= m
                for i in np.flatnonzero(m):
                    src[i].append(f"limit:{name}")
        if hit.any():
            rows.append(
                pd.DataFrame(
                    {
                        "point_id": pid,
                        "ts": g["hour_utc"].to_numpy()[hit],
                        "ratio": (g["score"] / g[threshold_col]).to_numpy()[hit],
                        "top_contributors": g["top_contributors"].to_numpy()[hit],
                        "source": [",".join(src[i]) for i in np.flatnonzero(hit)],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=EXCEED_COLS)

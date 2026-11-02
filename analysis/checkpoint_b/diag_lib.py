"""Scratch diagnosis library (checkpoint (b) diagnosis; nothing here is committed).

Loads the local reference, the run09 truth archive and the frozen model; rebuilds each point's model inputs
(scaled inputs z, T2 and SPE terms and per-input contributions); replays the Vendor Rules with their per-hour
alert levels; marks truth-healthy hours."""

from __future__ import annotations

import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dlt import read  # noqa: E402

from pdm import splits, tasks  # noqa: E402
from pdm.config import load_config  # noqa: E402
from pdm.models import nbm, severity, temperature  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TRUTH = REPO / "runs/run09_20261030T090522Z_3eced8fb/truth"
CFG = load_config()
B = splits.boundaries(CFG)
TZ = "America/Chicago"
DAY = pd.Timedelta(days=1)


def utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


@lru_cache(maxsize=1)
def model():
    m = pickle.load(open(REPO / "data/local_gold/ml_model.pkl", "rb"))["model"]  # noqa: S301
    if m.healthy_stats is None:
        m.healthy_stats = read("gold/point_healthy_stats")
    return m


@lru_cache(maxsize=1)
def ind() -> pd.DataFrame:
    d = read("silver/hourly_indicators")
    d["hour_utc"] = utc(d["hour_utc"])
    return d


@lru_cache(maxsize=1)
def windows() -> pd.DataFrame:
    w = read("silver/healthy_windows")
    w["start"] = utc(w["start"])
    return w


@lru_cache(maxsize=1)
def ambient() -> pd.Series:
    r = read("bronze/cmp_room", ["minute_utc", "ambient_c"])
    r["minute_utc"] = utc(r["minute_utc"])
    return temperature.hourly_ambient(r)


@lru_cache(maxsize=1)
def points() -> pd.DataFrame:
    return read("silver/points")


@lru_cache(maxsize=1)
def wo() -> pd.DataFrame:
    w = read("bronze/cmms_work_orders")
    for c in ("opened", "started", "closed"):
        w[c] = utc(w[c])
    return w


@lru_cache(maxsize=1)
def conds() -> pd.DataFrame:
    """Confirmed conditions as the evaluation uses them (credit_ts, asset_key, instrumented)."""
    c = read("silver/confirmed_conditions")
    c["confirmed_ts"] = utc(c["confirmed_ts"])
    w = wo().set_index("wo_id")
    c["credit_ts"] = c["confirmed_ts"].where(~c["pm_finding"].astype(bool), c["wo_id"].map(w["started"])).fillna(c["confirmed_ts"])
    c["instrumented"] = c["asset_id"].str.match(tasks.INSTRUMENTED)
    c["asset_key"] = np.where(c["instrumented"], c["asset_id"], c["asset_id"].str.extract(r"^([A-Z]+-\d+)")[0])
    return c


@lru_cache(maxsize=1)
def truth_hours() -> pd.DataFrame:
    cols = ["point_id", "hour_utc", "instance_id", "stage", "hi", "defect", "onset_cause", "lube_severity", "cavitation",
            "wear_frac", "measured", "fault_kind", "fault_level", "ingress_level", "temp_c"]
    t = pd.read_parquet(TRUTH / "point_hours.parquet", columns=cols)
    t["hour_utc"] = utc(t["hour_utc"])
    t["truth_healthy"] = (
        (t["stage"].fillna(0) == 0)
        & (t["lube_severity"].fillna(0) <= 0)
        & ~t["cavitation"].fillna(False).astype(bool)
        & (t["wear_frac"].fillna(0) <= 0)
        & (t["fault_level"].fillna(0) <= 0)
        & (t["ingress_level"].fillna(0) <= 0)
    )
    return t


@lru_cache(maxsize=1)
def events() -> pd.DataFrame:
    e = pd.read_parquet(TRUTH / "events.parquet")
    e["ts"] = utc(e["ts"])
    return e


def decompose(pm: nbm.PointModel, rows: pd.DataFrame, win: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, amb=None) -> pd.DataFrame:
    """Per scored hour: state, score, threshold, T2 and SPE terms (normalized as in the score), the scaled
    inputs z and each input's contribution to T2 and to SPE."""
    clean = nbm.prepare(rows, amb)
    clean[nbm.TEMP_RESID] = pm.temp_fit.residual(clean)
    w = win.sort_values("start").reset_index(drop=True)
    w = w[w["start"] < end].reset_index(drop=True)
    bands = nbm.speed_band(clean, pm.rated_rpm, CFG)
    keys = nbm.operating_key(clean, bands)
    x, cols = nbm.features(clean, pm.settings.candidate)
    segs = nbm.segment_levels(clean, x, keys, w, pm.offsets, CFG)
    seg = nbm.segment_ids(clean["hour_utc"], list(w["start"]))
    t = np.asarray(clean["hour_utc"].dt.tz_localize(None), dtype="datetime64[ns]")
    lvl = np.full_like(x, np.nan)
    for k, sg in enumerate(segs):
        if sg.level is not None:
            r = np.datetime64(pd.Timestamp(sg.ready).tz_convert(None))
            m = (seg == k) & (t >= r)
            lvl[m] = sg.level
    hr = clean["hour_utc"]
    span = ((hr >= start) & (hr < end)).to_numpy() & np.isfinite(lvl).all(axis=1) & np.isfinite(x).all(axis=1)
    if not span.any():
        return pd.DataFrame()
    raw = np.column_stack([x[span] - lvl[span], nbm._one_hot(clean[span], bands[span])])
    z = pm.scaler.transform(raw)
    p = pm.model
    tt = p.transform(z)
    lam = p.explained_variance_
    t2 = np.sum(tt**2 / lam, axis=1)
    rec = p.inverse_transform(tt)
    spe = np.sum((z - rec) ** 2, axis=1)
    # T2 contribution per input: z_j * sum_k p_jk t_k / lambda_k (sums to T2 when PCA is centred at zero mean)
    zc = z - p.mean_
    t2c = zc * ((tt / lam) @ p.components_)
    spec = (z - rec) ** 2
    names = cols + [f"regime_{r}" for r in nbm.REGIMES] + [f"band_{b}" for b in nbm.BANDS]
    out = pd.DataFrame({"point_id": pm.point_id, "hour_utc": hr[span].to_numpy(), "speed_band": bands[span],
                        "regime": clean["regime"].to_numpy()[span],
                        "t2_term": t2 / pm.extra["t2_med"], "spe_term": spe / pm.extra["spe_med"]})
    out["score"] = out["t2_term"] + out["spe_term"]
    out["threshold"] = pm.threshold_for(bands[span], CFG["models"]["nbm"]["threshold_percentile"])
    zdf = pd.DataFrame(z, columns=[f"z:{n}" for n in names])
    c2 = pd.DataFrame(t2c / pm.extra["t2_med"], columns=[f"t2:{n}" for n in names])
    cs = pd.DataFrame(spec / pm.extra["spe_med"], columns=[f"spe:{n}" for n in names])
    return pd.concat([out.reset_index(drop=True), zdf, c2, cs], axis=1)


GROUPS = {
    "third_octave": lambda n: n.startswith("toct_"),
    "envelope": lambda n: n.startswith("env_"),
    "hf": lambda n: n == "hf_rms_g",
    "temp_resid": lambda n: n == nbm.TEMP_RESID,
    "regime_band": lambda n: n.startswith("regime_") or n.startswith("band_"),
}


def group_of(name: str) -> str:
    for g, f in GROUPS.items():
        if f(name):
            return g
    return "other"


@lru_cache(maxsize=1)
def scalars() -> pd.DataFrame:
    d = read("bronze/cm_scalars")
    d["hour_utc"] = utc(d["hour_utc"])
    return d


def vendor_hourly(point_id: str) -> pd.DataFrame:
    """Hourly Vendor Rules replay for one point with the alert levels in force."""
    from dataclasses import replace

    from pdm.rules import vendor_baseline as vb
    from pdm.rules import vendor_replay as vr

    pclass = str(points().set_index("point_id").at[point_id, "point_class"])
    settings = replace(vb.VendorRuleSettings.for_point(CFG, pclass), suppression_days=0.0)
    rows = scalars()
    rows = rows[rows["point_id"] == point_id].sort_values("hour_utc")
    se = read("bronze/cm_sensor_events", ["point_id", "ts", "event"])
    se["ts"] = utc(se["ts"])
    rec: list[dict] = []
    orig = vb.VendorRuleState.update

    def update(self, ts, row):  # noqa: ANN001
        if self.baseline is not None and row is not None:
            th = self.thresholds()
            ratio = 10 ** (self.settings.env_db / 10.0)
            envr = max(
                (row[c] / (ratio * self.baseline[c]) for c in self.settings.env_columns() if self.baseline.get(c, 0) > 0 and pd.notna(row.get(c))),
                default=np.nan,
            )
            rec.append({"hour_utc": ts, "vel_ratio": row["vel_rms_mm_s"] / th["vel_alert"], "hf_ratio": row["hf_rms_g"] / th["hf_alert"],
                        "env_ratio": envr, "temp_c": row.get("temp_c"), "temp_warning_c": self.settings.temp_warning_c})
        return orig(self, ts, row)

    vb.VendorRuleState.update = update
    try:
        vr.replay_point(point_id, rows, se, settings, CFG)
    finally:
        vb.VendorRuleState.update = orig
    out = pd.DataFrame(rec)
    out["point_id"] = point_id
    out["temp_ratio"] = out["temp_c"].astype(float) / out["temp_warning_c"]
    return out


def asset_points(asset: str) -> list[str]:
    p = points()
    return sorted(p.loc[p["asset_id"] == asset, "point_id"])

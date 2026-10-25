"""Champion and challenger promotion for the per-point models (Unity Catalog model
<catalog>.ml.nbm_point_models).

The train task logs challenger runs only. The promote task registers the challenger run's model as a new
version (or reuses the version already registered from that run, when the task is repaired), evaluates
the gates (selection rule, parity of the registered model's scores with the train task's own scores,
segment outcomes, the challenger's false alert episodes in the latest complete month) and moves the
champion alias only when every gate passes and auto_promote is on. Every gate result is a tag on the
version, with keys free of Unity Catalog's reserved characters; a rejected version leaves the champion
where it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pdm import log, tasks
from pdm.evaluation import events as ev
from pdm.models import nbm

LOG = log.get_logger("pdm.registry")
MODEL = "nbm_point_models"
CHAMPION = "champion"
FITTED = ("segment fit", "window refit")
HOURS_PER_MONTH = 365.25 / 12 * 24
REL_TOL = 1e-9
ABS_FLOOR = 1e-12
RESERVED_TAG_CHARACTERS = frozenset(".=><%&?\\")
IDENTITY_TAGS = ("source_run_id", "candidate", "job_run_id", "model_version_hash", "pdm_version")


def tag_key(key: str) -> str:
    """A model version tag key Unity Catalog accepts: no reserved characters."""
    bad = sorted(set(key) & RESERVED_TAG_CHARACTERS)
    if not key or bad:
        raise ValueError(f"tag key {key!r} contains reserved characters {bad}")
    return key


def model_name(catalog: str) -> str:
    return f"{catalog}.ml.{MODEL}"


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str
    tags: dict[str, str] = field(default_factory=dict)


def gate_selection(run_tags: dict[str, str], candidates: pd.DataFrame) -> Gate:
    """The version's run is the selection rule's pick for the same training."""
    mv = run_tags.get("model_version")
    c = candidates[candidates["model_version"] == mv]
    picked = c.loc[c["champion"].astype(bool), "candidate"].tolist()
    ok = run_tags.get("challenger") == "true" and picked == [run_tags.get("candidate")]
    return Gate(
        "selection", ok, f"run candidate {run_tags.get('candidate')}, rule pick {picked}, training {mv}"
    )


def _close(a: pd.Series, b: pd.Series) -> np.ndarray:
    x, y = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    tol = np.maximum(REL_TOL * np.maximum(np.abs(x), np.abs(y)), ABS_FLOOR)
    return (np.abs(x - y) <= tol) | (np.isnan(x) & np.isnan(y))


def gate_parity(pred: pd.DataFrame, train: pd.DataFrame) -> Gate:
    """The registered model scores the same hours as the train task, every score and threshold within
    1e-9 relative (1e-12 absolute near zero), with the same segment, update, regime model and contributors."""
    key = ["point_id", "hour_utc"]
    p, t = pred.copy(), train.copy()
    for d in (p, t):
        d["hour_utc"] = pd.to_datetime(d["hour_utc"], utc=True)
    m = p.merge(t, on=key, how="outer", suffixes=("_reg", "_train"), indicator=True)
    only = (m["_merge"] != "both").sum()
    m = m[m["_merge"] == "both"]
    bad: dict[str, int] = {}
    for c in [c for c in t.columns if c == "score" or c.startswith("thr_")]:
        n = int((~_close(m[f"{c}_reg"], m[f"{c}_train"])).sum())
        if n:
            bad[c] = n
    for c in ("model_segment", "model_regime", "model_update", "top_contributors"):
        if c in t.columns:
            n = int((m[f"{c}_reg"].astype(str).to_numpy() != m[f"{c}_train"].astype(str).to_numpy()).sum())
            if n:
                bad[c] = n
    ok = only == 0 and not bad and len(m) == len(t)
    return Gate("parity", ok, f"hours {len(t)}, unmatched {int(only)}, differing {bad or 'none'}")


def gate_segments(model: Any, models_info: pd.DataFrame, train: pd.DataFrame) -> Gate:
    """Every segment the train task fitted is in the model with a version and has scores; no scored hour
    without a score; healthy stats present."""
    fitted = models_info[models_info["model"].isin(FITTED)]
    missing_model, missing_scores = [], []
    scored = set(zip(train["point_id"], train["model_segment"], strict=True))
    for pid, seg in zip(fitted["point_id"], fitted["segment"].astype(int), strict=True):
        segs = model.segments.get(pid, [])
        if not any(int(s["segment"]) == seg and s.get("versions") for s in segs):
            missing_model.append(f"{pid}/{seg}")
        if (pid, f"segment {seg}") not in scored:
            missing_scores.append(f"{pid}/{seg}")
    nulls = int(train["score"].isna().sum())
    hs = getattr(model, "healthy_stats", None)
    ok = not missing_model and not missing_scores and nulls == 0 and hs is not None and len(hs) > 0
    return Gate(
        "segments",
        ok,
        f"fitted {len(fitted)}, missing in model {missing_model or 'none'}, without scores "
        f"{missing_scores or 'none'}, null scores {nulls}, healthy stats rows {0 if hs is None else len(hs)}",
    )


def latest_complete_month(horizon: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The last local calendar month that ends at or before the horizon, in UTC."""
    h = horizon.tz_convert(tasks.LOCAL_TZ).tz_localize(None)
    end = h.normalize().replace(day=1)
    start = end - pd.offsets.MonthBegin(1)
    return (
        start.tz_localize(tasks.LOCAL_TZ).tz_convert("UTC"),
        end.tz_localize(tasks.LOCAL_TZ).tz_convert("UTC"),
    )


def healthy_hours(
    assets: list[str],
    month: tuple[pd.Timestamp, pd.Timestamp],
    conds: pd.DataFrame,
    wo: pd.DataFrame,
    window_days: float,
) -> dict[str, np.ndarray]:
    """Per asset, the hours of the month outside the catch window before a confirmed condition and
    outside open inspection or repair work (hourly grid, boolean)."""
    grid = pd.date_range(month[0], month[1], freq="h", inclusive="left")
    w = pd.Timedelta(days=window_days)
    out = {}
    for a in assets:
        ok = np.ones(len(grid), dtype=bool)
        for t in conds.loc[conds["asset_key"] == a, "credit_ts"]:
            ok &= ~np.asarray((grid >= t - w) & (grid < t))
        ww = wo[wo["asset_id"].str.startswith(a) & wo["wo_type"].isin(tasks.INSPECTION_OR_REPAIR)]
        for o, c in zip(ww["opened"], ww["closed"], strict=True):
            ok &= ~np.asarray((grid >= o) & (grid < (c if pd.notna(c) else month[1])))
        out[a] = ok
    return out


def gate_false_episodes(
    train: pd.DataFrame,
    pts: pd.DataFrame,
    conds: pd.DataFrame,
    wo: pd.DataFrame,
    curves: pd.DataFrame,
    cfg: dict[str, Any],
    horizon: pd.Timestamp,
) -> Gate:
    """The challenger's false alert episodes per healthy asset-month at the operating point in the latest
    complete month before the horizon, against the per-point models' validation rate times the monitoring
    tolerance."""
    nb = cfg["models"]["nbm"]
    op = float(nb["threshold_percentile"])
    catch = cfg["evaluation"]["catch"]
    ratio = float(cfg["monitoring"]["tolerance"]["alerts_per_asset_month_ratio"][1])
    month = latest_complete_month(horizon)
    conds = conds[conds["credit_ts"] < horizon]
    wo = wo[wo["opened"] < horizon].copy()
    wo["closed"] = wo["closed"].where(wo["closed"] < horizon)
    s = train.copy()
    s["hour_utc"] = pd.to_datetime(s["hour_utc"], utc=True)
    ex = nbm.exceedance_hours(s, op, nb["persistence_hours"])
    ex = ex.merge(pts[["point_id", "asset_id"]], on="point_id") if len(ex) else ex
    ep = ev.episodes(
        ex[["asset_id", "point_id", "ts"]] if len(ex) else pd.DataFrame(), catch["episode_gap_days"]
    )
    ep = ep[(ep["opened"] >= month[0]) & (ep["opened"] < month[1])].reset_index(drop=True)
    assets = sorted(pts["asset_id"].unique())
    hh = healthy_hours(assets, month, conds, wo, catch["opened_within_days"])
    grid = pd.date_range(month[0], month[1], freq="h", inclusive="left")
    if len(ep):
        pos = np.clip(grid.searchsorted(ep["opened"].dt.floor("h")), 0, len(grid) - 1)
        healthy_open = np.array([hh[a][i] for a, i in zip(ep["asset_id"], pos, strict=True)], dtype=bool)
        false = ev.false_episodes(ep, conds).to_numpy() & healthy_open
    else:
        false = np.zeros(0, dtype=bool)
    asset_months = float(sum(v.sum() for v in hh.values())) / HOURS_PER_MONTH
    n_false = int(false.sum())
    rate = n_false / asset_months if asset_months > 0 else float("inf")
    v = curves[(curves["method"] == "nbm") & (curves["level"].astype(float) == op)]["false_per_asset_month"]
    if v.empty:
        return Gate(
            "false_episodes", False, "no validation rate for the per-point models at the operating point"
        )
    budget = float(v.iat[0]) * ratio
    label = f"{month[0].tz_convert(tasks.LOCAL_TZ):%Y-%m}"
    tags = {
        "false_episodes_month": label,
        "false_episodes_healthy_asset_months": f"{asset_months:.3f}",
        "false_episodes_episodes": str(len(ep)),
        "false_episodes_false_episodes": str(n_false),
        "false_episodes_rate": f"{rate:.4f}",
        "false_episodes_budget": f"{budget:.4f}",
    }
    return Gate(
        "false_episodes",
        rate <= budget,
        f"{label}: {n_false} false of {len(ep)} episodes over {asset_months:.2f} healthy asset-months, "
        f"rate {rate:.4f} against budget {budget:.4f} ({float(v.iat[0]):.4f} x {ratio:g})",
        tags,
    )


class Registry(Protocol):
    def find_challenger(self, experiment: str, job_run_id: str) -> str: ...
    def run_tags(self, run_id: str) -> dict[str, str]: ...
    def register(self, run_id: str, name: str) -> str: ...
    def find_version(self, name: str, run_id: str) -> str | None: ...
    def load(self, name: str, version: str) -> Any: ...
    def set_tag(self, name: str, version: str, key: str, value: str) -> None: ...
    def delete_tag(self, name: str, version: str, key: str) -> None: ...
    def set_alias(self, name: str, alias: str, version: str) -> None: ...
    def alias_version(self, name: str, alias: str) -> str | None: ...


class MlflowRegistry:
    """The Unity Catalog model registry and the workspace tracking server."""

    def __init__(self) -> None:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_registry_uri("databricks-uc")
        self.mlflow = mlflow
        self.client = MlflowClient(registry_uri="databricks-uc")

    def find_challenger(self, experiment: str, job_run_id: str) -> str:
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            raise RuntimeError(f"experiment {experiment} not found")
        runs = self.client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.challenger = 'true' and tags.`pdm.job_run_id` = '{job_run_id}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(f"no challenger run in {experiment} for job run {job_run_id}")
        return runs[0].info.run_id

    def run_tags(self, run_id: str) -> dict[str, str]:
        return dict(self.client.get_run(run_id).data.tags)

    def register(self, run_id: str, name: str) -> str:
        return str(self.mlflow.register_model(f"runs:/{run_id}/model", name).version)

    def find_version(self, name: str, run_id: str) -> str | None:
        try:
            found = [
                int(v.version)
                for v in self.client.search_model_versions(f"name='{name}'")
                if v.run_id == run_id
            ]
        except Exception:  # noqa: BLE001
            return None
        return str(min(found)) if found else None

    def load(self, name: str, version: str) -> Any:
        return self.mlflow.pyfunc.load_model(f"models:/{name}/{version}").unwrap_python_model()

    def set_tag(self, name: str, version: str, key: str, value: str) -> None:
        self.client.set_model_version_tag(name, version, key, value)

    def delete_tag(self, name: str, version: str, key: str) -> None:
        try:
            self.client.delete_model_version_tag(name, version, key)
        except Exception:  # noqa: BLE001
            return

    def set_alias(self, name: str, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(name, alias, version)

    def alias_version(self, name: str, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(name, alias).version)
        except Exception:  # noqa: BLE001
            return None


def evaluate_gates(
    model: Any,
    run_tags: dict[str, str],
    tables: dict[str, pd.DataFrame],
    ind: pd.DataFrame,
    cfg: dict[str, Any],
    horizon: pd.Timestamp,
) -> list[Gate]:
    train = tables["nbm_train_scores"]
    mv = run_tags.get("model_version")
    info = tables["nbm_models"]
    info = info[info["model_version"] == mv] if "model_version" in info else info
    pred = model.score_frame(ind)
    return [
        gate_selection(run_tags, tables["nbm_candidates"]),
        gate_parity(pred, train.drop(columns=["asset_id", "model_version", "candidate"], errors="ignore")),
        gate_segments(model, info, train),
        gate_false_episodes(
            train,
            tables["points"],
            tables["conditions"],
            tables["work_orders"],
            tables["curves"],
            cfg,
            horizon,
        ),
    ]


def decide(registry: Registry, name: str, version: str, gates: list[Gate], auto_promote: bool) -> str:
    """Tag the version with every gate and move the champion alias when all pass and auto_promote is on."""
    failed = [g.name for g in gates if not g.passed]
    for g in gates:
        registry.set_tag(name, version, tag_key(f"gate_{g.name}"), "pass" if g.passed else "fail")
        registry.set_tag(name, version, tag_key(f"gate_{g.name}_detail"), g.detail[:4900])
        for k, v in g.tags.items():
            registry.set_tag(name, version, tag_key(k), v)
    if failed:
        registry.set_tag(name, version, tag_key("failed_gates"), ",".join(failed))
    else:
        registry.delete_tag(name, version, "failed_gates")
    if not auto_promote:
        status = "for_review"
    elif failed:
        status = "rejected"
    else:
        registry.set_alias(name, CHAMPION, version)
        status = "promoted"
    registry.set_tag(name, version, tag_key("status"), status)
    return status


def register_or_reuse(registry: Registry, run_id: str, name: str) -> tuple[str, bool]:
    """The version registered from this run, if one exists (a repaired promote task), otherwise a new one."""
    found = registry.find_version(name, run_id)
    if found is not None:
        return found, True
    return registry.register(run_id, name), False


def promote(
    store: Any,
    cfg: dict[str, Any],
    catalog: str,
    experiment_root: str,
    job_run_id: str | None = None,
    run_id: str | None = None,
    auto_promote: bool = True,
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Register the challenger, evaluate the gates and move the champion alias when allowed."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        pdm_version = pkg_version("pdm")
    except PackageNotFoundError:
        pdm_version = ""
    reg = registry or MlflowRegistry()
    if run_id is None:
        if not job_run_id:
            raise ValueError("either a run id or the job run id is required")
        run_id = reg.find_challenger(f"{experiment_root.rstrip('/')}/nbm", job_run_id)
    tags = reg.run_tags(run_id)
    name = model_name(catalog)
    version, reused = register_or_reuse(reg, run_id, name)
    LOG.info("%s version %s from run %s", "reusing" if reused else "registered", version, run_id)
    identity = (
        run_id,
        tags.get("candidate", ""),
        str(job_run_id or ""),
        tags.get("model_version", ""),
        pdm_version,
    )
    for k, v in zip(IDENTITY_TAGS, identity, strict=True):
        reg.set_tag(name, version, tag_key(k), v)
    model = reg.load(name, version)
    end = tasks.horizon(cfg)
    ind = store.read_pd("silver", "hourly_indicators")
    ind["hour_utc"] = tasks._utc(ind["hour_utc"])
    ind = ind[ind["hour_utc"] < end]
    tables = {
        "nbm_train_scores": store.read_pd("gold", "nbm_train_scores"),
        "nbm_models": store.read_pd("gold", "nbm_models"),
        "nbm_candidates": store.read_pd("gold", "nbm_candidates"),
        "curves": store.read_pd("gold", "evaluation_curves_validation")
        if store.exists("gold", "evaluation_curves_validation")
        else pd.DataFrame(columns=["method", "level", "false_per_asset_month"]),
        "points": tasks._points(store),
        "conditions": tasks._conditions(store, cfg),
        "work_orders": tasks._wo(store),
    }
    tables["nbm_train_scores"]["hour_utc"] = tasks._utc(tables["nbm_train_scores"]["hour_utc"])
    gates = evaluate_gates(model, tags, tables, ind, cfg, end)
    for g in gates:
        LOG.info("gate %s %s: %s", g.name, "pass" if g.passed else "fail", g.detail)
    status = decide(reg, name, version, gates, auto_promote)
    champion = reg.alias_version(name, CHAMPION)
    LOG.info("version %s %s; champion is version %s", version, status, champion)
    if champion is None:
        raise RuntimeError(f"{name} has no {CHAMPION} version; version {version} is {status}")
    return {
        "run_id": run_id,
        "version": version,
        "reused": reused,
        "status": status,
        "champion": champion,
        "gates": {g.name: g.passed for g in gates},
    }

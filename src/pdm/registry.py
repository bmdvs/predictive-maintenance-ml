"""Registration and promotion of the frozen ML Model (Unity Catalog model <catalog>.ml.nbm_point_models).

The training run logs the champion's run. The promote task registers that run's model as a new version (or
reuses the version already registered from that run, when the task is repaired), evaluates the gates once
(the selection rule's pick on both folds combined, parity of the registered model's training scores with
the training run's, every point's model and re-centering levels present, and the false-episode rate on the
last six months of training within twice the fold rate) and moves the champion alias when every gate passes
and auto_promote is on. Every gate result is a tag on the version, with keys free of Unity Catalog's
reserved characters; a rejected version leaves the champion where it is. (Revision 2: gates once on a
frozen model; originally a monthly champion and challenger cycle.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pdm import log, tasks

LOG = log.get_logger("pdm.registry")
MODEL = "nbm_point_models"
CHAMPION = "champion"
FITTED = ("segment fit", "window refit")
HOURS_PER_MONTH = 365.25 / 12 * 24
REL_TOL = 1e-9
ABS_FLOOR = 1e-12
RESERVED_TAG_CHARACTERS = frozenset(".=><%&?\\")
IDENTITY_TAGS = ("source_run_id", "settings", "job_run_id", "model_version_hash", "pdm_version")


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
    """The version's run is the selection rule's pick on both folds combined, for the same training."""
    mv = run_tags.get("model_version")
    c = candidates[candidates["model_version"] == mv] if "model_version" in candidates else candidates
    picked = c.loc[c["champion"].astype(bool), "settings"].tolist()
    ok = run_tags.get("challenger") == "true" and picked == [run_tags.get("settings")]
    return Gate(
        "selection", ok, f"run settings {run_tags.get('settings')}, rule pick {picked}, training {mv}"
    )


def _close(a: pd.Series, b: pd.Series) -> np.ndarray:
    x, y = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    tol = np.maximum(REL_TOL * np.maximum(np.abs(x), np.abs(y)), ABS_FLOOR)
    return (np.abs(x - y) <= tol) | (np.isnan(x) & np.isnan(y))


def gate_parity(pred: pd.DataFrame, train: pd.DataFrame) -> Gate:
    """The registered model scores the training hours as the training run did: the same hours and states,
    every score and threshold within 1e-9 relative (1e-12 absolute near zero)."""
    key = ["point_id", "hour_utc"]
    p, t = pred.copy(), train.copy()
    for d in (p, t):
        d["hour_utc"] = pd.to_datetime(d["hour_utc"], utc=True)
    m = p.merge(t, on=key, how="outer", suffixes=("_reg", "_train"), indicator=True)
    only = (m["_merge"] != "both").sum()
    m = m[m["_merge"] == "both"]
    bad: dict[str, int] = {}
    for c in [c for c in t.columns if c in ("score", "threshold", "temp_resid_c") or c.startswith("thr_")]:
        if f"{c}_reg" in m:
            n = int((~_close(m[f"{c}_reg"], m[f"{c}_train"])).sum())
            if n:
                bad[c] = n
    if "state_reg" in m:
        n = int((m["state_reg"].astype(str).to_numpy() != m["state_train"].astype(str).to_numpy()).sum())
        if n:
            bad["state"] = n
    ok = only == 0 and not bad and len(m) == len(t)
    return Gate("parity", ok, f"hours {len(t)}, unmatched {int(only)}, differing {bad or 'none'}")


def gate_points(model: Any, points: pd.DataFrame, train_segments: pd.DataFrame) -> Gate:
    """Every sensor point has its model, and every trained segment with enough measured hours has its
    re-centering level (replacing 'every fitted segment present')."""
    have = set(getattr(model, "models", {}))
    missing = sorted(set(points["point_id"]) - have)
    seg = train_segments[train_segments["trained"].astype(bool)] if len(train_segments) else train_segments
    no_level = seg[seg["level_ready"].isna()] if len(seg) else seg
    hs = getattr(model, "healthy_stats", None)
    ok = not missing and hs is not None and len(hs) > 0
    return Gate(
        "points",
        ok,
        f"points {len(points)}, models {len(have)}, missing {missing or 'none'}; "
        f"trained segments {len(seg)}, "
        f"without a level (too few hours, left to the Vendor Rules) {len(no_level)}; healthy stats rows "
        f"{0 if hs is None else len(hs)}",
    )


NUMERIC_FLOOR = 1e-6


def gate_numerics(model: Any, floor: float = NUMERIC_FLOOR) -> Gate:
    """Every point's score rests on well-conditioned numbers (parity failure decisions, 2026-10-31): each
    input's scale, and the T2 and (where used) SPE training medians that normalize the score, at or above
    the floor (1e-6, in the input's own units for the scales). Fails any point below it."""
    bad = []
    for pid, pm in sorted(getattr(model, "models", {}).items()):
        scale = np.asarray(getattr(pm.scaler, "scale", []), dtype=float)
        if scale.size and float(scale.min()) < floor:
            bad.append(f"{pid} input scale {float(scale.min()):.3g}")
        extra = getattr(pm, "extra", {}) or {}
        for k in ("t2_med", "spe_med"):
            v = extra.get(k)
            if v is not None and float(v) < floor:
                bad.append(f"{pid} {k} {float(v):.3g}")
    return Gate("numerics", not bad, f"floor {floor:g}; points below it: {', '.join(bad) or 'none'}")


def gate_false_episodes(settings: pd.DataFrame) -> Gate:
    """False episodes per asset-month on the last six months of training within twice the fold rate."""
    r = settings.iloc[0]
    rate, fold = float(r["reference_false_per_asset_month"]), float(r["fold_false_per_asset_month"])
    limit = 2.0 * fold
    tags = {
        "false_episodes_window": str(r["reference_window"]),
        "false_episodes_rate": f"{rate:.4f}",
        "false_episodes_fold_rate": f"{fold:.4f}",
        "false_episodes_limit": f"{limit:.4f}",
    }
    return Gate(
        "false_episodes",
        bool(rate <= limit),
        f"{r['reference_window']}: {rate:.4f} false episodes per asset-month against twice the fold rate "
        f"{limit:.4f} ({fold:.4f} x 2)",
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
    pred: pd.DataFrame,
) -> list[Gate]:
    """The gates, run once on the frozen champion (revision 2): selection, parity with the training run's
    scores, every point's model and re-centering levels, the false-episode rate on the last six months of
    training within twice the fold rate, and (2026-10-31) the numerics floor on scales and normalizers."""
    train = tables["ml_train_scores"]
    keep = [c for c in train.columns if c not in ("asset_id", "model_version", "settings")]
    return [
        gate_selection(run_tags, tables["ml_candidates"]),
        gate_parity(pred, train[keep]),
        gate_points(model, tables["points"], tables["ml_train_segments"]),
        gate_false_episodes(tables["ml_settings"]),
        gate_numerics(model),
    ]


def training_prediction(model: Any, store: Any, cfg: dict[str, Any]) -> pd.DataFrame:
    """The model's scores on the training span, in the training run's format, for the parity gate."""
    from pdm import ml_tasks, splits

    b = splits.boundaries(cfg)
    cutoff = splits.training_cutoff(cfg)
    inp = ml_tasks.load_inputs(store, cfg, b["train"][1])
    s, _ = model.score_span(inp["ind"], inp["win"], b["train"][0], cutoff, inp["ambient"])
    return s


def decide(
    registry: Registry,
    name: str,
    version: str,
    gates: list[Gate],
    auto_promote: bool,
    alias: str = "champion",
) -> str:
    """Tag the version with every gate and move the alias (champion, or challenger for the second finalist)
    when all pass and auto_promote is on."""
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
        registry.set_alias(name, alias, version)
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
    alias: str = CHAMPION,
) -> dict[str, Any]:
    """Register the training run's model, evaluate the gates and move the alias when allowed (champion, or
    challenger for the second finalist, 2026-10-31)."""
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
        run_id = reg.find_challenger(f"{experiment_root.rstrip('/')}/ml_model", job_run_id)
    tags = reg.run_tags(run_id)
    name = model_name(catalog)
    version, reused = register_or_reuse(reg, run_id, name)
    LOG.info("%s version %s from run %s", "reusing" if reused else "registered", version, run_id)
    identity = (
        run_id,
        tags.get("settings", ""),
        str(job_run_id or ""),
        tags.get("model_version", ""),
        pdm_version,
    )
    for k, v in zip(IDENTITY_TAGS, identity, strict=True):
        reg.set_tag(name, version, tag_key(k), v)
    model = reg.load(name, version)
    tables = {
        "ml_train_scores": store.read_pd("gold", "ml_train_scores"),
        "ml_train_segments": store.read_pd("gold", "ml_train_segments"),
        "ml_candidates": store.read_pd("gold", "ml_candidates"),
        "ml_settings": store.read_pd("gold", "ml_settings"),
        "points": tasks._points(store),
    }
    tables["ml_train_scores"]["hour_utc"] = tasks._utc(tables["ml_train_scores"]["hour_utc"])
    gates = evaluate_gates(model, tags, tables, training_prediction(model, store, cfg))
    for g in gates:
        LOG.info("gate %s %s: %s", g.name, "pass" if g.passed else "fail", g.detail)
    status = decide(reg, name, version, gates, auto_promote, alias)
    champion = reg.alias_version(name, CHAMPION)
    holder = reg.alias_version(name, alias)
    LOG.info(
        "version %s %s; %s is version %s; champion is version %s", version, status, alias, holder, champion
    )
    if holder is None:
        raise RuntimeError(f"{name} has no {alias} version; version {version} is {status}")
    return {
        "run_id": run_id,
        "version": version,
        "reused": reused,
        "status": status,
        "champion": champion,
        "alias": alias,
        "alias_version": holder,
        "gates": {g.name: g.passed for g in gates},
    }

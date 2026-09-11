"""Full-record CLI path on a short window, single process, in a temporary directory."""

import copy

import generate.__main__ as cli
import generate.checks.run_report as run_report
import generate.provenance as provenance
import generate.run_full as run_full

import pdm.paths as paths
from pdm.config import load_config


def test_full_path_short_window(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["meta"]["shadow"] = {"start": "2023-07-24", "end": "2023-07-26"}
    for mod in (cli, run_report, provenance):
        monkeypatch.setattr(mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "BRONZE_DIR", tmp_path / "data" / "bronze")
    monkeypatch.setattr(cli, "load_config", lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(run_full, "load_config", lambda: copy.deepcopy(cfg))
    ctx = cli.run(pilot=False, window_days=28, workers=1)
    report = (tmp_path / "runs" / ctx.run_id / "run_report.md").read_text(encoding="utf-8")
    assert "Headline counts (section 16)" in report
    for n in ("| 1 |", "| 5 |", "| 12 |", "| 14 |", "| 15 |", "| 16 |", "| 17 |"):
        assert n in report
    bronze = tmp_path / "data" / "bronze"
    for sub in (
        "cm_platform/scalars",
        "controller/minute_agg",
        "controller/hz1",
        "compressor_ctrl/minute",
        "cmms/work_orders.parquet",
        "erp/jobs.parquet",
        "register/sensors.parquet",
    ):
        assert (bronze / sub).exists(), sub

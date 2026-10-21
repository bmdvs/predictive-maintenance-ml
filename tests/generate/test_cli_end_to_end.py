"""End-to-end CLI path on a short pilot window in a temporary directory."""

import generate.__main__ as cli
import generate.checks.run_report as run_report
import generate.provenance as provenance

import pdm.paths as paths


def test_pilot_path_short_window(tmp_path, monkeypatch):
    for mod in (cli, run_report, provenance):
        monkeypatch.setattr(mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    ctx = cli.run(pilot=True, window_days=12)
    report = (tmp_path / "runs" / ctx.run_id / "run_report.md").read_text(encoding="utf-8")
    assert report.startswith(f"Run: {ctx.run_id}")
    assert "| 1 | Kinematics" in report
    cli.recheck(ctx.run_id)

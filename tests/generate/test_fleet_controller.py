"""Fleet machine engine and controller export on a short window."""

import pandas as pd
import pytest
from generate.controller_export import render_machine
from generate.fleet import FleetMachineGroup
from generate.provenance import RunContext, rng_for
from generate.shared import build_shared

from generate import shop_schedule
from pdm.config import load_config, load_register
from pdm.ingest.mtconnect_stream import aggregate_minutes, compare_aggregates


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    cfg = load_config()
    reg = load_register()
    ctx = RunContext("test", "0" * 64, "test", 1, "test")
    start = pd.Timestamp("2026-04-06", tz=shop_schedule.LOCAL_TZ).tz_convert("UTC")
    end = start + pd.Timedelta(days=3)
    shared, sats = build_shared(cfg, reg, ctx, start, end, set())
    sched, _, _ = shop_schedule.machine_schedule(
        cfg,
        reg.machines["VMC-01"],
        shared.frame,
        shared.holidays,
        sats["other"],
        rng_for(1, "s"),
        [],
        None,
        [],
    )
    axis_events = [
        {
            "machine_id": "VMC-01",
            "axis": "x",
            "start": start,
            "end": end + pd.Timedelta(days=30),
            "friction_end": 1.3,
            "preload_end": 0.6,
            "backlash_increase_um": 10.0,
        }
    ]
    grp = FleetMachineGroup(ctx, cfg, reg, shared, "VMC-01", sched, axis_events)
    out = grp.run()
    root = tmp_path_factory.mktemp("bronze")
    truth = tmp_path_factory.mktemp("truth")
    mh = pd.DataFrame(grp.machine_hours)
    counts = render_machine(
        cfg, reg, ctx, "VMC-01", sched, mh, out.events, shared.shop_ambient, (start, end), root, truth
    )
    return root, counts, mh


def test_machine_hours_align(rendered):
    _, _, mh = rendered
    assert len(mh) == 72
    assert mh["axis_mult_x"].iloc[-1] > 1.0
    assert mh["axis_mult_b"].isna().all()


def test_minute_aggregates_reconcile_to_1hz(rendered):
    root, counts, _ = rendered
    assert counts["hz1_rows"] > 0 and counts["minute_rows"] > 0
    hz = pd.read_parquet(root / "controller" / "hz1")
    stored = pd.read_parquet(root / "controller" / "minute_agg").drop(columns=["run_id", "config_hash"])
    rec = aggregate_minutes(hz.drop(columns=["run_id", "config_hash"]))
    res = compare_aggregates(stored, rec, tolerance=1e-6)
    assert res["rows_unmatched"] == 0 and res["within_tolerance"] == 1.0
    assert set(hz["execution"]) <= {"active", "ready", "stopped", "interrupted"}
    assert hz["b_load_pct"].isna().all()

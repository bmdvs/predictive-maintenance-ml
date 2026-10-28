"""Engine smoke test: three local days for one HMC and the compressor pair, in memory."""

import pandas as pd
import pytest
from generate.asset_group import CompressorGroup, HmcGroup
from generate.provenance import RunContext, rng_for
from generate.shared import build_shared

from generate import plant_air, shop_schedule
from pdm.config import load_config, load_register


@pytest.fixture(scope="module")
def outputs():
    cfg = load_config()
    reg = load_register()
    ctx = RunContext("test", "0" * 64, "test", 1, "test")
    start = pd.Timestamp("2023-07-10", tz=shop_schedule.LOCAL_TZ).tz_convert("UTC")
    end = start + pd.Timedelta(days=3)
    rendered = {"HMC-01-SP-F", "HMC-01-TSC", "CMP-01-MDE"}
    shared, sats = build_shared(cfg, reg, ctx, start, end, rendered)
    sched, _, _ = shop_schedule.machine_schedule(
        cfg, reg.machines["HMC-01"], shared.frame, shared.holidays, sats["hmc"], rng_for(1, "s"), [], None, []
    )
    hmc = HmcGroup(ctx, cfg, reg, shared, "HMC-01", sched).run()
    air = plant_air.compressor_hours(
        cfg, shared.frame, shared.plant_minutes, list(reg.compressors), rng_for(1, "a")
    )
    cmp_out = CompressorGroup(ctx, cfg, reg, shared, air).run()
    return hmc, cmp_out


def test_rows_and_bursts(outputs):
    hmc, cmp_out = outputs
    assert hmc.bursts_rendered > 0 and cmp_out.bursts_rendered > 0
    df = pd.DataFrame(hmc.scalars)
    assert {"point_id", "hour_utc", "axis", "no_measurement"} <= set(df.columns)
    assert not df.duplicated(["point_id", "hour_utc", "axis"]).any()
    spindle_axes = set(df.loc[df.point_id == "HMC-01-SP-F", "axis"])
    assert spindle_axes <= {"res", "y"}


def test_truth_rows_cover_all_points(outputs):
    hmc, _ = outputs
    t = pd.DataFrame(hmc.truth_hours)
    assert set(t.point_id) == {"HMC-01-SP-F", "HMC-01-SP-R", "HMC-01-TSC", "HMC-01-CP"}

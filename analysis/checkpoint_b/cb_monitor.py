"""Checkpoint (b) rev2 (diag27): the monitoring report's monthly evaluation on the headline definitions
(E2 to E7), so its figures match the model report. Per evaluation-year month (shop local) and method:
conditions confirmed, strict catches with their median attributable warning, truth-based false ALERT
episodes by opening month, false ALARMs on all criteria by ALARM start month, and alert-hours (hours with
an episode open on the asset). Writes build/report_data/monitoring_month.parquet."""

import numpy as np
import pandas as pd
import r21_common as Q
from cb_common import METH, REPO, TZ, core, inputs

from pdm.evaluation import rev2 as r2

OUT = REPO / "build" / "report_data"
K = core()
SPAN, P2A, AM, cp, conf = K["SPAN"], K["P2A"], K["AM"], K["cp"], K["conf"]
inp = inputs()
LEV = {"ML Model": K["ml"], "Vendor Rules": K["vd"]}
_, deg_a = Q.degraded_hours(tuple(sorted(P2A.items())))
months = pd.date_range(SPAN[0].tz_convert(TZ).tz_localize(None), periods=12, freq="MS").strftime("%Y-%m")
n_assets = AM / 12


def month(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m")


rows = []
for m in METH:
    res = K["res"][("E7", m)]
    ct = res["ct"]
    ct = ct.assign(month=month(ct["credit_ts"]))
    av = r2.alert_view(LEV[m])
    inside = ((av["opened"] >= SPAN[0]) & (av["opened"] < SPAN[1])).to_numpy()
    credited = {
        i
        for i, s in zip(ct["credited_episode"], ct["strict"], strict=True)
        if s and i is not None and not (isinstance(i, float) and np.isnan(i))
    }
    ftruth = np.zeros(len(av), dtype=bool)
    for j, r in enumerate(av.itertuples()):
        if inside[j] and av.index[j] not in credited:
            ftruth[j] = not Q.present_between(deg_a.get(r.asset_id), r.opened, r.closed)
    assert int(ftruth.sum()) == res["false_truth"], (m, int(ftruth.sum()), res["false_truth"])
    fmo = month(av.loc[ftruth, "opened"]).value_counts()
    emo = month(av.loc[inside, "opened"]).value_counts()
    ao = r2.alarm_outcomes(LEV[m], conf, inp["wo"], SPAN, SPAN[1], False, False)
    amo = month(ao.loc[ao["false_alarm"], "alarm_ts"]).value_counts()
    alarms_mo = month(ao["alarm_ts"]).value_counts()
    hrs = pd.Series(0, index=months, dtype=float)
    for _a, g in av.groupby("asset_id"):
        s = set()
        for x in g.itertuples():
            if x.closed > SPAN[0] and x.opened < SPAN[1]:
                s.update(
                    pd.date_range(
                        pd.Timestamp(max(x.opened, SPAN[0])).floor("h"),
                        pd.Timestamp(min(x.closed, SPAN[1])),
                        freq="h",
                        inclusive="left",
                    )
                )
        if s:
            hm = month(pd.Series(sorted(s))).value_counts()
            hrs = hrs.add(hm.reindex(months).fillna(0), fill_value=0)
    assert int(hrs.sum()) == res["alert_hours"], (m, hrs.sum(), res["alert_hours"])
    for mo in months:
        c = ct[ct["month"] == mo]
        rows.append(
            {
                "method": m,
                "month": mo,
                "conditions": len(c),
                "caught": int(c["strict"].sum()),
                "warning_days_median": float(c.loc[c["strict"], "attributable_warning"].median())
                if c["strict"].any()
                else np.nan,
                "alert_episodes": int(emo.get(mo, 0)),
                "false_alerts": int(fmo.get(mo, 0)),
                "false_per_asset_month": fmo.get(mo, 0) / n_assets,
                "alarms": int(alarms_mo.get(mo, 0)),
                "false_alarms": int(amo.get(mo, 0)),
                "false_alarms_per_asset_month": amo.get(mo, 0) / n_assets,
                "alert_hours_per_asset_month": hrs[mo] / n_assets,
            }
        )
out = pd.DataFrame(rows)
out.to_parquet(OUT / "monitoring_month.parquet", index=False)
pd.set_option("display.width", 250)
print(out.to_string())
y = out.groupby("method").agg(
    conditions=("conditions", "sum"),
    caught=("caught", "sum"),
    false_alerts=("false_alerts", "sum"),
    false_alarms=("false_alarms", "sum"),
    alarms=("alarms", "sum"),
)
y["false per asset-month"] = y["false_alerts"] / AM
y["false ALARMs per asset-month"] = y["false_alarms"] / AM
y["alert-hours per asset-month"] = out.groupby("method")["alert_hours_per_asset_month"].sum() / 12
print(y.to_string())

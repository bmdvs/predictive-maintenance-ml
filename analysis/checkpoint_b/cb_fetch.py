"""Checkpoint (b) rev2: reads the prod tables the analysis uses into build/cb_cache (read only), through the SQL
statement API (pdm.reports.queue_data.fetch: DATABRICKS_CONFIG_PROFILE and PDM_WAREHOUSE_ID from the environment)."""

import json
from pathlib import Path

from pdm.reports.queue_data import fetch

C = Path(__file__).resolve().parents[2] / "build" / "cb_cache"
C.mkdir(parents=True, exist_ok=True)
GOLD = ["eval_conditions", "eval_alert_outcomes", "eval_alarm_outcomes", "eval_catch_by_condition", "eval_combined", "eval_false_alarms_by_criterion",
        "eval_groups", "eval_inspection_monthly", "eval_matched", "eval_methods", "eval_ml_curve", "eval_value", "ml_candidates", "ml_candidates_fold",
        "ml_selection_curves", "ml_training_info", "ml_vendor_fold", "ml_settings", "ml_segments", "ml_train_segments", "alarm_criteria", "level_episodes",
        "diagnosis_agreement_shadow", "rule_floor_training_curve", "evaluation_headline_year"]
BRONZE = ["cmms_work_orders", "reg_machines", "erp_contribution_margin", "cmms_pm_schedule"]
out = {}
for n in GOLD:
    d = fetch(f"SELECT * FROM pdm.gold.{n}")
    d.to_parquet(C / f"{n}.parquet", index=False)
    out[n] = len(d)
for n in BRONZE:
    d = fetch(f"SELECT * FROM pdm.bronze.{n}")
    d.to_parquet(C / f"{n}.parquet", index=False)
    out[n] = len(d)
(C / "tables.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

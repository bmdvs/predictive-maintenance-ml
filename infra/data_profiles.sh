#!/usr/bin/env bash
# Databricks data profiles for monitoring (section 10.10), created with the Databricks CLI by the
# catalog owner: an inference profile on gold.scoring_log (monthly, classification: prediction is the
# exceedance at the operating percentile, label the 30-day outcome) and a time series profile on
# silver.hourly_indicators (monthly, sliced by point). The inference profile's baseline table is
# gold.scoring_log_baseline, the validation-period rows of the scoring log written by the monitor task.
# Profile outputs go to <catalog>.monitoring; the monthly monitor task refreshes both. --check reads them
# back (read-only); --set-baseline adds the baseline table to an existing inference profile.
#
# Optional: PDM_CATALOG (default pdm), PDM_PROFILE_ASSETS (default /Workspace/Shared/pdm_profiles),
# DATABRICKS_CONFIG_PROFILE
set -euo pipefail

CATALOG="${PDM_CATALOG:-pdm}"
ASSETS="${PDM_PROFILE_ASSETS:-/Workspace/Shared/pdm_profiles}"

if [[ "${1:-}" == "--check" ]]; then
  for t in gold.scoring_log silver.hourly_indicators; do
    databricks quality-monitors get "$CATALOG.$t" -o json
  done
  exit 0
fi

INFERENCE_LOG='{
    "granularities": ["1 month"],
    "timestamp_col": "hour_utc",
    "model_id_col": "model_version",
    "prediction_col": "prediction",
    "label_col": "outcome",
    "problem_type": "PROBLEM_TYPE_CLASSIFICATION"
  }'

if [[ "${1:-}" == "--set-baseline" ]]; then
  databricks quality-monitors update "$CATALOG.gold.scoring_log" --json '{
  "output_schema_name": "'"$CATALOG"'.monitoring",
  "baseline_table_name": "'"$CATALOG"'.gold.scoring_log_baseline",
  "inference_log": '"$INFERENCE_LOG"',
  "slicing_exprs": ["asset_id", "period"]
}'
  exit 0
fi

databricks quality-monitors create "$CATALOG.gold.scoring_log" --json '{
  "output_schema_name": "'"$CATALOG"'.monitoring",
  "assets_dir": "'"$ASSETS"'/scoring_log",
  "baseline_table_name": "'"$CATALOG"'.gold.scoring_log_baseline",
  "inference_log": '"$INFERENCE_LOG"',
  "slicing_exprs": ["asset_id", "period"]
}'

databricks quality-monitors create "$CATALOG.silver.hourly_indicators" --json '{
  "output_schema_name": "'"$CATALOG"'.monitoring",
  "assets_dir": "'"$ASSETS"'/hourly_indicators",
  "time_series": {"granularities": ["1 month"], "timestamp_col": "hour_utc"},
  "slicing_exprs": ["point_id"]
}'

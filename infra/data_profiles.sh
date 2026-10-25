#!/usr/bin/env bash
# Databricks data profiles for monitoring (section 10.10), created with the Databricks CLI by the
# catalog owner: an inference profile on gold.scoring_log (monthly, classification: prediction is the
# exceedance at the operating percentile, label the 30-day outcome) and a time series profile on
# silver.hourly_indicators (monthly, sliced by point). Profile outputs go to <catalog>.monitoring; the
# monthly monitor task refreshes both. --check reads them back (read-only).
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

databricks quality-monitors create "$CATALOG.gold.scoring_log" "$CATALOG.monitoring" "$ASSETS/scoring_log" --json '{
  "inference_log": {
    "granularities": ["1 month"],
    "timestamp_col": "hour_utc",
    "model_id_col": "model_version",
    "prediction_col": "prediction",
    "label_col": "outcome",
    "problem_type": "PROBLEM_TYPE_CLASSIFICATION"
  },
  "slicing_exprs": ["asset_id", "period"]
}'

databricks quality-monitors create "$CATALOG.silver.hourly_indicators" "$CATALOG.monitoring" "$ASSETS/hourly_indicators" --json '{
  "time_series": {"granularities": ["1 month"], "timestamp_col": "hour_utc"},
  "slicing_exprs": ["point_id"]
}'

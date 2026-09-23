#!/usr/bin/env bash
# Unity Catalog grants for the deploy service principal (prod run-as for the monthly job and the
# pipeline). Prints the SQL to run in the SQL editor as the owner of the catalog, and the CLI command that
# moves the registered model's ownership (SQL has no statement for it); --check reads the grants and
# owners back with the Databricks CLI (read-only). The application id comes from the environment and
# is never written to the repository.
#
# Required: PDM_DEPLOY_SP (application id of the service principal)
# Optional: PDM_CATALOG (default pdm), DATABRICKS_CONFIG_PROFILE for --check
set -euo pipefail

SP="${PDM_DEPLOY_SP:?set PDM_DEPLOY_SP to the service principal application id}"
CATALOG="${PDM_CATALOG:-pdm}"

# Tables the job tasks write (train, score, monitor); the pipeline's own tables are not listed.
JOB_TABLES=(
  asset_status cases cases_sweep component_lives compressor_discharge_daily controller_health_daily
  controller_health_exceedances diagnosis_agreement_validation evaluation_curves_validation
  evaluation_events_validation evaluation_headline_by_type_validation evaluation_headline_validation
  evaluation_matched_validation nbm_candidates nbm_exceedance_diagnosis nbm_exceedances nbm_models
  nbm_scores nbm_train_scores point_healthy_stats prognosis prognosis_skipped rule_floor_training_curve
  signals supervised_gate survival_aft_coefficients survival_class_weibull survival_shared_shape_weibull
  survival_summary work_requests
)

if [[ "${1:-}" == "--check" ]]; then
  databricks grants get-effective catalog "$CATALOG" --principal "$SP" -o json
  for s in bronze silver gold monitoring ml landing; do
    databricks grants get-effective schema "$CATALOG.$s" --principal "$SP" -o json
  done
  databricks grants get-effective volume "$CATALOG.landing.exports" --principal "$SP" -o json
  databricks registered-models get "$CATALOG.ml.nbm_point_models" -o json | grep '"owner"'
  for t in "${JOB_TABLES[@]}"; do
    printf '%s ' "$t"; databricks tables get "$CATALOG.gold.$t" -o json | grep '"owner"'
  done
  exit 0
fi

echo "GRANT USE CATALOG ON CATALOG $CATALOG TO \`$SP\`;"
for s in bronze silver gold monitoring; do
  echo "GRANT USE SCHEMA, SELECT, MODIFY, CREATE TABLE, CREATE MATERIALIZED VIEW ON SCHEMA $CATALOG.$s TO \`$SP\`;"
done
echo "GRANT USE SCHEMA, CREATE MODEL ON SCHEMA $CATALOG.ml TO \`$SP\`;"
echo "GRANT USE SCHEMA ON SCHEMA $CATALOG.landing TO \`$SP\`;"
echo "GRANT READ VOLUME, WRITE VOLUME ON VOLUME $CATALOG.landing.exports TO \`$SP\`;"
echo "-- Registered model owner (no SQL statement exists for this; run with the Databricks CLI or set it in Catalog Explorer):"
echo "-- databricks registered-models update $CATALOG.ml.nbm_point_models --owner $SP"
for t in "${JOB_TABLES[@]}"; do
  echo "ALTER TABLE $CATALOG.gold.$t OWNER TO \`$SP\`;"
done

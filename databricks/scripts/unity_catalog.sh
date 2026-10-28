#!/usr/bin/env bash
# Unity Catalog objects for the landing and managed storage, catalog, schemas and landing volume, as
# Databricks CLI commands so the setup can be reproduced, and a read-only check of an existing setup.
# Identifiers come from environment variables, never from the repository.
#
# Required: PDM_STORAGE_ACCOUNT, and for create only PDM_ACCESS_CONNECTOR_ID (resource id of ac-pdm)
# Usage: databricks/scripts/unity_catalog.sh [PROFILE]              creates the objects
#        databricks/scripts/unity_catalog.sh [PROFILE] --check      read-only verification against the expected settings
#        databricks/scripts/unity_catalog.sh [PROFILE] --dev        creates the development catalog (bundle dev target):
#                                                      catalog pdm_dev on a pdm_dev folder of the managed
#                                                      container, the same schemas, managed volume
#                                                      landing.autoloader for Auto Loader schema tracking
#        databricks/scripts/unity_catalog.sh [PROFILE] --dev-check  read-only verification of the development catalog
#
# The external locations are created with validation skipped (force-created): file events are not
# configured on the storage account, and Auto Loader reads by directory listing.
set -euo pipefail

PROFILE="${1:-${DATABRICKS_CONFIG_PROFILE:-DEFAULT}}"
MODE="${2:-}"
CATALOG="${PDM_CATALOG:-pdm}"
ACCOUNT="${PDM_STORAGE_ACCOUNT:?set PDM_STORAGE_ACCOUNT}"
LANDING="abfss://landing@${ACCOUNT}.dfs.core.windows.net/"
MANAGED="abfss://managed@${ACCOUNT}.dfs.core.windows.net/"
SCHEMAS=(landing bronze silver gold ml monitoring)
DEV_CATALOG="${PDM_DEV_CATALOG:-pdm_dev}"
DEV_MANAGED="${MANAGED}${DEV_CATALOG}"
db() { databricks --profile "${PROFILE}" "$@"; }

if [[ "${MODE}" == "--dev" ]]; then
  db catalogs create "${DEV_CATALOG}" --storage-root "${DEV_MANAGED}"
  for s in "${SCHEMAS[@]}"; do db schemas create "${s}" "${DEV_CATALOG}"; done
  db volumes create "${DEV_CATALOG}" landing autoloader MANAGED
  exit 0
fi

if [[ "${MODE}" == "--check" || "${MODE}" == "--dev-check" ]]; then
  fail=0
  expect() {
    local what="$1" got="$2" want="$3"
    if [[ "${got%/}" == "${want%/}" ]]; then echo "ok    ${what}: ${got}"; else echo "DIFF  ${what}: got '${got}', expected '${want}'"; fail=1; fi
  }
  field() { python -c "import json,sys; d=json.load(sys.stdin); print(d.get('$1') or '')"; }
  if [[ "${MODE}" == "--dev-check" ]]; then
    expect "catalog ${DEV_CATALOG} storage root" "$(db catalogs get "${DEV_CATALOG}" -o json | field storage_root)" "${DEV_MANAGED}"
    for s in "${SCHEMAS[@]}"; do
      expect "schema ${DEV_CATALOG}.${s} storage root" "$(db schemas get "${DEV_CATALOG}.${s}" -o json | field storage_root)" ""
    done
    expect "volume ${DEV_CATALOG}.landing.autoloader type" "$(db volumes read "${DEV_CATALOG}.landing.autoloader" -o json | field volume_type)" "MANAGED"
    exit "${fail}"
  fi
  expect "credential cred_pdm" "$(db storage-credentials get cred_pdm -o json | field name)" "cred_pdm"
  expect "loc_pdm_landing url" "$(db external-locations get loc_pdm_landing -o json | field url)" "${LANDING}"
  expect "loc_pdm_landing credential" "$(db external-locations get loc_pdm_landing -o json | field credential_name)" "cred_pdm"
  expect "loc_pdm_managed url" "$(db external-locations get loc_pdm_managed -o json | field url)" "${MANAGED}"
  expect "loc_pdm_managed credential" "$(db external-locations get loc_pdm_managed -o json | field credential_name)" "cred_pdm"
  expect "catalog ${CATALOG} storage root" "$(db catalogs get "${CATALOG}" -o json | field storage_root)" "${MANAGED}"
  for s in "${SCHEMAS[@]}"; do
    expect "schema ${CATALOG}.${s} storage root" "$(db schemas get "${CATALOG}.${s}" -o json | field storage_root)" ""
  done
  expect "volume ${CATALOG}.landing.exports type" "$(db volumes read "${CATALOG}.landing.exports" -o json | field volume_type)" "EXTERNAL"
  expect "volume ${CATALOG}.landing.exports location" "$(db volumes read "${CATALOG}.landing.exports" -o json | field storage_location)" "${LANDING}exports"
  exit "${fail}"
fi

db storage-credentials create --json "{\"name\": \"cred_pdm\", \"azure_managed_identity\": {\"access_connector_id\": \"${PDM_ACCESS_CONNECTOR_ID:?set PDM_ACCESS_CONNECTOR_ID}\"}}"
db external-locations create loc_pdm_landing "${LANDING}" cred_pdm --skip-validation
db external-locations create loc_pdm_managed "${MANAGED}" cred_pdm --skip-validation
db catalogs create "${CATALOG}" --storage-root "${MANAGED}"
for s in "${SCHEMAS[@]}"; do db schemas create "${s}" "${CATALOG}"; done
db volumes create "${CATALOG}" landing exports EXTERNAL --storage-location "${LANDING}exports"

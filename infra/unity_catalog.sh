#!/usr/bin/env bash
# Unity Catalog objects for the landing storage, catalog, schemas and landing volume (environment setup).
# They are created by hand first; this script records the same objects as CLI commands so the
# setup can be reproduced and checked. Identifiers come from environment variables, never the repo.
#
# Required: PDM_STORAGE_ACCOUNT, PDM_ACCESS_CONNECTOR_ID (resource id of ac-pdm)
# Usage: infra/unity_catalog.sh [PROFILE]          creates the objects
#        infra/unity_catalog.sh [PROFILE] --check  read-only verification
set -euo pipefail

PROFILE="${1:-${DATABRICKS_CONFIG_PROFILE:-DEFAULT}}"
MODE="${2:-}"
CATALOG="${PDM_CATALOG:-pdm}"
URL="abfss://landing@${PDM_STORAGE_ACCOUNT:?set PDM_STORAGE_ACCOUNT}.dfs.core.windows.net/"
db() { databricks --profile "${PROFILE}" "$@"; }

if [[ "${MODE}" == "--check" ]]; then
  db storage-credentials get cred_pdm
  db external-locations get loc_pdm_landing
  db catalogs get "${CATALOG}"
  for s in landing bronze silver gold ml monitoring; do db schemas get "${CATALOG}.${s}"; done
  db volumes read "${CATALOG}.landing.exports"
  exit 0
fi

db storage-credentials create --json "{\"name\": \"cred_pdm\", \"azure_managed_identity\": {\"access_connector_id\": \"${PDM_ACCESS_CONNECTOR_ID:?set PDM_ACCESS_CONNECTOR_ID}\"}}"
db external-locations create loc_pdm_landing "${URL}" cred_pdm
db catalogs create "${CATALOG}"
for s in landing bronze silver gold ml monitoring; do db schemas create "${s}" "${CATALOG}"; done
db volumes create "${CATALOG}" landing exports EXTERNAL --storage-location "${URL}exports"

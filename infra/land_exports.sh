#!/usr/bin/env bash
# Copy the local source exports to the landing volume. Idempotent: files already present with the
# same size are skipped by the CLI, changed files are overwritten.
#
# Requires: Databricks CLI authenticated to the workspace (profile in ~/.databrickscfg, not in the repo).
# Usage: infra/land_exports.sh [PROFILE]
set -euo pipefail

PROFILE="${1:-${DATABRICKS_CONFIG_PROFILE:-DEFAULT}}"
CATALOG="${PDM_CATALOG:-pdm}"
LOCAL_ROOT="${PDM_LOCAL_EXPORTS:-data/bronze}"
VOLUME="dbfs:/Volumes/${CATALOG}/landing/exports"

for system in cm_platform controller compressor_ctrl cmms erp register; do
  echo "landing ${system}"
  databricks --profile "${PROFILE}" fs cp -r --overwrite "${LOCAL_ROOT}/${system}" "${VOLUME}/${system}"
done
echo "done: ${VOLUME}"

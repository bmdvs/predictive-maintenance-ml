#!/usr/bin/env bash
# Read-only check of the Azure resources from the environment setup: the storage account (hierarchical
# namespace, replication, region), its containers, the access connector and its role on the account,
# and the workspace tier. Nothing is created or changed. Identifiers come from environment variables.
#
# Required: PDM_RESOURCE_GROUP (rg-pdm), PDM_STORAGE_ACCOUNT, PDM_LOCATION (centralus)
# Optional: PDM_WORKSPACE (dbw-pdm), PDM_ACCESS_CONNECTOR (ac-pdm)
# Usage: infra/azure_check.sh
set -euo pipefail

RG="${PDM_RESOURCE_GROUP:?set PDM_RESOURCE_GROUP}"
ACCOUNT="${PDM_STORAGE_ACCOUNT:?set PDM_STORAGE_ACCOUNT}"
LOCATION="${PDM_LOCATION:?set PDM_LOCATION}"
WORKSPACE="${PDM_WORKSPACE:-dbw-pdm}"
CONNECTOR="${PDM_ACCESS_CONNECTOR:-ac-pdm}"
fail=0
expect() {
  local what="$1" got="$2" want="$3"
  if [[ "${got}" == "${want}" ]]; then echo "ok    ${what}: ${got}"; else echo "DIFF  ${what}: got '${got}', expected '${want}'"; fail=1; fi
}

sa() { az storage account show -g "${RG}" -n "${ACCOUNT}" --query "$1" -o tsv; }
expect "storage hierarchical namespace" "$(sa isHnsEnabled)" "true"
expect "storage replication" "$(sa sku.name)" "Standard_LRS"
expect "storage region" "$(sa location)" "${LOCATION}"
containers="$(az storage container-rm list -g "${RG}" --storage-account "${ACCOUNT}" --query "[].name" -o tsv | sort | tr '\n' ' ')"
for c in landing managed; do
  if [[ " ${containers} " == *" ${c} "* ]]; then echo "ok    container ${c}"; else echo "DIFF  container ${c} missing"; fail=1; fi
done

connector_id="$(az resource show -g "${RG}" -n "${CONNECTOR}" --resource-type Microsoft.Databricks/accessConnectors --query id -o tsv)"
principal="$(az resource show --ids "${connector_id}" --query identity.principalId -o tsv)"
scope="$(sa id)"
role="$(az role assignment list --assignee "${principal}" --scope "${scope}" --query "[?roleDefinitionName=='Storage Blob Data Contributor'].roleDefinitionName" -o tsv)"
expect "access connector role on storage" "${role}" "Storage Blob Data Contributor"

ws() { az resource show -g "${RG}" -n "${WORKSPACE}" --resource-type Microsoft.Databricks/workspaces --query "$1" -o tsv; }
expect "workspace tier" "$(ws sku.name)" "trial"
expect "workspace region" "$(ws location)" "${LOCATION}"
exit "${fail}"

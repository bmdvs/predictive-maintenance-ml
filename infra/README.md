# infra

Setup steps the bundle does not express, as Azure CLI and Databricks CLI commands with variables. No
workspace URLs, tenant or subscription identifiers, keys or tokens are stored here; they come from
environment variables or the local CLI profiles (`~/.databrickscfg`, `~/.azure`, both outside the
repository).

| Script | Purpose | Stage |
|---|---|---|
| `azure_check.sh` | Read-only check of the Azure resources: storage account (hierarchical namespace, LRS, region), containers `landing` and `managed`, access connector `ac-pdm` with Storage Blob Data Contributor on the account, workspace tier and region | L0, L1 |
| `unity_catalog.sh` | Storage credential `cred_pdm`; external locations `loc_pdm_landing` (container `landing`) and `loc_pdm_managed` (container `managed`), both created with validation skipped; catalog `pdm` with its managed location on `loc_pdm_managed`; schemas `landing`, `bronze`, `silver`, `gold`, `ml`, `monitoring` with no schema-level locations; external volume `pdm.landing.exports`. `--check` verifies read-only | L1 |
| `land_exports.sh` | Copies `data/bronze/<system>/` to `/Volumes/pdm/landing/exports/<system>/` (idempotent) | L2 |

Environment variables:

| Variable | Meaning |
|---|---|
| `DATABRICKS_CONFIG_PROFILE` | CLI profile for the workspace |
| `PDM_CATALOG` | catalog name (default `pdm`) |
| `PDM_RESOURCE_GROUP` | Azure resource group (`rg-pdm`) |
| `PDM_LOCATION` | Azure region (`centralus`) |
| `PDM_STORAGE_ACCOUNT` | ADLS Gen2 account name (`stpdm<suffix>`) |
| `PDM_WORKSPACE` | workspace name (default `dbw-pdm`) |
| `PDM_ACCESS_CONNECTOR` | access connector name (default `ac-pdm`) |
| `PDM_ACCESS_CONNECTOR_ID` | resource id of the access connector (create only) |
| `PDM_LOCAL_EXPORTS` | local export root (default `data/bronze`) |

File events are not configured on the storage account (they need roles that were not granted), so both
external locations were created with validation skipped. Auto Loader reads the landing volume by
directory listing (`cloudFiles.useNotifications` false); nothing depends on file events.

The Azure resources themselves (resource group `rg-pdm`, workspace `dbw-pdm`, storage account,
access connector and its role assignment) are created by hand in the Azure portal.

## Bundle

`resources/unity_catalog.yml` declares the six schemas and the landing volume for the `prod` target
only (development mode would otherwise create prefixed copies). The catalog, its managed location, the
storage credential and the external locations are not bundle resource types and stay in
`unity_catalog.sh`. The objects already exist, so before the first prod deploy each is bound to the
existing object, with `landing_volume_location` set for the target:

```bash
export DATABRICKS_CONFIG_PROFILE=<profile for the workspace>
export BUNDLE_VAR_landing_volume_location=abfss://landing@<storage account>.dfs.core.windows.net/exports
for s in landing bronze silver gold ml monitoring; do databricks bundle deployment bind "$s" "pdm.$s" -t prod; done
databricks bundle deployment bind exports pdm.landing.exports -t prod
```

Deployment of the pipeline and the monthly job is by the bundle: `databricks bundle deploy -t dev`
(and `-t prod` from GitHub Actions). The workspace host comes from the CLI profile or
`DATABRICKS_HOST`; bundles do not accept a variable for the host.

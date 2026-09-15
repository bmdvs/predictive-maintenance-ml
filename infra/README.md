# infra

Setup steps the bundle does not express, as Databricks CLI commands with variables. No workspace URLs,
tenant or subscription identifiers, keys or tokens are stored here; they come from environment
variables or the local CLI profile (`~/.databrickscfg`, gitignored).

| Script | Purpose | Stage |
|---|---|---|
| `unity_catalog.sh` | Storage credential `cred_pdm`, external location `loc_pdm_landing`, catalog `pdm`, schemas `landing`, `bronze`, `silver`, `gold`, `ml`, `monitoring`, volume `pdm.landing.exports`; `--check` verifies read-only | L1 |
| `land_exports.sh` | Copies `data/bronze/<system>/` to `/Volumes/pdm/landing/exports/<system>/` (idempotent) | L2 |

Environment variables:

| Variable | Meaning |
|---|---|
| `DATABRICKS_CONFIG_PROFILE` | CLI profile for the workspace |
| `PDM_CATALOG` | catalog name (default `pdm`) |
| `PDM_STORAGE_ACCOUNT` | ADLS Gen2 account name (`stpdm<suffix>`) |
| `PDM_ACCESS_CONNECTOR_ID` | resource id of the access connector `ac-pdm` |
| `PDM_LOCAL_EXPORTS` | local export root (default `data/bronze`) |

The Azure resources themselves (resource group `rg-pdm`, workspace `dbw-pdm`, storage account,
access connector) are created by hand in the Azure portal.

Deployment of the pipeline and the monthly job is by the bundle: `databricks bundle deploy -t dev`
(and `-t prod` from GitHub Actions), with `workspace_host` supplied as a variable.

# infra

Setup steps the bundle does not express, as Azure CLI and Databricks CLI commands with variables. No
workspace URLs, tenant or subscription identifiers, keys or tokens are stored here; they come from
environment variables or the local CLI profiles (`~/.databrickscfg`, `~/.azure`, both outside the
repository).

| Script | Purpose | Stage |
|---|---|---|
| `azure_check.sh` | Read-only check of the Azure resources: storage account (hierarchical namespace, LRS, region), containers `landing` and `managed`, access connector `ac-pdm` with Storage Blob Data Contributor on the account, workspace tier and region | Environment setup |
| `unity_catalog.sh` | Storage credential `cred_pdm`; external locations `loc_pdm_landing` (container `landing`) and `loc_pdm_managed` (container `managed`), both created with validation skipped; catalog `pdm` with its managed location on `loc_pdm_managed`; schemas `landing`, `bronze`, `silver`, `gold`, `ml`, `monitoring` with no schema-level locations; external volume `pdm.landing.exports`. `--check` verifies read-only | Environment setup |
| `land_exports.sh` | Copies `data/bronze/<system>/` to `/Volumes/pdm/landing/exports/<system>/` (idempotent); `--check` compares file counts and bytes per folder, read-only | Data landing |
| `grant_deploy_sp.sh` | Prints the Unity Catalog grants for the deploy service principal (catalog, schemas, landing volume, ownership of the registered model and of the job-written gold tables) to run in the SQL editor as the catalog owner; `--check` reads them back, read-only | Deployment |

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
| `PDM_DEPLOY_SP` | application id of the deploy service principal (grants script; never committed) |
| `BUNDLE_VAR_deploy_sp` | the same application id for the bundle's prod `run_as` |

File events are not configured on the storage account (they need roles that were not granted), so both
external locations were created with validation skipped. Auto Loader reads the landing volume by
directory listing (`cloudFiles.useNotifications` false); nothing depends on file events.

The Azure resources themselves (resource group `rg-pdm`, workspace `dbw-pdm`, storage account,
access connector and its role assignment) are created in the Azure portal; `azure_check.sh` verifies them.

## Bundle

Unity Catalog objects (catalog, managed location, credential, external locations, schemas and the
landing volume) are not bundle resources; `unity_catalog.sh` holds them and `--check` verifies them. The
bundle deploys the pipeline and the monthly job.

Targets:

| Target | Mode | Catalog | Run as | Root path |
|---|---|---|---|---|
| `dev` | development (prefixed names, paused schedules, development pipeline) | `pdm_dev`; landing read from `/Volumes/pdm/landing/exports`, Auto Loader schemas in `pdm_dev.landing.autoloader` | the deploying user | the user's `.bundle/pdm/dev` |
| `prod` | production | `pdm` | the deploy service principal (`BUNDLE_VAR_deploy_sp`) | `/Workspace/Shared/.bundle/pdm/prod` |

The workspace host comes from the CLI profile or `DATABRICKS_HOST`; bundles do not accept a variable for
the host. Runs are started with `--no-wait` and followed in the UI, so no waiting command ends a run.

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run pdm_pipeline -t dev --no-wait
databricks bundle run pdm_monthly -t dev --no-wait --params auto_promote=false
```

First job run on an empty registry (a new catalog): promote registers version 1 for review, the
false-episode gate fails closed (no validation table yet) and the task then stops because no champion
exists. Set the `champion` alias on version 1 by hand, then repair promote, score and monitor with
`auto_promote=false`; promote reuses version 1. Changes that touch registration, model version tags or
aliases run on the dev target before prod.

The prod pipeline and job already exist; bind them before the first prod deploy so the bundle takes
them over with their ids and run history:

```bash
export BUNDLE_VAR_deploy_sp=<application id>
databricks bundle deployment bind pdm_pipeline <pipeline id> -t prod
databricks bundle deployment bind pdm_monthly <job id> -t prod
```

Prod deploys run from GitHub Actions (`.github/workflows/ci.yml`): bundle validate on pull requests and
pushes to main, deploy on pushes to main, with the repository secrets `DATABRICKS_HOST`,
`DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` (the service principal's OAuth secret).

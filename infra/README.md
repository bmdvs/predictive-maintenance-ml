# infra

Setup steps the bundle does not express, as Azure CLI and Databricks CLI commands with variables. No
workspace URLs, tenant or subscription identifiers, keys or tokens are stored here; they come from
environment variables or the local CLI profiles (`~/.databrickscfg`, `~/.azure`, both outside the
repository).

| Script | Purpose | Stage |
|---|---|---|
| `azure_check.sh` | Read-only check of the Azure resources: storage account (hierarchical namespace, LRS, region), containers `landing` and `managed`, access connector `ac-pdm` with Storage Blob Data Contributor on the account, workspace tier and region | Environment setup |
| `unity_catalog.sh` | Storage credential `cred_pdm`; external locations `loc_pdm_landing` (container `landing`) and `loc_pdm_managed` (container `managed`), both created with validation skipped; catalog `pdm` with its managed location on `loc_pdm_managed`; schemas `landing`, `bronze`, `silver`, `gold`, `ml`, `monitoring` with no schema-level locations; external volume `pdm.landing.exports`. `--check` verifies read-only. `--dev` creates the development catalog `pdm_dev` (managed location on a `pdm_dev` folder of the managed container, the same schemas, managed volume `landing.autoloader` for Auto Loader schema tracking); `--dev-check` verifies it read-only | Environment setup |
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

## Local Java for Spark tests

The local Spark tests and the parity runner need a JDK 17 through `JAVA_HOME`. The Microsoft OpenJDK
installer sets the machine-level `JAVA_HOME` and updates it when a JDK update installs into a new folder
(for example `jdk-17.0.20-hotspot` became `jdk-17.0.20.101-hotspot`). Do not set a user-level `JAVA_HOME`:
it overrides the machine variable, and once an update removes the folder it names, local Spark hangs at
startup. Check from a new terminal:

```
[Environment]::GetEnvironmentVariable("JAVA_HOME", "User")     # empty
[Environment]::GetEnvironmentVariable("JAVA_HOME", "Machine")  # the installed JDK folder
```

Spark workers must start the same interpreter as the driver. `tests/conftest.py` sets `PYSPARK_PYTHON` to
the running interpreter before any Spark session starts, so the tests need no environment setting. For the
parity runner on Windows, set it in the terminal for that run (`$env:PYSPARK_PYTHON = (Get-Command python).Source`
inside the activated virtual environment) rather than as a user variable.

## Bundle

Unity Catalog objects (catalog, managed location, credential, external locations, schemas and the
landing volume) are not bundle resources; `unity_catalog.sh` holds them and `--check` verifies them. The
bundle deploys the pipeline and the monthly job.

Targets:

| Target | Mode | Catalog | Run as | Root path |
|---|---|---|---|---|
| `dev` | development (prefixed names, paused schedules, development pipeline) | `pdm_dev`; landing read from `/Volumes/pdm/landing/exports`, Auto Loader schemas in `pdm_dev.landing.autoloader` | the deploying user | the user's `.bundle/pdm/dev` |
| `prod` | production | `pdm` | the job: the deploy service principal (`BUNDLE_VAR_deploy_sp`); the pipeline: its owner | `/Workspace/Shared/.bundle/pdm/prod` |

The prod root path is under `/Workspace/Shared`, so the binding (run by a person) and the deploys (run by the
service principal) share one deployment state. Workspace folder permissions are additive and `/Workspace/Shared`
gives all users manage access by inheritance, so bundle permissions cannot narrow that folder; they would also
apply to the job and the pipeline. The folder stays as it is; `bundle validate -t prod` warns about it.

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

The prod pipeline and job already exist. Before the first prod deploy, the service principal needs
permissions on both, then both are bound so the bundle takes them over with their ids and run history:

- job: the service principal is made its owner (`IS_OWNER`), so the job runs as the service principal;
  the person who built it keeps `CAN_MANAGE`;
- pipeline: the service principal gets `CAN_MANAGE`; the owner stays the person who built it, because
  only a metastore admin can change a pipeline's owner or run-as identity, so the pipeline keeps running
  as that owner.

```bash
databricks permissions update jobs <job id> --json '{"access_control_list":[{"service_principal_name":"<application id>","permission_level":"IS_OWNER"}]}'
databricks permissions update pipelines <pipeline id> --json '{"access_control_list":[{"service_principal_name":"<application id>","permission_level":"CAN_MANAGE"}]}'
```

Then bind:

```bash
export BUNDLE_VAR_deploy_sp=<application id>
databricks bundle deployment bind pdm_pipeline <pipeline id> -t prod
databricks bundle deployment bind pdm_monthly <job id> -t prod
```

The deploy service principal needs owner rights on the registered model: the model owner creates versions,
sets their tags and moves aliases. Only a metastore admin can make a service principal the owner; a
non-admin owner can transfer a model only to a group they belong to. Without metastore admin rights the
model is owned by the built-in group `account users`, which contains both the catalog owner and the
service principal, so every identity in the Databricks account has owner rights on that model. In an
account with other users, a metastore admin should instead give the model to the service principal or to a
dedicated deploy group.

Prod deploys run from GitHub Actions (`.github/workflows/ci.yml`): bundle validate on pull requests and
pushes to main, deploy on pushes to main, with the repository secrets `DATABRICKS_HOST`,
`DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` (the service principal's OAuth secret).

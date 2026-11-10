# Machine Learning Model: Predictive Maintenance

This project is built on generated vibration and temperature sensor data, machine and compressor controller data, and ERP and CMMS exports. The data is generated from each machine’s physical properties and published degradation mechanisms, so that it reflects the behavior of real equipment. See [Data Generation](#data-generation). 

The sensor data is transmitted wirelessly to the vendor's cloud platform and exported. The machine controller data is read over MTConnect, and the compressor controller data over Modbus into Ignition; both are published on an MQTT UNS and written to files. All files land in Azure Databricks, where a Lakeflow pipeline transforms them and a monthly job trains, scores and monitors the models through MLflow and the Unity Catalog model registry. See [Architecture](#architecture). 

All code related to industrial connectivity, the data platform and MLOps, and the ML model (training, scoring, alerting and evaluation) is organized below in the [Code](#code) section. 

---

<br>

A **Predictive Maintenance ML model** (anomaly detection) was built on the past three years of the shop's records. Every hour, it reads vibration and temperature measurements from 40 wireless sensor points on eight HMCs and four plant compressors, along with each machine’s load and speed from its controller. When readings depart from a sensor point’s normal behavior, the model raises an ALERT (inspection needed), escalating to an ALARM (urgent repair) when the condition becomes severe. The model is detailed in the [model report](https://bmdvs.github.io/predictive-maintenance-ml/reports/model_report.html) and summarized below.

- **Results.** The table below compares the predictive maintenance model (ML Model) against the Vendor Rules method, which is the shop's existing condition monitoring layer. Vendor Rules applies fixed threshold alerts on sensor measurements: velocity, high-frequency acceleration and bearing fault bands, and its ALERTs and ALARMs drive maintenance requests. The ML Model ran in shadow mode beside Vendor Rules during the evaluation year (July 2025 to June 2026), with its ALERTs and ALARMs recorded and not acted on, in order to evaluate its performance. As shown below, the ML Model achieved notable improvements over Vendor Rules.
  - Against 32 confirmed conditions, the ML Model caught 23 against Vendor Rules’ 22, with 41% fewer false ALERTs and 61% fewer false ALARMs.
  - On the 18 conditions both methods caught, the ML Model's ALERTs gave a median 23.7 days of warning against 21.3.

<div align="center">

<b>ML Model vs. Vendor Rules, July 2025 to June 2026</b>

| | Conditions caught | Median warning (days) | False ALERTs per asset-month | False ALARMs per asset-month | Inspection hours on false alerts |
|:---|:---:|:---:|:---:|:---:|:---:|
| Vendor Rules | 22 of 32 | 21.3 | 0.235 | 0.137 | 63 |
| **ML Model** | **23 of 32** | **23.7** | **0.140** | **0.054** | **37** |

</div>

- **Purpose and use.** Each hour, the ML Model scores every sensor point against its learned normal behavior. It raises an ALERT (inspection needed) or an ALARM (urgent repair) and opens a work request with the suspected condition and a recommended inspection. During the evaluation year, Vendor Rules drove the maintenance queue and the ML Model's status was shown beside it. 
- **Model training.** The model was trained on July 2023 to June 2025 and selected on expanding-window folds within that period. The evaluation year, July 2025 to June 2026, was held out until the model was chosen. The record holds 84 confirmed conditions on the sensor points: 52 in training and 32 in evaluation.
  - An autoencoder (a neural network ML model) is trained for each sensor point and learns its normal behavior from 13 sensor measurements (velocity, high-frequency acceleration, envelope fault bands, shaft-order energies, crest factor, kurtosis and temperature), plus operating data from the machines' controllers.
  - The model is trained on the point's healthy hours and re-centered after each repair and sensor remount.
  - The model flags ALERTs above the 99.5th percentile of healthy scores, held for six hours. An open ALERT escalates to an ALARM when the asset also crosses a severity criterion for its class: velocity in the ISO danger zone, a late-stage bearing signature, a spindle or compressor temperature limit, or sustained pump cavitation.
- **Model evaluation.**
  - **Selection.** Four model types (autoencoder, Mahalanobis distance, PCA, isolation forest) were compared across five input sets, trained and scored on the training data, and the best model (autoencoder) was selected.
  - **Detection.** Evaluation is event-based against confirmed conditions, with the same scope, crediting and false-alert definitions applied to both methods. On 32 conditions, the ML Model’s 23 catches vs. Vendor Rules' 22 is within what chance could produce; the ML Model's lower false-alert rate is found to be statistically significant (90% interval: 16% to 61% lower).
  - **Diagnosis.** The ML Model’s work requests named the confirmed condition for 14 of the 23 conditions it caught, including all 6 bearing defects; Vendor Rules alerts name only the measurement that crossed its limit.
  - Full evaluation and technical detail are in the [model report](https://bmdvs.github.io/predictive-maintenance-ml/reports/model_report.html).
- **Production monitoring.** A monthly monitor evaluates the ML Model against its retraining rules: catch share and alert rate are the primary triggers, and feature and score drift the secondary, alongside data-quality checks and Databricks data profiling. Over the evaluation year the primary trigger fired in September and October 2025, on catch shares of 0.33 and 0.50 against a 0.54 limit, so a full refit is recommended, as detailed in the [monitoring report](https://bmdvs.github.io/predictive-maintenance-ml/reports/monitoring_report.html).

The Vendor Rules' and ML Model's statuses and work requests are presented in a [maintenance queue](https://bmdvs.github.io/predictive-maintenance-ml/) in the shop's CMMS, one row per asset at ALERT or ALARM:

[![Maintenance queue](docs/screenshots/queue.png)](https://bmdvs.github.io/predictive-maintenance-ml/)

Clicking on an asset in the maintenance queue reveals a detail section showing measurement trends, fault-band energies, envelope spectra and controller signals, as well as a maintenance history table (not shown in the screenshot below):

[![Maintenance queue asset detail](docs/screenshots/queue_detail.png)](https://bmdvs.github.io/predictive-maintenance-ml/)

---

## Business Context

A precision machining shop runs 40 CNC machines. Its eight horizontal machining centers (HMCs) are the bottleneck: they carry 29% of the shop’s cutting hours at 71% utilization, and an hour of HMC downtime costs $185 in contribution margin, 1.7x the average for the rest of the fleet. Spindle and pump failures on the HMCs stop the shop's highest-margin work.

In July 2023 the shop began a pilot of sensor-based condition monitoring on the HMC cell and the compressors: wireless sensors on 40 points, monitored as 28 assets (each HMC's spindle, through-spindle coolant (TSC) pump and flood coolant pump, and each compressor). These sensors were monitored by the sensor vendor's threshold alerts, Vendor Rules, alongside interval-based PM. Vendor Rules was found to produce a high rate of false alerts: its ALERTs led to 204 machine inspections from July 2023 to June 2025, and 79% of those found nothing wrong.

To lower this false alert rate, and to catch developing faults earlier, the pilot's second phase added the ML Model. Built on the same sensor record, it ran in shadow mode beside Vendor Rules over the evaluation year to prove its effectiveness.

---

## Deliverables

| # | Deliverable | What it is | Links |
|---|---|---|---|
| 1 | Maintenance queue | The shop's CMMS maintenance queue: one row per asset at ALERT or ALARM under either method, with Vendor Rules' status, the ML Model's shadow status, the suspected condition and the work request. Each asset's detail section holds the ML Model score and measurement trends against Vendor Rules limits, fault-band energies, envelope spectra at the alert and now, controller signals and maintenance history. | [View](https://bmdvs.github.io/predictive-maintenance-ml/) |
| 2 | Model report | Detection and warning against Vendor Rules, escalation to ALARM and diagnosis; the model overview, training data detail, scoring summary and condition signatures; model selection, validation, false alerts and misses, and limitations. | [View](https://bmdvs.github.io/predictive-maintenance-ml/reports/model_report.html) |
| 3 | Monitoring report | The ML Model’s retraining status for the evaluation year: the retraining rules evaluated month by month; catch share, alert rate and false alerts by month; feature and score drift; data quality; and the champion model and post-repair re-centering. | [View](https://bmdvs.github.io/predictive-maintenance-ml/reports/monitoring_report.html) |

---

## Code

### Industrial Connectivity (MTConnect, Modbus, Ignition, MQTT UNS)

| File | What it does |
|---|---|
| [`replay/build_native.py`](connectivity/src/idc/replay/build_native.py) | Writes the HMCs' MTConnect SHDR logs and the compressors' Modbus register images from the controller record. |
| [`replay/shdr_adapter.py`](connectivity/src/idc/replay/shdr_adapter.py) | Serves each HMC's MTConnect SHDR log to the MTConnect agent, as the machine's adapter. |
| [`replay/modbus_server.py`](connectivity/src/idc/replay/modbus_server.py), [`replay/register_map.py`](connectivity/src/idc/replay/register_map.py) | Serves the compressor controllers' holding registers over Modbus TCP, with the register map for encoding and decoding. |
| [`bridge/mtconnect_to_mqtt.py`](connectivity/src/idc/bridge/mtconnect_to_mqtt.py) | Reads the MTConnect agent's sample stream and publishes each machine's record to the MQTT UNS. |
| [`ignition/`](connectivity/ignition/) | Ignition configuration: Modbus device, compressor UDT and tags, and MQTT Transmission publishing Sparkplug B to the MQTT broker. |
| [`landing/controller_writer.py`](connectivity/src/idc/landing/controller_writer.py), [`landing/minute_writer.py`](connectivity/src/idc/landing/minute_writer.py) | Lands the HMCs' 1 Hz controller stream from the UNS as Parquet, and builds its per-minute aggregates. |
| [`landing/compressor_writer.py`](connectivity/src/idc/landing/compressor_writer.py) | Decodes the compressors' Sparkplug B messages and lands them as Parquet. |
| [`checks/acceptance.py`](connectivity/src/idc/checks/acceptance.py), [`checks/opcua_readback.py`](connectivity/src/idc/checks/opcua_readback.py) | Compares the landed files with the bronze controller inputs row for row, and reads the compressor tags back over OPC UA. |
| [`run_chain.py`](connectivity/src/idc/run_chain.py) | Runs the chain end to end: replay, publish to the UNS, land and acceptance. |
| [`docker-compose.yml`](connectivity/docker-compose.yml) | Runs the MTConnect agent, the Mosquitto MQTT broker and the Ignition gateway. |

### Data Platform & MLOps (Azure Databricks) 

| File | What it does |
|---|---|
| [`databricks/scripts/`](databricks/scripts/) | Sets up the platform: Unity Catalog, landing the exports, service principal grants and the data profiles, as reproducible CLI scripts. |
| [`databricks.yml`](databricks.yml), [`databricks/`](databricks/) | Deploys the Declarative Automation Bundle: dev and prod targets, the Lakeflow pipeline and the monthly job. |
| [`src/pdm/entrypoints.py`](src/pdm/entrypoints.py), [`src/pdm/tasks.py`](src/pdm/tasks.py) | Runs the monthly job as Python wheel tasks: train, promote, score and monitor. |
| [`src/pdm/registry.py`](src/pdm/registry.py) | Promotes the champion and challenger in the Unity Catalog model registry, through the five promotion gates. |
| [`src/pdm/monitoring/performance.py`](src/pdm/monitoring/performance.py) | Monitors event performance per method, feature and score drift each month, and makes the retraining decision. |
| [`src/pdm/reports/`](src/pdm/reports/) | Publishes the queue and both reports from the gold snapshot in [`data/gold_snapshot/`](data/gold_snapshot/). |
| [`src/pdm/parity.py`](src/pdm/parity.py) | Tests the pipeline and every job task locally on the exports, checked against the workspace's tables. |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Lints and tests every change, and validates the bundle when workspace credentials are configured. |

### ML Model (training, scoring, alerting and evaluation)

| File | What it does |
|---|---|
| [`models/nbm.py`](src/pdm/models/nbm.py), [`models/nbm_pyfunc.py`](src/pdm/models/nbm_pyfunc.py) | Trains and scores the anomaly detection ML models (autoencoder, Mahalanobis distance, PCA, isolation forest), one for each sensor point, with inputs re-centered at each repair or remount, and wraps them in MLflow for the registry. |
| [`models/controller_health.py`](src/pdm/models/controller_health.py) | Scores the controller signals: spindle and axis load drift per program, the spindle thermistor residual and the compressor discharge temperature residual, for all 40 machines. |
| [`rules/`](src/pdm/rules/) | Alerting by Vendor Rules, replayed from the sensor platform export, and by the ISO 20816-3 zone rule, with the persistence and suppression every alerting layer shares. |
| [`advisory/requests.py`](src/pdm/advisory/requests.py) | Opens work requests at ALERT, with a suspected condition and recommended inspection from the measurement pattern. |
| [`evaluation/events.py`](src/pdm/evaluation/events.py) | Evaluates each method against confirmed conditions: catches, warning, and false alerts and false ALARMs per asset-month. |

---

## Architecture

### Industrial Connectivity

```mermaid
flowchart LR
  subgraph FLOOR["Field devices"]
    SENS["Wireless sensors<br/>40 points"]
    HMC["HMC controllers<br/>8 machines"]
    CMP["Compressor controllers<br/>4 compressors, room unit"]
  end
  subgraph CONN["Connectivity"]
    ADP["MTConnect adapters"]
    AGENT["MTConnect agent"]
    BRIDGE["MTConnect-to-MQTT<br/>bridge"]
    IGN["Ignition<br/>Modbus driver,<br/>MQTT Transmission"]
    UNS[("MQTT UNS<br/>Mosquitto")]
    WRT["Landing writers"]
  end
  subgraph BIZ["Business systems"]
    CMMS["CMMS<br/>work orders, PM records"]
    ERP["ERP<br/>jobs, asset register"]
  end
  VEND["Sensor vendor's<br/>cloud platform"]
  LAND[("Landing volume<br/>Parquet")]

  SENS -- "wireless" --> VEND -- "scheduled export" --> LAND
  HMC --> ADP -- "SHDR" --> AGENT --> BRIDGE -- "JSON" --> UNS
  CMP -- "Modbus TCP" --> IGN -- "Sparkplug B" --> UNS
  UNS --> WRT --> LAND
  HMC -. "event log export" .-> LAND
  CMMS -- "export" --> LAND
  ERP -- "export" --> LAND
```

The sensor data is transmitted wirelessly to the vendor’s cloud platform and exported. The HMC controller data is read over MTConnect, and the compressor controller data over Modbus into Ignition; both are published on an MQTT UNS and written to files. Together with the CMMS and ERP exports, these files land in a Unity Catalog volume, which feeds the data platform below.

### Data Platform & MLOps

```mermaid
flowchart LR
  LAND[("Landing volume")]
  subgraph PIPE["Lakeflow pipeline"]
    BRZ["Bronze<br/>Auto Loader streaming tables"]
    SLV["Silver<br/>hourly indicators, healthy windows,<br/>controller features, labels"]
    GLD[("Gold<br/>baselines, rolling features,<br/>rule floor")]
  end
  subgraph JOB["Monthly Lakeflow job"]
    TRAIN["Train<br/>ML models,<br/>controller health"]
    PROM["Promote<br/>Unity Catalog registry,<br/>champion alias, five gates"]
    SCORE["Score<br/>per-point scores,<br/>advisory"]
    MON["Monitor<br/>evaluation, drift, data quality,<br/>retraining decision"]
  end
  LAND --> BRZ
  BRZ --> SLV --> GLD
  GLD --> TRAIN --> PROM --> SCORE
  SCORE --> CASES[("Cases and<br/>work requests")]
  CASES --> MON
  CASES --> QUEUE["Maintenance queue"]
  MON --> REP["Model and<br/>monitoring reports"]
```

A Lakeflow pipeline loads the landed files into bronze tables and builds the silver and gold tables, with data-quality expectations on each. Along the way, each hourly vibration burst is reduced to condition indicators, and each point's healthy windows are set from the asset register and CMMS. The monthly job then:
- trains the challengers;
- promotes one to champion in the Unity Catalog registry when it passes its gates;
- scores every point from the champion and opens work requests from its ALERTs;
- monitors performance, drift and data quality.

The pipeline and job are deployed from a Declarative Automation Bundle, and GitHub Actions lints and tests every change.

---

## Data Generation

The data is generated from each machine’s physical properties and published degradation mechanisms to reflect the behavior of real equipment. The faults follow those mechanisms alone and were not shaped to suit any monitoring method.

- **Healthy operation.** Each point's healthy signal is built from its physical properties:
  - vibration bursts synthesized at 25,600 samples a second from the point's kinematics, with shaft speed from the controller and bearing defect frequencies from the asset register;
  - shaft-order energy at 1×, 2× and 3× running speed, and lobe-pass energy on the compressor airends;
  - structural noise shaped by the point's own housing resonance, between 4 and 8 kHz;
  - levels that shift with spindle speed and rise with cutting load, following the ERP's jobs and programs;
  - housing temperature that rises over room temperature by asset type, lags its load and follows season;
  - a new healthy level after each sensor remount, from a small shift in the sensor's gain and resonance.
- **Degradation.** Each condition develops through its published signature:
  - bearing defects through the four degradation stages, from envelope peaks at the fault frequency to rising kurtosis and, late, velocity;
  - lubrication loss as rising high-frequency energy and temperature;
  - pump seal and impeller wear as energy at running speed and broadband;
  - rotary-union leaks as a drop in through-spindle coolant pressure;
  - spindle cooling faults as a temperature rise.
- **Maintenance and the record.** Inspections confirm a condition or find nothing; PMs, repairs and remounts reset levels, and failures stop machines. CMMS failure codes carry a real record's miscoding rate, and the sensor data carries idle periods and implausible rows.
- **Operating events that are not faults.** Heavy cuts, titanium jobs, new programs and coolant drift move the measurements without any condition behind them.

The record covers July 2023 to June 2026 as typical exports from a wireless vibration monitoring platform, CNC and compressor controllers, a CMMS and an ERP:
- hourly sensor scalars and retained vibration bursts;
- controller 1 Hz stream, minute aggregates and events;
- compressor controller data;
- CMMS work orders and PM records;
- ERP jobs and the asset register.

The connectivity chain replays the controller data in its native device formats (MTConnect SHDR logs for the HMCs, Modbus register images for the compressors), written from this record.

Realism checks test each mechanism against its published behavior, and the record was frozen before any model was trained. The generators are in [`generate/`](generate/) and the checks in [`generate/checks/`](generate/checks/).

---

## Running it locally

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,spark]"
pip install -e connectivity

# 2. Generate the source record, run the realism checks and write the run report
python -m generate

# 3. Replay the controller data through the connectivity chain and verify the landed files
cd connectivity
python -m idc.replay.build_native       # MTConnect SHDR logs and Modbus register images
docker compose up -d                    # MTConnect agent, MQTT broker, Ignition gateway
python -m idc.run_chain                 # replay, publish to the UNS, land Parquet files and run acceptance
python -m idc.checks.acceptance         # rerun acceptance on its own, row for row against the bronze inputs
cd ..

# 4. The pipeline and every job task on the local exports
python -m pdm.parity

# 5. The queue and the reports from the gold snapshot
python -m pdm.reports.build

# 6. Lint and tests
ruff check . && pytest
(cd connectivity && pytest)
```

Ignition's gateway setup is described in [`connectivity/ignition/`](connectivity/ignition/).

On the platform, from a configured Databricks CLI:

```bash
databricks bundle validate -t prod
databricks bundle deploy -t prod
databricks bundle run -t prod pdm_monthly --no-wait
```

The page build writes standalone HTML to [`docs/`](docs/), which GitHub Pages serves.

---

## Stack

| Layer | Tools |
|---|---|
| Industrial connectivity | MTConnect (cppagent), Modbus TCP (pymodbus), Ignition (Modbus driver, OPC UA, MQTT Transmission), MQTT (Mosquitto, Sparkplug B), Docker |
| Platform | Azure Databricks, Unity Catalog, Azure Data Lake Storage Gen2 |
| Ingestion & transformation | Lakeflow Declarative Pipelines (Auto Loader, materialized views, expectations), PySpark, Delta Lake |
| Orchestration & deployment | Lakeflow Jobs (Python wheel tasks), Declarative Automation Bundles, GitHub Actions, service principal |
| Signal processing | NumPy, SciPy (filtering, envelope analysis, spectra) |
| Modeling | scikit-learn (autoencoder, PCA, isolation forest, Mahalanobis distance) |
| Model tracking & registry | MLflow, Unity Catalog model registry |
| Monitoring | SciPy (Wasserstein distance), Databricks data profiling |
| Data generation | Python, NumPy, pandas |
| Reporting | matplotlib, HTML/CSS, GitHub Pages |

---

**Author:** Brian Davis. Data engineering and applied analytics/ML for manufacturers.

**Link:** https://github.com/bmdvs

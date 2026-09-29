# Page text

The text of the pages, assembled with their exhibits by pdm.reports.build.

## A. Maintenance queue page (`docs/index.html`)

**Title:** Condition Monitoring Queue

**Line under the byline:** Status as of June 30, 2026, 23:00. 34 monitored assets: 20 machines by controller data, with sensor coverage on the HMC cell and both compressors.

**Queue header:** 4 assets in ALARM and 30 OK. One OK asset, HMC-03-SP, still has an open case.

**Column labels:** Asset, Class, Status, Case opened, Days open, Suspected condition, Points, Triggering layer, Request id, Request status.

**Status legend:**
- ALARM: persistent evidence of a developing condition;
- WATCH: elevated, not persistent;
- OK.

**Legend note:** "A case opens at ALARM and closes at a repair work order or after 7 quiet days, so an asset can return to OK with its case still open."

**Triggering layer labels:**
- Vendor rules
- Per-point model
- Controller health

**Request status label:** "Open (shadow): logged, not sent to maintenance".

**Detail panel section labels** (as 13.3), in order:
1. Case summary
2. Why flagged
3. Trend, 90 days
4. Spectral evidence
5. Controller context
6. Remaining time
7. History

**Fixed panel lines:**
- **Remaining time, where no estimate exists:** "No remaining-time estimate is offered for this asset."
- **Spectral evidence, where no burst exists at the first alarm:** "Nearest retained burst: {offset} days from the first alarm." The offset comes from the panel data.
- **Coolant pressure:** "Through-spindle coolant pressure while the pump runs. Rating 69 bar."
- **Asset with no open case:** "No open case."
- **HMC-03-SP:** "Status OK; case open until a repair work order or 7 quiet days close it."

---

## B. Model report (`docs/reports/model_report.html`)

**Title:** Predictive Maintenance Model Report

**Subtitle:** Shadow-mode evaluation, April to June 2026, with results for July 2025 to June 2026

### 1. Summary

Over the April to June 2026 shadow quarter, the model layer warned before 3 of the 4 conditions confirmed on the 14 instrumented assets, against 2 of 4 for the vendor's alert rules. Its false-alert rate was similar: 0.22 false episodes per asset-month against 0.19. Over the evaluation year, July 2025 to June 2026, it caught 6 of 12 conditions against 4 of 12, with fewer false episodes (0.19 against 0.21) and a longer median warning (34.8 against 31.8 days). In the quarter alone, its median warning was shorter (18.9 against 25.5 days).

The model layer's additional catches were a coolant pump seal and impeller failure and an HMC spindle cooling fault. The vendor rules caught neither type in the year. On machine components without sensors, controller data caught 1 of 6 conditions: a VMC spindle bearing, 9.3 days before an 80-hour stoppage.

The difference between the two methods rests on two conditions, so it is consistent with the model layer's advantage but too small to establish it. Work requests named the confirmed cause for 3 of 7 conditions.

Through-spindle coolant pressure, measured while the pump runs, fell 5 to 10 percent ahead of each of the three rotary-union leaks in the record. The advisory reads a pressure series that hides those drops. Moving it to the pump-on series is the highest-value change identified (section 2.1).

**Exhibit: Shadow quarter, model layer beside vendor alert rules**

Tiles:

| Tile | Model layer | Vendor rules |
|---|---|---|
| Conditions caught | 3 of 4 | 2 of 4 |
| Median warning | 18.9 days | 25.5 days |
| False episodes per asset-month | 0.22, or 0.14 with outcome known | 0.19, or 0.10 with outcome known |
| Work requests | 11 raised, 2 confirmed, 4 open at the record end | |

### 2. Model results

#### 2.1 Detection and warning

The model layer caught more conditions than the vendor rules in the shadow quarter and over the year, at a false-alert rate no higher than theirs, and its additional catches came from condition types the vendor rules did not catch. A condition counts as caught when an alert or case was open on the asset at confirmation and had opened within the 60 days before it.

In the quarter, the model layer opened an alert on the HMC-04 spindle cooling fault 35.8 days before the cooler repair. Neither the vendor rules nor controller health raised one. Both methods caught the HMC-03 lubrication episode and the HMC-04 coolant pump imbalance, and the vendor rules warned earlier on both (30.0 and 21.0 days, against 18.9 days each). Every method missed the HMC-01 spindle cooling fault at its operating point; the vendor rules and controller health would have caught it at looser settings.

**Exhibit: Conditions confirmed in the shadow quarter, alert periods by method**

The event timeline as specified in 13.4.

**Exhibit: Warning days and miss reasons, shadow quarter**

The compact table as specified, including the two conditions on uninstrumented components: the VMC-06 spindle bearing, caught by controller health with 9.3 days' warning, and the HMC-04 Y-axis stoppage, which had no precursor.

At the vendor rules' false-alert rate for the year, 0.214 per asset-month:
- the per-point models caught 58 percent of conditions;
- the fused model layer caught 50 percent;
- the vendor rules caught 33 percent.

Chance at that rate is about 20 percent for each. In validation alone, the per-point models caught 50 percent and the vendor rules 25 percent.

**Exhibit: Conditions caught against false episodes per asset-month, by method**

Over the year, warning on conditions caught by the vendor rules, the per-point models and the model layer ran from 18.8 to 49.5 days. Controller health's two catches came 9.3 and 10.3 days ahead. On the two conditions both methods caught in validation, the model layer warned about a day earlier (34.7 against 34.0 days on the HMC-04 flood pump bearing, and 34.9 against 33.7 days on the CMP-01 imbalance). In the shadow quarter, the vendor rules warned earlier.

**Exhibit: Warning days per caught condition, by method**

On the HMC cell, every catch in the year involved the vibration sensors. The sensor layers caught 6 of the 10 conditions on instrumented components. Controller data caught 1, the lubrication episode, which the sensors also caught. The 4 misses were:
- the two rotary-union leaks;
- a TSC pump seal failure;
- the HMC-01 spindle cooling fault.

On machine components without sensors, controller data caught 1 of 6 conditions: the VMC-06 spindle bearing, 9.3 days ahead of an 80.2-hour stoppage. The other five were four machine stoppages with no precursor and a crash.

**Exhibit: Conditions by the layers that caught them, HMC cell and uninstrumented machines**

Both rotary-union leaks were missed because the pressure series the advisory reads does not show them.
- **Pump-on pressure.** Measured while the pump runs, through-spindle pressure holds at the pump's 69 bar rating. It fell 3.5 to 6.8 bar (5 to 10 percent), starting 5 to 23 days before each of the three leak work orders in the record.
- **The advisory series.** The advisory reads a cutting-time average that includes minutes with the pump off. That series varies 3.7 to 4.6 bar from day to day, as much as the leak drops, so no work request's pressure-drop score reached the alert level of 3.
- **The change.** Move the pressure-drop signal to the pump-on series, with a scale suited to its very low day-to-day spread, and pass it to fusion so that it can open a case. That is the highest-value change this evaluation identified.

**Exhibit (NEW): Through-spindle coolant pressure while the pump runs, before each rotary-union leak**
- One small panel per leak: HMC-03 December 2023, HMC-03 October 2025, HMC-04 January 2026.
- Each panel shows the daily pump-on pressure for the 60 days before the work order, with the low span shaded and the work order opening marked, and the advisory's duty-weighted series on the same axis in a muted color.
- Source: the package's leak table and `gold.tsc_pressure_daily`, plus silver for the advisory series.

The year's results rest on 12 conditions and the quarter's on 4, so a two-condition difference between methods is within what chance could produce.

#### 2.2 Work requests and diagnosis

Most work requests closed without a recorded finding, and the suspected cause matched the confirmed one on 3 of 7 conditions. Requests work as a prompt to inspect, with the suspected condition as a starting point.

In validation, 25 of 29 requests closed with no finding recorded and 4 led to confirmed findings. In the shadow quarter, 2 of 11 led to confirmed findings and 4 were still open at the record end.

**Exhibit: Work requests by outcome**

At the vendor rules' false-alert rate, false alerts cost about 3.6 inspection hours a month across the 14 instrumented assets, at a median 1.19 hours per inspection. Over the year:
- the model layer's 32 false episodes represent 38.1 inspection hours;
- the vendor rules' 36 represent 42.8.

Of the false episodes in the quarter, 3 of the model layer's and 4 of the vendor rules' opened in June and were still inside their 30-day window at the record end. They include the open cases on CMP-01 and HMC-01-TSC. With those excluded, the quarter's rates are 0.14 and 0.10.

**Exhibit: Detection at the vendor rules' false-alert rate, with inspection hours and outcome-pending episodes**

Requests correctly identified:
- the CMP-01 imbalance;
- the HMC-01 coolant pump seal and impeller wear;
- the HMC-04 spindle cooling fault.

They misread:
- two bearing failures, as cavitation or wear on the HMC-04 flood pump and as spindle cooling on VMC-06;
- the HMC-04 flood pump imbalance, as cavitation or wear;
- the HMC-03 lubrication episode, as spindle cooling.

Some CMMS finding codes are assigned inconsistently, so agreement is approximate.

**Exhibit: Suspected condition against confirmed finding, July 2025 to June 2026**

#### 2.3 Component life and remaining time

Component failure rates rise with service time, which supports age-based replacement. But 9 failures across 39 component lives are too few to set intervals from, and remaining-time estimates were not accurate enough to plan from.

Across compressors, flood coolant pumps, TSC pumps and spindles, the fitted Weibull shape is 1.91 (90 percent interval 1.22 to 2.98): the failure rate increases with service time. Median lives run from about 920 days for TSC pumps to about 1,730 days for spindles, with 90 percent intervals spanning a factor of 2.4 to 5.6 by class. Machine age, crash count and lubrication compliance did not improve the fit over the fleet curve.

At the TSC pumps' two-year replacement interval, the fitted curve places 36 percent of pumps failing before replacement. That estimate rests on 2 failures in 15 lives; it is reviewed again with another year of record.

**Exhibit: Survival by component class, fleet Weibull fit against the empirical curves**

**Exhibit: Component lives, failures and median life by class**

Remaining-time estimates were tested on the three pump-bearing failures that had an open alert, at 25, 50 and 75 percent of the time from detection to failure.
- None of the 9 estimates fell within 20 percent of the actual remaining life.
- The prediction interval contained it in 6 of 9.
- On the held-out HMC-04 flood pump, velocity stayed near 1.9 mm/s until about a week before failure, so the early estimates ran long.

Remaining-time estimates are not offered for planning, and none is offered for spindles.

**Exhibit: Remaining-time estimates against the actual failure, HMC-04 flood coolant pump**

### 3. Model overview

#### 3.1 What the system does

The system reads three sources:
- hourly vibration and temperature measurements from 20 wireless sensor points on the HMC cell and the two compressors;
- controller data from all 20 machines;
- maintenance history from the CMMS.

It scores each hour against three layers: the vendor's alert rules, a normal-behavior model per sensor point, and controller-based health models. It combines the results into one status per asset (OK, WATCH or ALARM), opens a case at ALARM, and raises a work request with the suspected condition and a recommended inspection.

Scoring ran monthly on Azure Databricks from April to June 2026 in shadow mode: requests were logged, not sent to maintenance.

**Exhibit: Inputs, scoring and outputs**

The queue lists every monitored asset with its status, open case and work request. Each asset opens a detail panel with indicator trends, spectral evidence and history.

**Exhibit: Condition monitoring queue, June 30, 2026** (screenshot)

#### 3.2 Monitored assets and signals

- **HMCs:** each carries four sensor points, at the spindle front and rear bearing housings, the through-spindle coolant pump and the flood coolant pump.
- **Compressors:** each carries two, at the motor drive end and the airend inlet bearing.
- **Measurement:** every point records a one-second, three-axis vibration burst each hour, which is reduced to condition indicators.
- **Controllers:** all 20 machines supply spindle and axis loads, the spindle thermistor and, on the HMCs, through-spindle coolant pressure.

**Exhibit: Sensor points by asset**

**Exhibit: HMC spindle sensor positions**

**Exhibit: Controller data by source**

#### 3.3 Training data

The record runs from July 2023 to June 2026. The first two years trained the models. Nine months of validation, July 2025 to March 2026, were used to select and set them. The April to June 2026 shadow quarter was held out until the models were final.

The record holds 64 confirmed conditions: 46 in training, 12 in validation and 6 in the shadow quarter. Lubrication episodes (14) and machine stoppages (9) make up half of training. Each sensor point's normal-behavior model is trained on its healthy hours, restarted after each repair and sensor remount.

**Exhibit: Record periods**

**Exhibit: Confirmed conditions by type and period**

**Exhibit: Healthy hours per sensor point**

#### 3.4 Conditions and their signatures

Condition types, where each appears, and the layers that caught them in validation and the shadow quarter.

**Exhibit: Conditions, signals and catching layers** (descriptive table, as 13.4)

### Appendix

**A. Scoring layers**
- **Vendor rules:** per-point alerts at baseline plus a share of the ISO 20816 zone B width on velocity and high-frequency acceleration, an envelope fault-band rule, and ISO zone boundaries. These are the alert rules the shop used through the record.
- **Per-point models:** one normal-behavior model per sensor point and segment, trained on healthy hours. Each hour gets an anomaly score against a threshold at the 99.5th percentile of healthy scores, with six-hour persistence.
- **Controller health:** per-program spindle and axis load drift, a spindle thermistor residual against speed and season, and a compressor discharge temperature residual against room temperature and load.
- **Fusion and cases:** signals from the three layers within 72 hours combine into one status per asset. A case opens at ALARM and closes at a repair work order or after 7 quiet days.
- **Advisory:** a work request per case, with the suspected condition from the indicator pattern and a recommended inspection.
- **Survival:** a fleet Weibull model of component lives for PM interval review.
- **Prognosis:** a remaining-time model on pump and compressor bearings from detection to the ISO zone C/D velocity boundary.
- **Supervised layer:** gated on event count (appendix F).

Sensitivity: the bearing-wear-out-only survival fit gives a shape of 2.12 (1.26 to 3.58).

**Exhibit: Scoring flow**

**B. Condition indicators**

Indicators computed from each hourly burst, with the band and what each measures.

**Exhibit: Indicators**

**C. Evaluation definitions**

**Exhibit: Definitions** (from the package's appendix C table)

**D. Model selection**

Four normal-behavior candidates were trained per point and compared on the training period, before validation, at their operating points:
- an autoencoder on one-third-octave bands, which caught 47 percent of training conditions at 0.064 false episodes per asset-month;
- Mahalanobis distance, at 35 percent;
- PCA, at 29 percent;
- isolation forest, at 21 percent.

The autoencoder was selected by the rule set before validation. Thresholds come from each point's healthy hours and were not adjusted to results.

**Exhibit: Candidate models on the training period**

**Exhibit: MLflow run comparison** (screenshot)

**E. Model governance**
- **The cycle.** Each monthly cycle trains a challenger and registers it. It becomes the champion only if it passes four gates: the selection rule's pick, the registered model reproducing training's scores (to 1e-9 relative), every fitted segment present, and false episodes on the latest month within twice the validation rate.
- **The record.** Versions 5, 6 and 7 were promoted in April, May and June. May's false-episode rate, 0.47, came closest to the 0.51 budget.

**Exhibit: Registered model versions and gate results**

**Exhibit: Registered model page** (screenshot)

**F. Supervised layer**

The training period holds 6 failures on the rotating auxiliaries, against a minimum of 8 for training a classifier. No classifier was trained, so no leakage audit applies.

**G. Validation results, July 2025 to March 2026**

**Exhibit: Validation headline and results by condition type**

**H. Platform**

The system runs on Azure Databricks with Unity Catalog:
- a Lakeflow pipeline from landed exports to bronze, silver and gold tables, with data-quality expectations;
- a monthly Lakeflow job: pipeline update, training, promotion, scoring, monitoring;
- MLflow experiments and the Unity Catalog model registry;
- Databricks data profiling for monitoring.

It is deployed from a Declarative Automation Bundle with development and production targets. GitHub Actions validates every pull request and deploys production as a service principal. Every cloud result was checked against a local run of the same code.

**Exhibit: Architecture**

**Exhibit: Pipeline graph** (screenshot)

**Exhibit: Lineage of the cases table** (screenshot)

**Exhibit: Job run history** (screenshot)

**I. Limitations**

**Exhibit: Limitations** (the package's appendix I table, plus the following rows)
- Small counts: 12 conditions in the year, 4 in the quarter, none in June.
- Acted-on alerts remove the failures that would label them.
- CMMS finding codes are assigned inconsistently on some work orders.
- Crashes have no precursor.
- The supervised layer was not trained.
- Remaining-time estimates did not meet the accuracy target.

---

## C. Monitoring report (`docs/reports/monitoring_report.html`)

**Title:** Model Monitoring Report

**Subtitle:** April to June 2026

### 1. Summary

The monitoring rule calls for a review and full refit of the model layer after May and of controller health after May and June. Both verdicts come from alert rates measured against low validation-period rates, where one alert more or less moves the ratio across the 0.5 to 2.0 band:
- the model layer's rate moved from a third of its reference in April to 2.3 times it in May;
- controller health's reference is 0.017 alerts per asset-month.

The drift trigger fired in April and June without changing the verdict. Each month's challenger passed its promotion gates, and versions 5 to 7 became champion in turn. A full refit is a review step and was not run within the record.

**Exhibit: Monitoring by month, model layer**

#### 2.1 Performance and alert rates

The model layer caught both conditions in May and one of two in April; June had no confirmed conditions. Its alert rate was:
- 0.07 per asset-month in April;
- 0.50 in May, when 6 of its 7 alerts were not followed by a confirmed condition;
- 0.21 in June.

The vendor rules stayed within their band all quarter, at 0.21 to 0.29.

**Exhibit: Alerts per asset-month against the validation rate and its tolerance band**

**Exhibit: Event performance by month and method**

#### 2.2 Drift and data quality

Feature drift flagged 8, 5 and 8 of the 20 points in April, May and June. The largest distances were on the CMP-01 airend, which carries an open bearing case at the record end, and the HMC-04 flood pump.

The drift trigger fired for two reasons:
- **April:** points whose post-repair references were still forming left the comparison, which raised the drifted share among the rest.
- **June:** score drift on four points whose post-repair references are narrow.

Data-quality exceptions rose from 9 to 15 point-months over the quarter, 35 in all. Most are isolated implausible rows spread across 18 points; the standby compressor, which takes no measurement while idle, accounts for 6.

**Exhibit: Feature drift by point and month**

**Exhibit: Data quality by point and month**

**Exhibit: Data profiling dashboard** (screenshot)

#### 2.3 Model updates and baseline resets

- **Model versions.** Versions 5, 6 and 7 passed all four gates and were promoted in April, May and June. May's false-episode rate was 0.47 against a 0.51 budget.
- **Baseline resets.** Five points started new baselines after repairs in the quarter: both HMC-01 spindle points, both HMC-04 spindle points, and the HMC-04 flood pump.

**Exhibit: Champion per monthly cycle, with gate results**

**Exhibit: Post-repair baseline resets**

### Appendix
- the retraining rule as defined, including the action a retrain verdict triggers;
- drift reference definitions, including the window-only points;
- per-point tables.

Acted-on alerts remove the failures that would label them, so recall in a live deployment reads lower than the condition record supports.

---

## D. README

The sections follow the 13.6 order.

**Title:** Predictive Maintenance ML: Condition Monitoring for a Precision Machining Shop

**Summary**

A condition-monitoring and predictive maintenance system for a 20-machine contract machining shop, built end to end on Azure Databricks. It combines:
- hourly vibration measurements from 20 wireless sensor points on the HMC cell and the plant compressors;
- controller data from every machine;
- maintenance history from the CMMS.

From these it scores vendor alert rules, per-point normal-behavior models and controller-based health models. It fuses them into one status per asset, opens cases, and raises work requests with a suspected condition.

Evaluated in shadow mode over April to June 2026:
- the model layer warned before 3 of 4 conditions on the instrumented assets, against 2 of 4 for the vendor rules;
- over July 2025 to June 2026, it caught 6 of 12 against 4 of 12, at a lower false-alert rate.

Included:
- the maintenance queue;
- a model report;
- a monitoring report;
- the full pipeline, model and deployment code.

(Screenshots: `queue.png`, `queue_detail.png`.)

**Business Context**

The shop machines precision parts in the northwest suburbs of Chicago on 20 CNC machines. Its four-machine HMC cell is the bottleneck. In July 2023 it installed wireless vibration sensors on the HMC cell and its two compressors, and through June 2026 it maintained them on the sensor vendor's threshold alerts and interval-based PM.

Spindle and pump failures on the HMC cell stop the shop's highest-margin work. Vendor alerts raised inspections that most often found nothing. The shop built a model layer on the three-year record to catch more developing faults earlier, at no more inspection workload, and evaluated it alongside the vendor alerts for a quarter before acting on it.

**Models and methods**

- **Rule floor:** the vendor's baseline alerts and ISO 20816 zones.
- **Per-point normal-behavior models:** an autoencoder on one-third-octave vibration bands, selected from four candidates.
- **Controller health:** load drift and thermal residual models.
- **Fusion:** a case per asset.
- **Advisory:** rule-based diagnosis.
- **Survival:** fleet Weibull models for PM review.
- **Remaining time:** for pump and compressor bearings.
- **Evaluation:** event-based against confirmed conditions, with chance levels and false episodes per asset-month.
- **Model governance:** a monthly champion and challenger cycle with automated promotion gates.

**Data**

- **The record:** three years, July 2023 to June 2026. It holds hourly sensor scalars, retained vibration bursts, controller minute aggregates and events, compressor controller data, CMMS work orders and PM records, ERP jobs and the asset register.
- **How it was generated:** from degradation and failure mechanisms published for rolling bearings, pumps, compressors and machine tools, including bearing defect signatures, four-stage degradation, lubrication loss, crash damage and coolant system faults. Realism checks test the published behavior.
- **Why:** the record is generated so the full workflow can be shared.

**Architecture**

- **Platform:** Azure Databricks with Unity Catalog, organized as landing, bronze, silver, gold, ml and monitoring schemas.
- **Pipeline:** a Lakeflow Declarative Pipeline (Auto Loader ingest, materialized views, expectations).
- **Job:** a monthly Lakeflow Job: pipeline update, train, promote, score, monitor.
- **Models:** MLflow experiments and the Unity Catalog model registry, with a `champion` alias.
- **Monitoring:** Databricks data profiling.
- **Deployment:** `databricks bundle deploy -t prod` from a Declarative Automation Bundle with dev and prod targets. GitHub Actions lints, tests and validates every pull request and deploys production as a service principal.
- **Cost:** about $4 in Azure infrastructure through October 26, 2026, with Databricks usage covered by the trial allowance.

(Architecture diagram; pipeline graph and job history screenshots.)

**How to run**

As 13.6:
- locally: the parity runner and the page build;
- on the platform: the bundle deploy and the job.

**Author:** Brian Davis. Data engineering and applied analytics/ML for manufacturers.

**Link:** https://github.com/bmdvs

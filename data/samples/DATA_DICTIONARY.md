# Data dictionary

Columns of the source exports in `data/bronze/`. Every table also carries `run_id` and `config_hash`. Timestamps are UTC.

## cmms/work_orders.parquet

One row per work order.

| Column | Type | Description |
|---|---|---|
| `wo_id` | string | Work order id. PM work orders are `PM-nnnnnn`; sensor installs and remounts are `PRJ-<point>-<date>` |
| `asset_id` | string | Asset: machine (`HMC-01`), spindle (`HMC-01-SP`), pump (`HMC-01-TSC`, `HMC-01-CP`) or compressor (`CMP-01`) |
| `component` | string | Component from the fixed list (`SPINDLE`, `TSC_PUMP`, `COOLANT_PUMP`, `COMPRESSOR`, `ATC`, `HYDRAULIC`, `SERVO`, `ELECTRICAL`, `LUBE`, `WAY_COVER`, `CHIP_CONVEYOR`, `AXIS`); null where none applies |
| `wo_type` | string | `unplanned_repair`, `condition_based`, `corrective`, `inspection`, `pm`, `project`, `rebuild_order` |
| `failure_code` | string | Failure or finding code from the fixed list per component (`config.yaml`, `cmms.failure_codes`), uppercase and hyphenated with a component or condition prefix: spindle `BRG-WEAR`, `BRG-SPALL`, `BRG-LUBE`, `CRASH-DMG`, `SP-COOLANT-INGRESS`, `SP-COOLER`; rotary union `RU-LEAK`; pumps `PMP-SEAL`, `PMP-BRG`, `PMP-IMP`, `PMP-CAV`; compressors `CMP-BRG-MOTOR`, `CMP-BRG-AIREND`, `CMP-TRIP`, `CMP-VALVE`, `CMP-OILCOOL-FOUL`, `CMP-LUBE`; vibration conditions on pumps and compressors `VIB-IMBAL`, `VIB-MISALIGN`, `VIB-LOOSE`; other machine components as listed in the config. Repairs carry the failure code; confirmed inspections and PM findings carry the finding code (spindle lubrication findings `BRG-LUBE`); `PMP-CAV` and `CMP-LUBE` appear only on inspection and PM findings; null on inspections with no fault found, inspections still open at the record end, routine PM and projects. Unplanned repairs carry the shop's label noise: some codes are wrong and some are missing |
| `opened` | timestamp | When the work order was opened |
| `started` | timestamp | When work started |
| `closed` | timestamp | When the work order was closed in the CMMS (end-of-week batch closes lag the work); null while open |
| `downtime_hours` | double | Asset downtime recorded on the work order |
| `technician` | string | Technician id |
| `parts` | string | Parts used, semicolon-separated |
| `labor_hours` | double | Technician labor |
| `notes` | string | Free text: the reason for the work and what was found and done |
| `no_fault_found` | boolean | Inspections: true when the inspection found nothing, false when it confirmed a condition, null while the inspection is open. False on all other work orders |
| `platform_alert_id` | string | The condition-monitoring platform alert that raised the inspection; null otherwise |
| `source_wo_id` | string | The work order this one follows from: on a condition-based repair, the inspection that found the condition; on a corrective repair found at a PM check, that PM work order; on a rebuild order, the spindle repair it belongs to. Null otherwise |

## register

| Table and column | Type | Description |
|---|---|---|
| `machines.live_tooling` | boolean | True for lathes with live tooling; null where it does not apply |
| `compressors.motor_bearing_replaced` | string | Date of the last motor bearing replacement before the record (YYYY-MM-DD); null if none |
| `compressors.airend_rebuilt` | string | Date of the last airend rebuild before the record (YYYY-MM-DD); null if none |
| `sensors.radial_axis` | string | Sensor axis reported as the radial axis on spindle points (`x` or `y`); null on points that report all three axes |

# Data samples

A small extract of the source exports in `data/bronze/`, copied as exported, to show their shape without the
full record. The full exports are generated locally (`python -m generate`) and are not in the repository.

| File | Contents |
|---|---|
| `cm_platform/scalars_2026-06-15.parquet` | Hourly sensor scalars for all 20 measurement points on 2026-06-15 (shop local day), every reported axis and the resultant: 936 rows |
| `cm_platform/burst_index.parquet` | Index rows for the five retained bursts below (point, hour, reason, sampling rate, speed, path, run id and config hash) |
| `cm_platform/bursts/CMP-01-AE/20260311T16.npz` | Compressor airend, weekly reference burst |
| `cm_platform/bursts/CMP-01-AE/20260630T00.npz` | Compressor airend, latest burst before the record end |
| `cm_platform/bursts/HMC-04-SP-F/20260429T20.npz` | HMC spindle front point, weekly reference burst |
| `cm_platform/bursts/HMC-04-SP-F/20260624T11.npz` | HMC spindle front point, latest burst before the record end |
| `cm_platform/bursts/HMC-03-CP/20260627T12.npz` | Flood coolant pump, latest burst before the record end |
| `controller/minute_agg_HMC-01_2026-06.parquet` | Controller per-minute aggregates for HMC-01, June 2026, by program and execution state: 24,486 rows |

Each burst holds one second of triaxial acceleration (`accel_g`, g) at `fs_hz` samples per second. Hours are
stored in UTC.

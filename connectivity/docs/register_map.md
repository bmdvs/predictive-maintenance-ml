# Modbus register map

Holding registers (function code 3), zero-based wire addresses; the 4xxxx column is the one-based form used in vendor manuals. Registers are big-endian; 32-bit values are two registers, high word first. Scaled values decode as raw / scale.

## Unit IDs

| Device | Unit ID | Block registers |
|---|---|---|
| CMP-01 | 1 | 8 |
| CMP-02 | 2 | 8 |
| CMP-03 | 3 | 8 |
| CMP-04 | 4 | 8 |
| room | 5 | 5 |

## Compressor block (CMP-01 to CMP-04)

| Offset | 4xxxx (single block) | Field | Type | Scale | Unit | Note |
|---|---|---|---|---|---|---|
| 0-1 | 40001-40002 | record_ts | uint32 | 1 | s | record minute, UTC epoch seconds |
| 2 | 40003 | state | uint16 | 1 |  | 0 off, 1 loaded, 2 unloaded |
| 3 | 40004 | discharge_pressure_bar | uint16 | 100 | bar | 0.01 bar resolution |
| 4 | 40005 | discharge_temp_c | int16 | 10 | C | 0.1 C resolution, signed |
| 5-6 | 40006-40007 | run_hours | uint32 | 100 | h | cumulative, 0.01 h resolution |
| 7 | 40008 | snapshot | uint16 | 1 |  | block sequence counter; wraps at 65536 |

## Room and dryer block

| Offset | 4xxxx (single block) | Field | Type | Scale | Unit | Note |
|---|---|---|---|---|---|---|
| 0-1 | 40001-40002 | record_ts | uint32 | 1 | s | record minute, UTC epoch seconds |
| 2 | 40003 | ambient_c | int16 | 10 | C | 0.1 C resolution, signed |
| 3 | 40004 | dryer_dew_point_c | int16 | 10 | C | 0.1 C resolution, signed |
| 4 | 40005 | snapshot | uint16 | 1 |  | block sequence counter; wraps at 65536 |

## Ring-buffer layout

Each unit exposes 15 slots of 8 registers: holding registers 0 to 119, read in one request. Slot s occupies registers 8*s to 8*s+7; offsets within a slot follow the block tables above. Room slots use offsets 0 to 4; offsets 5 to 7 read as zero.

A record minute m is written to slot `minute_index(m) mod 15`, where `minute_index` counts whole minutes from 2026-01-01T00:00:00+00:00. The snapshot register holds `minute_index(m) mod 65536`, so record_ts and snapshot together identify the minute a slot holds.

Each server tick writes the next N record minutes into their slots for all units at once, then holds. With N minutes per tick a slot is overwritten every 15/N ticks, so each minute stays readable for that many ticks.

## Reading rules

- Read a unit's registers in one request; the server swaps all blocks atomically.
- A slot belongs to the minute in its record_ts registers, not to the time it was read. Accept a slot only when record_ts and snapshot agree.
- int16 fields: 0x8000 to 0xFFFF are negative.
- Unscale by dividing by the scale.

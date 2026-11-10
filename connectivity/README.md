# Industrial connectivity

This folder holds the connectivity chain that replays the HMC and compressor controller data through MTConnect, Modbus TCP, Ignition and the MQTT UNS and lands it as Parquet, verified row for row against the bronze controller exports.

```bash
cd connectivity
python -m idc.replay.build_native       # MTConnect SHDR logs and Modbus register images
docker compose up -d                    # MTConnect agent, MQTT broker, Ignition gateway
python -m idc.run_chain                 # replay, publish to the UNS, land Parquet files and run acceptance
python -m idc.checks.acceptance         # rerun acceptance on its own, row for row against the bronze inputs
```

- Code: [Industrial Connectivity (MTConnect, Modbus, Ignition, MQTT UNS)](../README.md#industrial-connectivity-mtconnect-modbus-ignition-mqtt-uns)
- Architecture: [Industrial Connectivity](../README.md#industrial-connectivity)
- Ignition gateway setup: [`ignition/README.md`](ignition/README.md)

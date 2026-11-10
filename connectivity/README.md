# Industrial connectivity

This folder holds the connectivity chain that replays the HMC and compressor controller data through MTConnect, Modbus TCP, Ignition and the MQTT UNS and lands it as Parquet, verified row for row against the bronze controller exports.

```bash
cd connectivity
python -m idc.replay.build_native       # MTConnect SHDR logs and Modbus register images
docker compose up -d                    # MTConnect agent, MQTT broker, Ignition gateway
python -m idc.run_chain                 # replay, publish to the UNS and land Parquet files
python -m idc.checks.acceptance         # landed files against the bronze inputs, row for row
```

- Code: [Industrial Connectivity (MTConnect, Modbus, Ignition, MQTT UNS)](../README.md#industrial-connectivity-mtconnect-modbus-ignition-mqtt-uns)
- Architecture: [Industrial Connectivity](../README.md#industrial-connectivity)
- Ignition gateway setup: [`ignition/README.md`](ignition/README.md)

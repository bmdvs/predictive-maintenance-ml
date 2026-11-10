# Ignition gateway

The gateway reads the compressor controllers over Modbus TCP and publishes their tags as Sparkplug B through MQTT Transmission. It runs as the `ignition` service in [`docker-compose.yml`](../docker-compose.yml).

## Image and edition

- Image: `inductiveautomation/ignition:8.3.10`, pinned in the compose file.
- Edition: `IGNITION_EDITION: standard`, in trial mode. The trial runs two hours.
- Trial reset: the Reset Trial button on the gateway's web page, or a container restart (`docker restart idc-ignition`) while trial time remains. `python -m idc.run_chain` restarts the gateway every 90 minutes (`ignition.trial_restart_minutes` in [`config/chain.yaml`](../config/chain.yaml)).
- [Maker Edition](https://inductiveautomation.com/ignition/maker-edition) is Ignition's free license for non-commercial use.
- Gateway admin credentials: `IGNITION_ADMIN_USER` and `IGNITION_ADMIN_PASSWORD` in `connectivity/.env` (gitignored).

## MQTT Transmission module

1. Download Cirrus Link MQTT Transmission 5.0.5 for Ignition 8.3 (`com.cirruslink.mqtt.transmission.gateway`) from [Third-Party Modules for Ignition 8.3.10](https://inductiveautomation.com/downloads/third-party-modules/8.3.10).
2. Save it as `connectivity/ignition/modules/MQTT-Transmission-signed.modl` (gitignored).
3. The compose file mounts it read-only into `/usr/local/bin/ignition/user-lib/modules/`, and `ACCEPT_MODULE_LICENSES` and `ACCEPT_MODULE_CERTS` accept its license and certificate at startup.

## Gateway settings

Set in the gateway web page (`http://localhost:8088`) and the Designer.

1. **Modbus device connection.** New Modbus TCP device:
   - name `compressor-replay`;
   - hostname `192.168.65.254`, port `5020`;
   - zero-based addressing off, 125 holding registers per request.
2. **Tag group.** `Fast`: direct mode, 100 ms.
3. **Tag import.** In the Designer, import [`tags/compressors.json`](tags/compressors.json) into the `default` tag provider. It holds:
   - UDT `Compressor`: 120 unsigned 16-bit OPC tags `s00_r0` to `s14_r7` at `[compressor-replay]{UnitId}.HRUS<n>`, tag group `Fast`, parameter `UnitId`;
   - folder `compressors` with instances `CMP-01` to `CMP-04` (unit IDs 1 to 4) and `room` (unit ID 5).
4. **MQTT Transmission server.** URL `tcp://mosquitto:1883`.
5. **MQTT Transmission transmitter:**
   - group ID `shop`, edge node ID `main`;
   - tag provider `default`, tag path `compressors`;
   - tag pacing period 100 ms;
   - device-level UDTs as devices on.
6. **OPC UA server**, for `python -m idc.checks.opcua_readback`:
   - tag providers exposed;
   - anonymous access on;
   - security policy None added beside Basic256Sha256;
   - bound to all interfaces.

`python -m idc.ignition.provision` writes items 2 to 5 into the gateway's file-based configuration under `ignition/data/config/resources` and the tag file under `ignition/tags/`. Restart the gateway after it runs.

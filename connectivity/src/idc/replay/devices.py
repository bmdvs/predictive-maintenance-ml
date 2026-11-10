"""Generate the MTConnect device model (Devices.xml) from the fleet register.

Each machine becomes one MTConnect Device. The data items a device carries depend on its
class: every machine has execution, program, spindle speed/load/temperature, feed-rate
override, an alarm condition, availability and a record-timestamp event; axis loads follow
the axes the class has; hydraulic and through-spindle coolant pressures exist on HMCs only.

SHDR keys (the DataItem name attribute) are the same on every device so one adapter key set
serves the fleet. DataItem ids must be unique across the agent, so each id is prefixed with
the machine id.

Usage:
    python -m idc.replay.devices --register <path/to/register.yaml> --out mtconnect/Devices.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

SCHEMA_VERSION = "2.7"

# Axes per machine class as the register names them.
CLASS_LAYOUT: dict[str, dict] = {
    "hmc": {"linear": ["X", "Y", "Z"], "rotary": ["B"], "pressures": True,
            "label": "Horizontal machining center"},
    "vmc": {"linear": ["X", "Y", "Z"], "rotary": [], "pressures": False,
            "label": "Vertical machining center"},
    "lathe": {"linear": ["X", "Z"], "rotary": [], "pressures": False,
              "label": "CNC lathe"},
    "mill_turn": {"linear": ["X", "Y", "Z"], "rotary": [], "pressures": False,
                  "label": "Mill-turn center"},
    "grinder": {"linear": ["X", "Y", "Z"], "rotary": [], "pressures": False,
                "label": "CNC grinder"},
}

# Base TCP port for adapter connections; machine N (in register order) uses BASE + N.
ADAPTER_PORT_BASE = 7878


def _device_xml(machine: dict, port: int) -> str:
    """Render one register entry as an MTConnect Device element."""
    mid = machine["machine_id"]
    layout = CLASS_LAYOUT[machine["class"]]
    p = mid.lower().replace("-", "_")
    L: list[str] = []
    a = L.append
    a(f'    <Device id="{p}" uuid="{mid}" name="{mid}">')
    a(f'      <Description manufacturer="{machine.get("controller", "")}" '
      f'model="{layout["label"]}">{layout["label"]}, installed {machine.get("install_year", "")}; '
      f'adapter port {port}</Description>')
    a('      <DataItems>')
    a(f'        <DataItem category="EVENT" id="{p}_avail" name="avail" type="AVAILABILITY"/>')
    a(f'        <DataItem category="EVENT" id="{p}_record_ts" name="record_ts" '
      f'type="x:RECORD_TIMESTAMP"/>')
    a('      </DataItems>')
    a('      <Components>')
    a(f'        <Axes id="{p}_axes" name="axes">')
    a('          <Components>')
    for ax in layout["linear"]:
        x = ax.lower()
        a(f'            <Linear id="{p}_{x}" name="{ax}">')
        a('              <DataItems>')
        a(f'                <DataItem category="SAMPLE" id="{p}_{x}_load" name="{ax}load" '
          f'type="LOAD" units="PERCENT"/>')
        a('              </DataItems>')
        a('            </Linear>')
    for ax in layout["rotary"]:
        x = ax.lower()
        a(f'            <Rotary id="{p}_{x}" name="{ax}">')
        a('              <DataItems>')
        a(f'                <DataItem category="SAMPLE" id="{p}_{x}_load" name="{ax}load" '
          f'type="LOAD" units="PERCENT"/>')
        a('              </DataItems>')
        a('            </Rotary>')
    a(f'            <Rotary id="{p}_s" name="S">')
    a('              <DataItems>')
    a(f'                <DataItem category="SAMPLE" id="{p}_srpm" name="Srpm" '
      f'type="ROTARY_VELOCITY" subType="ACTUAL" units="REVOLUTION/MINUTE"/>')
    a(f'                <DataItem category="SAMPLE" id="{p}_sload" name="Sload" type="LOAD" '
      f'units="PERCENT"/>')
    a(f'                <DataItem category="SAMPLE" id="{p}_stemp" name="Stemp" '
      f'type="TEMPERATURE" units="CELSIUS"/>')
    a('              </DataItems>')
    a('            </Rotary>')
    a('          </Components>')
    a('        </Axes>')
    a(f'        <Controller id="{p}_ctrl" name="controller">')
    a('          <DataItems>')
    a(f'            <DataItem category="CONDITION" id="{p}_alarm" name="alarm" type="SYSTEM"/>')
    a('          </DataItems>')
    a('          <Components>')
    a(f'            <Path id="{p}_path" name="path">')
    a('              <DataItems>')
    a(f'                <DataItem category="EVENT" id="{p}_exec" name="exec" type="EXECUTION"/>')
    a(f'                <DataItem category="EVENT" id="{p}_program" name="program" '
      f'type="PROGRAM"/>')
    a(f'                <DataItem category="EVENT" id="{p}_fovr" name="Fovr" '
      f'type="PATH_FEEDRATE_OVERRIDE" subType="PROGRAMMED"/>')
    a('              </DataItems>')
    a('            </Path>')
    a('          </Components>')
    a('        </Controller>')
    if layout["pressures"]:
        a(f'        <Systems id="{p}_sys" name="systems">')
        a('          <Components>')
        a(f'            <Hydraulic id="{p}_hyd" name="hydraulic">')
        a('              <DataItems>')
        a(f'                <DataItem category="SAMPLE" id="{p}_hpres" name="Hpres" '
          f'type="PRESSURE" units="BAR"/>')
        a('              </DataItems>')
        a('            </Hydraulic>')
        a(f'            <Coolant id="{p}_cool" name="coolant">')
        a('              <DataItems>')
        a(f'                <DataItem category="SAMPLE" id="{p}_cpres" name="Cpres" '
          f'type="PRESSURE" units="BAR"/>')
        a('              </DataItems>')
        a('            </Coolant>')
        a('          </Components>')
        a('        </Systems>')
    a('      </Components>')
    a('    </Device>')
    return "\n".join(L)


def adapter_ports(machines: list[dict]) -> dict[str, int]:
    """Map each machine id to the TCP port its adapter connection uses, in register order."""
    return {m["machine_id"]: ADAPTER_PORT_BASE + i for i, m in enumerate(machines)}


def render_devices(machines: list[dict]) -> str:
    """Render the complete Devices.xml document for the machines in register order."""
    ports = adapter_ports(machines)
    head = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<!-- Generated by idc.replay.devices from the fleet register. Do not edit. -->',
        f'<MTConnectDevices xmlns="urn:mtconnect.org:MTConnectDevices:{SCHEMA_VERSION}"',
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '  xmlns:x="urn:idc:extensions"',
        f'  xsi:schemaLocation="urn:mtconnect.org:MTConnectDevices:{SCHEMA_VERSION} '
        f'/schemas/MTConnectDevices_{SCHEMA_VERSION}.xsd">',
        '  <Header creationTime="2026-11-08T00:00:00Z" sender="idc-agent" instanceId="0" '
        f'version="{SCHEMA_VERSION}" bufferSize="131072" assetBufferSize="0" assetCount="0"/>',
        '  <Devices>',
    ]
    body = [_device_xml(m, ports[m["machine_id"]]) for m in machines]
    tail = ['  </Devices>', '</MTConnectDevices>', '']
    return "\n".join(head + body + tail)


def render_adapters_block(machines: list[dict], host: str = "192.168.65.254") -> str:
    """Render the Adapters section of agent.cfg: one connection per machine."""
    ports = adapter_ports(machines)
    lines = ["Adapters {"]
    for m in machines:
        mid = m["machine_id"]
        lines += [f"  {mid} {{", f"    Host = {host}", f"    Port = {ports[mid]}", "  }"]
    lines.append("}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devices")
    parser.add_argument("--register", type=Path, required=True, help="fleet register YAML")
    parser.add_argument("--out", type=Path, default=Path("mtconnect/Devices.xml"),
                        help="path for the Devices.xml output")
    parser.add_argument("--adapters-out", type=Path, default=None,
                        help="also write the Adapters block for agent.cfg to this path")
    parser.add_argument("--adapter-host", default="192.168.65.254",
                        help="address of the adapter host as seen from the agent container; "
                             "192.168.65.254 is Docker Desktop's IPv4 host alias on Windows")
    args = parser.parse_args(argv)
    machines = yaml.safe_load(args.register.read_text(encoding="utf-8"))["machines"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_devices(machines), encoding="utf-8")
    print(f"wrote {args.out} with {len(machines)} devices")
    if args.adapters_out:
        args.adapters_out.write_text(render_adapters_block(machines, args.adapter_host) + "\n",
                                     encoding="utf-8")
        print(f"wrote {args.adapters_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

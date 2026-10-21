"""Rolling-bearing defect frequencies from register geometry.

Input: ball count, ball diameter, pitch diameter, contact angle. Output: BPFO, BPFI, BSF and FTF
as orders of shaft speed or in Hz for a given shaft speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pdm.config import BearingGeometry

FAULT_KEYS = ("bpfo", "bpfi", "bsf", "ftf")


@dataclass(frozen=True)
class FaultOrders:
    bpfo: float
    bpfi: float
    bsf: float
    ftf: float

    def as_dict(self) -> dict[str, float]:
        return {"bpfo": self.bpfo, "bpfi": self.bpfi, "bsf": self.bsf, "ftf": self.ftf}


def fault_orders(n: int, d_mm: float, pitch_mm: float, contact_deg: float) -> FaultOrders:
    ratio = d_mm / pitch_mm * math.cos(math.radians(contact_deg))
    return FaultOrders(
        bpfo=n / 2.0 * (1.0 - ratio),
        bpfi=n / 2.0 * (1.0 + ratio),
        bsf=pitch_mm / (2.0 * d_mm) * (1.0 - ratio**2),
        ftf=0.5 * (1.0 - ratio),
    )


def bearing_fault_orders(geometry: BearingGeometry) -> FaultOrders:
    return fault_orders(geometry.n, geometry.d_mm, geometry.pitch_mm, geometry.contact_deg)


def fault_frequencies_hz(orders: FaultOrders, shaft_hz: float) -> dict[str, float]:
    return {k: v * shaft_hz for k, v in orders.as_dict().items()}

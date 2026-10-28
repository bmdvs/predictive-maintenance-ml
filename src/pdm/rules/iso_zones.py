"""ISO 20816-3 evaluation zone for a velocity RMS value.

Input: velocity RMS in mm/s, machine group and foundation type. Output: zone letter A to D.
"""

from pdm.constants import ISO_20816_3_ZONES_MM_S


def zone_boundaries(group: int, foundation: str) -> tuple[float, float, float]:
    return ISO_20816_3_ZONES_MM_S[(group, foundation)]


def zone(vel_mm_s: float, group: int, foundation: str) -> str:
    ab, bc, cd = zone_boundaries(group, foundation)
    if vel_mm_s < ab:
        return "A"
    if vel_mm_s < bc:
        return "B"
    if vel_mm_s < cd:
        return "C"
    return "D"


def zone_b_width(group: int, foundation: str) -> float:
    ab, bc, _ = zone_boundaries(group, foundation)
    return bc - ab

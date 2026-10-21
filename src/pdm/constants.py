"""Physical and mathematical constants and ISO 20816-3 zone boundaries.

Zone boundaries are velocity RMS in mm/s, keyed by machine group and foundation type,
as (A/B, B/C, C/D).
"""

STANDARD_GRAVITY_M_S2 = 9.80665
MM_PER_M = 1000.0
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0
DAYS_PER_YEAR = 365.25

ISO_20816_3_ZONES_MM_S: dict[tuple[int, str], tuple[float, float, float]] = {
    (1, "rigid"): (2.3, 4.5, 7.1),
    (1, "flexible"): (3.5, 7.1, 11.0),
    (2, "rigid"): (1.4, 2.8, 4.5),
    (2, "flexible"): (2.3, 4.5, 7.1),
}

ISO_4406_UPPER_PER_ML_SCALE = 0.01
ISO_4406_LOWER_FRAC_OF_UPPER = 0.5


def iso_4406_upper_per_ml(code: int) -> float:
    """Upper particle count per ml for an ISO 4406 range code."""
    return ISO_4406_UPPER_PER_ML_SCALE * 2.0**code

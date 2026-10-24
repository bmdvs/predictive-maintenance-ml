"""Shared setup for job task entry points; the logic lives in pdm.entrypoints."""

from pdm.entrypoints import register, setup

__all__ = ["register", "setup"]

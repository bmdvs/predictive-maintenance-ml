"""Shared fixtures: the bronze export root, for tests that compare against the exports."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def export_root() -> Path:
    """Export root from IDC_EXPORT_ROOT, else from the chain configuration; skips if absent."""
    if "IDC_EXPORT_ROOT" in os.environ:
        root = Path(os.environ["IDC_EXPORT_ROOT"])
    else:
        from idc.checks.compare_exports import load_config
        config = load_config(Path(__file__).resolve().parents[1] / "config" / "chain.yaml")
        root = Path(config["paths"]["export_root"])
    if not (root / "controller").exists():
        pytest.skip("bronze exports not present")
    return root

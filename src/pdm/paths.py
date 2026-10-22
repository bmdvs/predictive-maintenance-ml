"""Absolute paths for the repository, resolved from the package location.

Every module that reads or writes files takes its locations from here.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
REGISTER_FILE = CONFIG_DIR / "register.yaml"
DATA_DIR = REPO_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SAMPLES_DIR = DATA_DIR / "samples"
EXPORTS_DIR = REPO_ROOT / "exports"
LOCAL_GOLD_DIR = EXPORTS_DIR / "local_gold"
GOLD_SNAPSHOT_DIR = EXPORTS_DIR / "gold_snapshot"
DOCS_DIR = REPO_ROOT / "docs"
MLRUNS_DIR = REPO_ROOT / "mlruns"


def bronze_system_dir(system: str, root: Path | None = None) -> Path:
    """Directory holding one bronze system's export."""
    return (root or BRONZE_DIR) / system

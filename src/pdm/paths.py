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
SCORES_DIR = DATA_DIR / "scores"
SAMPLES_DIR = DATA_DIR / "samples"
WAREHOUSE_FILE = DATA_DIR / "warehouse.duckdb"
DOCS_DIR = REPO_ROOT / "docs"
MLRUNS_DIR = REPO_ROOT / "mlruns"


def bronze_system_dir(system: str, root: Path | None = None) -> Path:
    """Directory holding one bronze system's export."""
    return (root or BRONZE_DIR) / system

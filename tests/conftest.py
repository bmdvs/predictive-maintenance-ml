"""Shared fixtures: a local Spark session (skipped when no Java runtime is available)."""

import os
import shutil

import pytest


def _java_available() -> bool:
    home = os.environ.get("JAVA_HOME")
    return bool(shutil.which("java") or (home and os.path.exists(os.path.join(home, "bin"))))


@pytest.fixture(scope="session")
def spark():
    if not _java_available():
        pytest.skip("no Java runtime for local Spark")
    from pdm.spark import local_session

    s = local_session("pdm-tests", cores=2, shuffle_partitions=2)
    yield s
    s.stop()

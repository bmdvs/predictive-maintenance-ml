"""Independence from the generator, config hygiene and text style."""

import re
import subprocess

from generate.checks.realism import check_independence

from pdm.config import load_config
from pdm.paths import REPO_ROOT

BANNED = re.compile(r"case study|portfolio|demonstration|simulation|synthetic", re.IGNORECASE)


def test_pdm_does_not_depend_on_generator():
    assert check_independence().passed


def test_config_has_no_pilot_override():
    assert "override" not in load_config()
    assert "forced_instances" not in (REPO_ROOT / "config" / "config.yaml").read_text(encoding="utf-8")


def _tracked_text_files():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [
        REPO_ROOT / p for p in out if p.endswith((".py", ".md", ".yaml", ".yml", ".toml", ".sql", ".html"))
    ]


def test_no_em_or_en_dashes():
    dashes = (chr(0x2014), chr(0x2013))
    bad = [p.name for p in _tracked_text_files() if any(d in p.read_text(encoding="utf-8") for d in dashes)]
    assert not bad


def test_no_banned_words_in_code_text():
    bad = []
    for p in _tracked_text_files():
        if (
            p.suffix == ".py"
            and ("src" in p.parts or "generate" in p.parts)
            and BANNED.search(p.read_text(encoding="utf-8"))
        ):
            bad.append(p.name)
    assert not bad

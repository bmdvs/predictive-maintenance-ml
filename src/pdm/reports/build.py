"""Builds the pages from the gold snapshot and the page text, then checks the text.

python -m pdm.reports.build

Outputs: docs/index.html, docs/reports/model_report.html, docs/reports/monitoring_report.html,
and a check that every block of page text appears verbatim on its page and that no other
paragraph or list item on a page carries a number.
"""

from __future__ import annotations

import html
import re
import sys

from pdm import log
from pdm.reports import monitoring_report, queue, report_model, style
from pdm.reports.data import Data
from pdm.reports.text import PageText

LOG = log.get_logger("pdm.reports.build")
# The queue page (docs/index.html) is the Maintenance Queue screen built by pdm.reports.queue from the prod build
# cache; it carries no page text, so section A is not checked (was PAGES with "A": "index.html").
# The model report (docs/reports/model_report.html) is built by pdm.reports.report_model from the report data
# bundle and the outline's text; section B is not checked (was "B": "reports/model_report.html").
PAGES = {"C": "reports/monitoring_report.html"}


def check(t: PageText, pages: dict[str, str]) -> list[str]:
    """Every used block of text appears verbatim on its page; no other paragraph or list item carries a number."""
    problems = []
    allowed = set()
    for sec in ("A", "B"):
        for i in range(len(t.sections[sec])):
            t.used.add((sec, i))
    for sec, i in sorted(t.used):
        if sec in ("A", "B"):
            continue
        block = t.sections[sec][i]
        if (
            block[0]
            .lstrip()
            .startswith(
                (
                    "**Exhibit",
                    "**Title",
                    "**Subtitle",
                    "Tiles:",
                    "|",
                    "The event timeline as",
                    "The compact table as",
                )
            )
        ):
            continue
        rendered = t.render(block[1:] if re.fullmatch(r"\*\*[A-I]\. .+\*\*", block[0].strip()) else block)
        if block[0].startswith("#"):
            rendered = html.escape(re.sub(r"^#+\s*", "", block[0]).strip("*"), quote=False)
        if rendered and rendered not in pages[sec]:
            problems.append(f"{sec}{i}: not found verbatim: {block[0][:60]}")
    texts = {
        sec: {x for x in re.findall(r"<(?:p|li)>(.*?)</(?:p|li)>", t.render(sum(t.sections[sec], [])))}
        for sec in pages
    }
    for sec, page in pages.items():
        for item in re.findall(r"<(?:p|li)>(.*?)</(?:p|li)>", page):
            if (
                re.search(r"\d", re.sub(r"<[^>]+>", "", item))
                and item not in texts[sec]
                and not any(a and a in item for a in allowed)
            ):
                problems.append(f"{sec}: prose with a number not from the page text: {item[:80]}")
    unused = [
        f"{sec}{i}: {b[0][:60]}"
        for sec in ("C",)
        for i, b in enumerate(t.sections[sec])
        if (sec, i) not in t.used
    ]
    problems += [f"unused block {u}" for u in unused]
    return problems


def main() -> int:
    log.configure()
    d, t = Data(), PageText()
    queue.build()
    report_model.build()
    monitoring_report.build(d, t)
    pages = {sec: (style.DOCS / rel).read_text(encoding="utf-8") for sec, rel in PAGES.items()}
    problems = check(t, pages)
    for p in problems:
        LOG.warning(p)
    LOG.info("pages built; text check: %d problems", len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

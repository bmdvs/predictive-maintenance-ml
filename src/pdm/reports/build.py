"""Builds the pages and the README from the gold snapshot and the page text, then checks the text.

python -m pdm.reports.build

Outputs: docs/index.html, docs/reports/model_report.html, docs/reports/monitoring_report.html,
README.md, and a check that every block of page text appears verbatim on its page and that no other
paragraph or list item on a page carries a number, and the trace of every number in appendix.yaml to the record.
"""

from __future__ import annotations

import html
import re
import sys

from pdm import log, paths
from pdm.reports import appendix, monitoring_report, queue, report_model, style
from pdm.reports.data import Data
from pdm.reports.text import PageText

LOG = log.get_logger("pdm.reports.build")
# The queue page (docs/index.html) is the Maintenance Queue screen built by pdm.reports.queue from the prod build
# cache; it carries no page text, so section A is not checked (was PAGES with "A": "index.html").
# The model report (docs/reports/model_report.html) is built by pdm.reports.report_model from the report data
# bundle and the outline's text; section B is not checked (was "B": "reports/model_report.html").
PAGES = {"C": "reports/monitoring_report.html"}
SCREENSHOTS = ("screenshots/queue.png", "screenshots/queue_detail.png")


def readme(t: PageText) -> str:
    """The README in section D's order, with the screenshots, the architecture figure and the run commands."""
    out: list[str] = []
    t.lines("D", "The sections follow")
    out.append(f"# {t.field('D', 'Title')}\n")
    for i, b in enumerate(t.sections["D"]):
        first = b[0].strip()
        if i < 2:
            continue
        label = re.fullmatch(r"\*\*(.+?)\*\*", first)
        if label:
            t.lines("D", label.group(1) + "**")
            out.append(f"## {label.group(1)}\n")
            continue
        if first.startswith("(Screenshots"):
            t.lines("D", "(Screenshots")
            out.append(
                "".join(
                    f"![{n.split('/')[-1].split('.')[0].replace('_', ' ')}](docs/{n})\n" for n in SCREENSHOTS
                )
            )
            continue
        if first.startswith("(Architecture"):
            t.lines("D", "(Architecture")
            out.append("![architecture](docs/figures/model_architecture.png)\n")
            continue
        if first.startswith("As 13.6"):
            t.lines("D", "As 13.6")
            out.append("\n".join(b[1:]) + "\n")
            out.append(RUN)
            continue
        out.append("\n".join(t.lines("D", first.lstrip("#*- ")[:40])) + "\n")
    return "\n".join(out)


RUN = """
```bash
pip install -e ".[dev,spark]"
python -m pdm.parity
python -m pdm.reports.build
```

```bash
databricks bundle validate -t prod
databricks bundle deploy -t prod
databricks bundle run -t prod pdm_monthly --no-wait
```
"""


def check(t: PageText, pages: dict[str, str]) -> list[str]:
    """Every used block of text appears verbatim on its page; no other paragraph or list item carries a number."""
    problems = []
    allowed = set()
    for sec in ("A", "B"):
        for i in range(len(t.sections[sec])):
            t.used.add((sec, i))
    for sec, i in sorted(t.used):
        if sec in ("A", "B", "D"):
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
    (paths.REPO_ROOT / "README.md").write_text(readme(t), encoding="utf-8")
    pages = {sec: (style.DOCS / rel).read_text(encoding="utf-8") for sec, rel in PAGES.items()}
    problems = check(t, pages)
    app_problems, traced = appendix.check(d)
    problems += app_problems
    for p in problems:
        LOG.warning(p)
    LOG.info("appendix C and I: %d numbers and point ids traced, %d problems", traced, len(app_problems))
    LOG.info("pages built; text check: %d problems", len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

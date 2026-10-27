"""The page text: paragraphs and labelled lines from page_text.md, rendered to HTML verbatim.

Inputs: page_text.md, sections A (queue page), B (model report), C (monitoring report), D (README).
Outputs: HTML (or Markdown for the README) for one block of text, found by the start of its first line.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

SOURCE = Path(__file__).with_name("page_text.md")


def inline(s: str) -> str:
    """Bold and code spans in one line of text, escaped for HTML."""
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


class PageText:
    def __init__(self, path: Path = SOURCE) -> None:
        raw = path.read_text(encoding="utf-8")
        self.sections: dict[str, list[list[str]]] = {}
        for part in re.split(r"^## ", raw, flags=re.M)[1:]:
            key = part[0]
            blocks, cur = [], []
            for line in part.splitlines()[1:]:
                if line.strip() in ("", "---"):
                    if cur:
                        blocks.append(cur)
                        cur = []
                else:
                    cur.append(line)
            if cur:
                blocks.append(cur)
            self.sections[key] = blocks
        self.used: set[tuple[str, int]] = set()

    def _find(self, section: str, start: str) -> tuple[int, list[str]]:
        hits = [(i, b) for i, b in enumerate(self.sections[section]) if b[0].lstrip("#*- ").startswith(start)]
        if len(hits) != 1:
            raise KeyError(f"{section}: {len(hits)} blocks start with {start!r}")
        return hits[0]

    def lines(self, section: str, start: str) -> list[str]:
        i, b = self._find(section, start)
        self.used.add((section, i))
        return b

    def html(self, section: str, start: str) -> str:
        """A block as HTML: a paragraph, a bullet list, or a paragraph followed by its bullets."""
        return self.render(self.lines(section, start))

    @staticmethod
    def render(lines: list[str]) -> str:
        out, items = [], []
        for line in lines:
            if line.startswith("- "):
                items.append(f"<li>{inline(line[2:])}</li>")
            else:
                if items:
                    out.append("<ul>" + "".join(items) + "</ul>")
                    items = []
                out.append(f"<p>{inline(line)}</p>")
        if items:
            out.append("<ul>" + "".join(items) + "</ul>")
        return "".join(out)

    def field(self, section: str, label: str) -> str:
        """The text after a bold label, as in '**Title:** Condition Monitoring Queue'."""
        line = self.lines(section, label)[0]
        m = re.match(r"\*\*" + re.escape(label) + r":?\*\*:?\s*(.*)", line.strip())
        if not m:
            raise KeyError(f"{section}: no field {label!r}")
        value = m.group(1).strip()
        quoted = re.match(r'^"(.+?)"\.?$', value)
        return quoted.group(1) if quoted else value

    def items(self, section: str, label: str) -> list[str]:
        """The bullet items under a labelled line."""
        b = self.lines(section, label)
        return [re.sub(r"^\d+\.\s*|^- ", "", x).strip() for x in b[1:]]

    def heading(self, section: str, start: str) -> str:
        line = self.lines(section, start)[0]
        return re.sub(r"^#+\s*", "", line).strip()

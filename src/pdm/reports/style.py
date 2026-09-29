"""Shared palette, chart helpers, HTML tables and page shells for the pages under docs/.

Inputs: matplotlib figures and DataFrames from the page builders. Outputs: PNG files under docs/figures
and HTML strings. The palette and layout follow the shop's other analytics pages.
"""

from __future__ import annotations

import hashlib
import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pdm import paths  # noqa: E402

BRAND_BLUE = "#3D5166"
ACCENT = "#6B8FA8"
LIGHT_BLUE = "#A8C0D1"
AMBER = "#D4881E"
RED = "#CC0000"
GREEN = "#1A7A3A"
GREY = "#AAAAAA"
DARK_GREY = "#555555"
TEXT = "#222222"
CREDIT = "Created by Brian Davis"
METHOD_COLOR = {"rule_floor": GREY, "nbm": BRAND_BLUE, "controller": AMBER, "model_layer_instrumented": GREEN}

DOCS = paths.REPO_ROOT / "docs"
CHART_W, CHART_H, CHART_DPI = 8.2, 3.8, 130
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#DDDDDD",
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": CHART_DPI,
    }
)


def chart_style(ax: Any, grid: str = "y") -> None:
    (ax.yaxis if grid == "y" else ax.xaxis).grid(True, color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#DDDDDD")


def fig(
    h: float | None = None, w: float | None = None, ncols: int = 1, nrows: int = 1, grid: str = "y", **kw: Any
):
    f, ax = plt.subplots(nrows, ncols, figsize=(w or CHART_W, h or CHART_H), **kw)
    for a in ax.ravel() if hasattr(ax, "ravel") else [ax]:
        chart_style(a, grid)
    return f, ax


def save(f: Any, name: str, alt: str = "", up: int = 0) -> str:
    """Write the figure under docs/figures and return the img tag for a page `up` levels below docs."""
    out = DOCS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    f.savefig(out / f"{name}.png", format="png", bbox_inches="tight", dpi=CHART_DPI)
    plt.close(f)
    version = hashlib.sha1((out / f"{name}.png").read_bytes()).hexdigest()[:8]
    return f'<img alt="{html.escape(alt)}" src="{"../" * up}figures/{name}.png?v={version}">'


def chart(title: str, img: str) -> str:
    return f"<div class='chart-title'>{html.escape(title)}</div>{img}"


def screenshot(name: str, title: str, up: int = 0) -> str:
    """A platform or page screenshot under docs/; a frame naming the file while it is not yet captured."""
    p = DOCS / name
    if p.exists():
        version = hashlib.sha1(p.read_bytes()).hexdigest()[:8]
        img = f'<img class="shot" alt="{html.escape(title)}" src="{"../" * up}{name}?v={version}">'
    else:
        img = f"<div class='pending'>{html.escape(name)}</div>"
    return chart(title, img)


def fmt_num(v: Any, d: int) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{d}f}"
    return html.escape(str(v))


def table(
    df: pd.DataFrame,
    decimals: dict[str, int] | None = None,
    default: int = 2,
    cls: str = "data",
    row_attrs: list[str] | None = None,
) -> str:
    """An HTML table; numbers right-aligned, decimals per column."""
    decimals = decimals or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for i, (_, r) in enumerate(df.iterrows()):
        cells = ""
        for c in df.columns:
            v = r[c]
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            cells += f'<td class="{"num" if num else ""}">{fmt_num(v, decimals.get(c, default))}</td>'
        attrs = f" {row_attrs[i]}" if row_attrs else ""
        body.append(f"<tr{attrs}>{cells}</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def tiles(items: list[tuple[str, str]]) -> str:
    return (
        "<div class='kpis'>"
        + "".join(
            f"<div class='kpi'><div class='v'>{html.escape(v)}</div><div class='l'>{html.escape(lab)}</div></div>"
            for lab, v in items
        )
        + "</div>"
    )


CSS = f"""
:root {{ --brand: {BRAND_BLUE}; --accent: {ACCENT}; --text: {TEXT}; --muted: {DARK_GREY}; --rule: #DDDDDD; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #F4F5F7; color: var(--text); font-family: "Segoe UI", Helvetica, Arial, sans-serif; font-size: 15px; line-height: 1.55; }}
h2 {{ color: var(--brand); font-size: 21px; margin: 36px 0 8px; border-bottom: 1px solid var(--rule); padding-bottom: 4px; }}
h3 {{ color: var(--brand); font-size: 16px; margin: 22px 0 6px; }}
h4 {{ color: var(--brand); font-size: 14px; margin: 18px 0 6px; }}
p {{ margin: 8px 0 12px; }}
ul {{ margin: 6px 0 12px; padding-left: 22px; }}
img {{ max-width: 100%; display: block; margin: 10px 0 4px; }}
img.shot {{ border: 1px solid var(--rule); }}
.pending {{ border: 1px dashed {GREY}; color: var(--muted); font-size: 12px; padding: 40px 12px; text-align: center; margin: 10px 0; }}
table.data {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0 16px; }}
table.data th {{ background: var(--brand); color: white; text-align: left; padding: 6px 8px; font-weight: 600; }}
table.data td {{ border-bottom: 1px solid #EEE; padding: 5px 8px; vertical-align: top; }}
table.data td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.nowrap1 td:nth-child(1), table.nowrap2 td:nth-child(2) {{ white-space: nowrap; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 16px 0 22px; }}
.kpi {{ border: 1px solid var(--rule); border-top: 3px solid var(--accent); padding: 10px 12px; background: white; }}
.kpi .v {{ font-size: 20px; font-weight: 700; color: var(--brand); }}
.kpi .l {{ font-size: 12px; color: var(--muted); }}
.kpi-head {{ font-size: 12px; font-weight: 700; color: var(--brand); text-transform: uppercase; letter-spacing: 0.5px; margin: 14px 0 0; }}
.chart-title {{ color: var(--brand); font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; text-align: center; margin: 22px 0 0; }}
.byline {{ color: var(--muted); font-size: 12px; }}
"""

REPORT_CSS = (
    CSS
    + """
body { background: white; }
.page-header { background: var(--brand); color: white; padding: 14px 40px; display: flex; justify-content: space-between; align-items: center; gap: 24px; }
.page-header h1 { margin: 0; font-size: 21px; font-weight: 700; letter-spacing: -0.3px; color: white; }
.page-header .sub { font-size: 12px; white-space: nowrap; }
.layout { display: flex; max-width: 1200px; margin: 0 auto; padding: 0 40px; }
nav.toc { width: 230px; flex-shrink: 0; margin: 0; padding: 36px 20px 40px 0; position: sticky; top: 0; height: 100vh; overflow-y: auto;
          border-right: 1px solid var(--rule); font-size: 13px; }
nav.toc .toc-title { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; font-weight: 700; }
nav.toc a { display: block; color: var(--muted); padding: 4px 0 4px 10px; border-left: 2px solid transparent; line-height: 1.4; text-decoration: none; }
nav.toc a:hover { color: var(--brand); border-left-color: var(--brand); }
nav.toc a.sub { font-size: 12px; padding-left: 22px; }
.content { flex: 1; min-width: 0; max-width: 880px; padding: 24px 0 80px 52px; }
.subtitle { color: var(--muted); font-size: 14px; margin: 4px 0 4px; }
@media (max-width: 900px) {
  .page-header { padding: 12px 16px; } .layout { display: block; padding: 0 16px; }
  nav.toc { position: static; width: auto; height: auto; border-right: 0; border-bottom: 1px solid var(--rule); padding: 16px 0 12px; }
  .content { padding: 16px 0 40px; }
}
@media print { nav.toc { display: none; } .layout { display: block; max-width: none; padding: 0; } .content { max-width: none; padding: 0; } }
"""
)


def report_shell(title: str, subtitle: str, body: str, toc: list[tuple[str, ...]]) -> str:
    """A report page: the title in a bar across the top with the byline, the contents down the left."""
    nav = (
        '<nav class="toc"><div class="toc-title">Contents</div>'
        + "".join(
            f'<a{" class=" + chr(34) + "sub" + chr(34) if len(x) > 2 else ""} href="#{x[0]}">{html.escape(x[1])}</a>'
            for x in toc
        )
        + "</nav>"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{REPORT_CSS}</style></head>
<body><div class="page-header"><h1>{html.escape(title)}</h1><div class="sub">{CREDIT}</div></div>
<div class="layout">{nav}
<main class="content"><div class="subtitle">{html.escape(subtitle)}</div>{body}
</main></div></body></html>"""


def write(relative: str, text: str) -> Path:
    p = DOCS / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p

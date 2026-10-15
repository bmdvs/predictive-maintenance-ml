"""Inline SVG charts for the maintenance queue: time-series trends with limit lines, level bands, colored
segments past a limit and in-plot labels placed without overlap; grouped bars; spectra with markers.

Inputs: plain arrays and label lists. Output: SVG strings.
"""

from __future__ import annotations

import html
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

GRAY, AMBER, RED = "#8a929c", "#c77700", "#c8322b"
INK, MUTED, GRID, AXIS = "#2b3038", "#69717c", "#eceef1", "#c9ced5"
POINT_COLORS = ("#3c4655", "#9aa3ae")
BLUE, REF = "#2f6fd6", "#b9bfc7"
FONT = 10.0
ROW = 15
CHAR_W = 5.7


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


@lru_cache(maxsize=4)
def _font(size: float):
    try:
        from PIL import ImageFont

        return ImageFont.truetype(
            str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf"),
            size=int(round(size * 4)),
        )
    except OSError:
        return None


def text_w(s: str, size: float = FONT) -> float:
    """Rendered width of a label in px (Segoe UI metrics at 4x for precision; a wide estimate without the font),
    plus 2 px for the halo."""
    f = _font(size)
    if f is not None:
        return f.getlength(s) / 4 + 2
    return (
        sum(7.2 if c.isupper() else 6.2 if c.isdigit() else 3.2 if c in " ,.:;" else 6.0 for c in s)
        * size
        / FONT
        + 2
    )


@dataclass
class Placer:
    """Keeps the boxes of placed labels and finds a free position for each new one."""

    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)

    def free(self, b: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = b
        return all(x1 <= c[0] or x0 >= c[2] or y1 <= c[1] or y0 >= c[3] for c in self.boxes)

    def take(self, b: tuple[float, float, float, float]) -> None:
        self.boxes.append(b)


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step) + 1)]


def fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
    if a >= 1:
        return f"{v:.1f}".rstrip("0").rstrip(".") if not float(v).is_integer() else f"{v:.0f}"
    if a >= 0.01:
        return f"{v:.2f}".rstrip("0")
    return f"{v:.0e}"


class TimeChart:
    """A time-series chart with a time x axis in shop local time."""

    def __init__(
        self,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
        width: int = 1060,
        height: int = 170,
        log: bool = False,
        tz: str = "America/Chicago",
        ylabel: str = "",
    ) -> None:
        self.t0, self.t1, self.w, self.h, self.log, self.tz = t0, t1, width, height, log, tz
        self.ylabel = ylabel
        self.ml, self.mr, self.mt, self.mb = 64 if ylabel else 52, 14, 8, 22
        self.parts: list[str] = []
        self.labels: list[tuple[str, dict]] = []
        self.legend: list[tuple[str, str, str, str]] = []
        self.lo, self.hi = np.inf, -np.inf
        self.placer = Placer()
        self.top_rows = 0

    def _key(self, kind: str, color: str, dash: str, label: str | None) -> None:
        if label and (kind, color, dash, label) not in self.legend:
            self.legend.append((kind, color, dash, label))

    # ---- scales
    def fit(self, values: np.ndarray, pad: float = 0.08) -> None:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if self.log:
            v = v[v > 0]
        if v.size:
            self.lo, self.hi = min(self.lo, float(v.min())), max(self.hi, float(v.max()))

    def _domain(self) -> tuple[float, float]:
        lo, hi = self.lo, self.hi
        if not np.isfinite(lo):
            lo, hi = (0.1, 10.0) if self.log else (0.0, 1.0)
        if self.log:
            a, b = math.log10(lo), math.log10(hi)
            if b - a < 0.3:
                a, b = a - 0.15, b + 0.15
            return a - 0.05 * (b - a), b + 0.08 * (b - a)
        if hi - lo < 1e-12:
            lo, hi = lo - 1, hi + 1
        span = hi - lo
        return lo - 0.06 * span, hi + 0.08 * span

    @property
    def plot(self) -> tuple[float, float, float, float]:
        top = self.mt + self.top_rows * ROW
        return self.ml, top, self.w - self.mr, self.h - self.mb

    def x(self, ts: pd.Timestamp | np.ndarray) -> np.ndarray | float:
        x0, _, x1, _ = self.plot
        t0, t1 = self.t0.value, self.t1.value
        v = ts.value if isinstance(ts, pd.Timestamp) else pd.DatetimeIndex(pd.to_datetime(ts)).asi8
        return x0 + (np.asarray(v, dtype=float) - t0) / (t1 - t0) * (x1 - x0)

    def y(self, v: np.ndarray | float) -> np.ndarray | float:
        _, y0, _, y1 = self.plot
        a, b = self._domain()
        vv = np.asarray(v, dtype=float)
        if self.log:
            vv = np.log10(np.clip(vv, 1e-12, None))
        return y1 - (vv - a) / (b - a) * (y1 - y0)

    # ---- marks (stored as callbacks so the plot box can grow for label rows first)
    def series(
        self,
        ts: pd.Series,
        vals: pd.Series,
        color: str,
        limits: list[tuple[pd.Series, str]] | None = None,
        width: float = 1.1,
        gap_hours: float = 30,
        name: str | None = None,
    ) -> None:
        self._key("line", color, "", name)
        self.labels.append(
            (
                "series",
                {
                    "ts": pd.to_datetime(ts).reset_index(drop=True),
                    "v": pd.Series(vals).reset_index(drop=True).astype(float),
                    "color": color,
                    "limits": limits or [],
                    "width": width,
                    "gap": gap_hours,
                },
            )
        )

    def hline(
        self,
        ts0: pd.Timestamp,
        ts1: pd.Timestamp,
        v: float,
        color: str,
        dash: str = "4 3",
        label: str | None = None,
    ) -> None:
        self._key("dash", color, dash, label)
        self.labels.append(
            (
                "hline",
                {"t0": ts0, "t1": ts1, "v": v, "color": color, "dash": dash, "label": None, "key": label},
            )
        )

    def band(
        self, ts0: pd.Timestamp, ts1: pd.Timestamp, lo: float, hi: float, color: str, label: str | None = None
    ) -> None:
        self._key("rect", color, "", label)
        self.labels.append(
            ("band", {"t0": ts0, "t1": ts1, "lo": lo, "hi": hi, "color": color, "label": None})
        )

    def event(self, ts: pd.Timestamp, text: str, color: str = MUTED) -> None:
        self.labels.append(("event", {"ts": ts, "text": text, "color": color}))

    # ---- render
    def _events_layout(self) -> list[tuple[float, int, str, str, str]]:
        """Event labels in rows above the plot area, each row free of overlaps; returns (x, row, anchor, text, color)."""
        x1 = self.w - self.mr
        evs = sorted([d for k, d in self.labels if k == "event"], key=lambda d: d["ts"])
        rows: list[list[tuple[float, float]]] = []
        out = []
        for d in evs:
            tx = float(self.x(pd.Timestamp(d["ts"])))
            if d["text"] is None:
                out.append((tx, -1, "", None, d["color"]))
                continue
            wv = text_w(d["text"])
            anchor, a, b = "start", tx + 3, tx + 3 + wv
            if b > x1:
                anchor, a, b = "end", tx - 3 - wv, tx - 3
            r = next((i for i, row in enumerate(rows) if all(b + 6 <= c or a >= e + 6 for c, e in row)), None)
            if r is None:
                rows.append([])
                r = len(rows) - 1
            rows[r].append((a, b))
            out.append((tx, r, anchor, d["text"], d["color"]))
        self.top_rows = len(rows)
        return out

    state_labels = ("Above the alert limit", "Above the alarm limit")

    def svg(self, title_id: str = "") -> str:
        evs = self._events_layout()
        for k, d in self.labels:
            if k == "series" and d["limits"]:
                st = self._states(d)
                if (st == 1).any():
                    self._key("line", AMBER, "", self.state_labels[0])
                if (st == 2).any():
                    self._key("line", RED, "", self.state_labels[1])
        a_, b_ = self._domain()
        shown = {
            d["key"]
            for k, d in self.labels
            if k == "hline"
            and d["v"] > (0 if self.log else -np.inf)
            and a_ <= (math.log10(d["v"]) if self.log else d["v"]) <= b_
        }
        order = {"line": 0, "rect": 1, "dash": 2}
        items = sorted(
            [it for it in self.legend if it[0] != "dash" or it[3] in shown],
            key=lambda it: 3 if it[3] in self.state_labels else order.get(it[0], 0),
        )
        leg, extra = legend_layout(items, self.ml, self.w - self.mr)
        self.mb += extra
        self.h = self.h + self.top_rows * ROW + extra
        x0, y0, x1, y1 = self.plot
        p = [
            f'<svg class="chart" viewBox="0 0 {self.w} {self.h}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">'
        ]
        if self.ylabel:
            p.append(ylabel_svg(self.ylabel, (y0 + y1) / 2))
        p.extend(legend_svg(leg, y1 + 24))
        a, b = self._domain()
        if self.log:
            ticks = [10**k for k in range(math.floor(a), math.ceil(b) + 1) if a <= k <= b]
            if len(ticks) < 2:
                ticks = [
                    v
                    for k in range(math.floor(a), math.ceil(b) + 1)
                    for v in (10**k, 2 * 10**k, 5 * 10**k)
                    if a <= math.log10(v) <= b
                ]
        else:
            ticks = nice_ticks(*self._lin_extent())
        for tv in ticks:
            yy = float(self.y(tv))
            if y0 - 0.5 <= yy <= y1 + 0.5:
                p.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}"/>')
                p.append(
                    f'<text class="tick" x="{x0 - 5}" y="{yy + 3:.1f}" text-anchor="end">{fmt(tv)}</text>'
                )
        days = pd.date_range(
            self.t0.tz_convert(self.tz).normalize() + pd.Timedelta(days=1),
            self.t1.tz_convert(self.tz),
            freq="D",
        )
        mondays = [d for d in days if d.weekday() == 0][::2]
        for d in mondays:
            xx = float(self.x(d.tz_convert("UTC")))
            p.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y0}" y2="{y1}" stroke="{GRID}"/>')
            p.append(
                f'<text class="tick" x="{xx:.1f}" y="{y1 + 14}" text-anchor="middle">{d:%b} {d.day}</text>'
            )
        p.append(
            f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="{AXIS}"/>'
        )
        p.append(
            f'<clipPath id="c{title_id}"><rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/></clipPath>'
        )
        inside: list[str] = []
        texts: list[tuple[str, dict]] = []
        for kind, d in self.labels:
            if kind == "band":
                xa, xb = float(self.x(pd.Timestamp(d["t0"]))), float(self.x(pd.Timestamp(d["t1"])))
                ya, yb = float(self.y(d["hi"])), float(self.y(d["lo"]))
                inside.append(
                    f'<rect x="{xa:.1f}" y="{ya:.1f}" width="{max(xb - xa, 0):.1f}" height="{max(yb - ya, 1):.1f}" fill="{d["color"]}" opacity="0.16"/>'
                )
                if d["label"]:
                    texts.append(
                        (
                            "band",
                            {
                                "x": xa + 4,
                                "ya": ya,
                                "yb": yb,
                                "text": d["label"],
                                "color": d["color"],
                                "xb": xb,
                            },
                        )
                    )
        for kind, d in self.labels:
            if kind == "hline":
                xa, xb = float(self.x(pd.Timestamp(d["t0"]))), float(self.x(pd.Timestamp(d["t1"])))
                yy = float(self.y(d["v"]))
                if y0 <= yy <= y1:
                    inside.append(
                        f'<line x1="{xa:.1f}" x2="{xb:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{d["color"]}" stroke-dasharray="{d["dash"]}" stroke-width="1.1"/>'
                    )
                    if d["label"]:
                        texts.append(
                            ("hline", {"x": xb - 3, "y": yy, "text": d["label"], "color": d["color"]})
                        )
        for kind, d in self.labels:
            if kind == "series":
                inside.extend(self._series_paths(d))
        p.append(f'<g clip-path="url(#c{title_id})">' + "".join(inside) + "</g>")
        for tx, r, anchor, text, color in evs:
            if text is None:
                p.append(
                    f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{y0}" y2="{y1}" stroke="{color}" stroke-width="0.8" stroke-dasharray="2 2" opacity="0.7"/>'
                )
                continue
            p.append(
                f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{self.mt + r * ROW + 4:.1f}" y2="{y1}" stroke="{color}" stroke-width="0.8" stroke-dasharray="2 2" opacity="0.7"/>'
            )
            xt = tx + 3 if anchor == "start" else tx - 3
            yt = self.mt + r * ROW + 10
            p.append(
                f'<text class="lbl ev" x="{xt:.1f}" y="{yt:.1f}" text-anchor="{anchor}" fill="{color}">{esc(text)}</text>'
            )
            wv = text_w(text)
            self.placer.take(
                (
                    xt if anchor == "start" else xt - wv,
                    yt - 11,
                    (xt + wv) if anchor == "start" else xt,
                    yt + 3,
                )
            )
        # limit labels at the right edge, stacked by line height so none overlap
        hl = sorted((d for k, d in texts if k == "hline"), key=lambda d: d["y"])
        ys: list[float] = []
        for d in hl:
            want = d["y"] - 3
            ys.append(max(want, ys[-1] + ROW) if ys else want)
        if ys and ys[-1] > y1 - 3:
            shift = ys[-1] - (y1 - 3)
            ys = [y - shift for y in ys]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + ROW)
        for d, yb in zip(hl, ys, strict=True):
            wv = text_w(d["text"])
            yb = max(yb, y0 + 11)
            p.append(
                f'<text class="lbl" x="{x1 - 3:.1f}" y="{yb:.1f}" text-anchor="end" fill="{d["color"]}">{esc(d["text"])}</text>'
            )
            self.placer.take((x1 - 3 - wv, yb - 11, x1 - 3, yb + 3))
        for kind, d in texts:
            if kind == "hline":
                continue
            wv = text_w(d["text"])
            if kind == "hline":
                cands = [
                    (d["x"] - wv, d["y"] - 15),
                    (d["x"] - wv, d["y"] + 1),
                    (x0 + 4 + (x1 - x0) * 0.55, d["y"] - 15),
                    (x0 + 4 + (x1 - x0) * 0.55, d["y"] + 1),
                    (x0 + 4 + (x1 - x0) * 0.3, d["y"] - 15),
                    (x0 + 4 + (x1 - x0) * 0.3, d["y"] + 1),
                ]
            else:
                mid = (d["ya"] + d["yb"]) / 2
                cands = [
                    (d["x"], mid - 7),
                    (d["x"], d["ya"] - 15),
                    (d["x"], d["yb"] + 1),
                    (d["x"] + (d["xb"] - d["x"]) * 0.4, mid - 7),
                    (d["x"] + (d["xb"] - d["x"]) * 0.2, d["ya"] - 15),
                    (d["x"] + (d["xb"] - d["x"]) * 0.2, d["yb"] + 1),
                ]
            for cx, cy in cands:
                b_ = (cx, cy, cx + wv, cy + 14)
                if cy >= y0 and cy + 14 <= y1 and cx >= x0 and cx + wv <= x1 and self.placer.free(b_):
                    self.placer.take(b_)
                    p.append(
                        f'<text class="lbl" x="{cx:.1f}" y="{cy + 11:.1f}" fill="{d["color"]}">{esc(d["text"])}</text>'
                    )
                    break
        p.append("</svg>")
        return "".join(p)

    def _lin_extent(self) -> tuple[float, float]:
        a, b = self._domain()
        return a, b

    @staticmethod
    def _states(d: dict) -> np.ndarray:
        v = d["v"]
        state = np.zeros(len(v), dtype=int)
        for lim, level in d["limits"]:
            lv = pd.Series(lim).reset_index(drop=True).astype(float).to_numpy()
            hit = np.isfinite(lv) & (v.to_numpy() >= lv)
            state = np.maximum(state, np.where(hit, 2 if level == "alarm" else 1, 0))
        return state

    def _series_paths(self, d: dict) -> list[str]:
        ts, v = d["ts"], d["v"]
        if len(ts) == 0:
            return []
        xs = np.asarray(self.x(ts), dtype=float)
        ys = np.asarray(self.y(v.to_numpy()), dtype=float)
        hrs = pd.DatetimeIndex(ts).asi8 / 3.6e12
        state = self._states(d)
        color = {0: d["color"], 1: AMBER, 2: RED}
        out, seg, cur = [], [], None
        for i in range(len(v)):
            if not np.isfinite(ys[i]):
                if seg:
                    out.append((cur, seg))
                seg, cur = [], None
                continue
            gap = i > 0 and hrs[i] - hrs[i - 1] > d["gap"]
            if seg and (gap or state[i] != cur):
                if not gap:
                    seg.append((xs[i], ys[i]))
                out.append((cur, seg))
                seg = [] if gap else [seg[-1]]
            if not seg:
                cur = state[i]
            seg.append((xs[i], ys[i]))
            cur = state[i]
        if seg:
            out.append((cur, seg))
        paths = []
        for st, pts in out:
            if len(pts) == 1:
                x_, y_ = pts[0]
                paths.append(f'<circle cx="{x_:.1f}" cy="{y_:.1f}" r="1.2" fill="{color[st]}"/>')
                continue
            dd = "M" + "L".join(f"{x_:.1f},{y_:.1f}" for x_, y_ in pts)
            paths.append(
                f'<path d="{dd}" fill="none" stroke="{color[st]}" stroke-width="{d["width"] + (0.4 if st else 0)}" stroke-linejoin="round"/>'
            )
        return paths


LEG_ROW = 14


def legend_layout(
    items: list[tuple[str, str, str, str]], x0: float, x1: float
) -> tuple[list[tuple[float, int, tuple[str, str, str, str]]], float]:
    """Legend items placed left to right in rows from x0, wrapping at x1; returns (x, row, item) and the
    height the rows need."""
    out, x, row = [], x0, 0
    for it in items:
        w = 20 + text_w(it[3]) + 14
        if x > x0 and x + w > x1:
            x, row = x0, row + 1
        out.append((x, row, it))
        x += w
    return out, ((row + 1) * LEG_ROW + 6) if items else 0


def legend_svg(placed: list[tuple[float, int, tuple[str, str, str, str]]], ytop: float) -> list[str]:
    p = []
    for x, row, (kind, color, dash, label) in placed:
        yy = ytop + row * LEG_ROW + 5
        if kind == "box":
            p.append(f'<rect x="{x:.1f}" y="{yy - 4:.1f}" width="16" height="8" fill="{color}"/>')
        elif kind == "rect":
            p.append(
                f'<rect x="{x:.1f}" y="{yy - 4:.1f}" width="16" height="8" fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-opacity="0.5"/>'
            )
        else:
            da = f' stroke-dasharray="{dash or "4 3"}"' if kind == "dash" else ""
            p.append(
                f'<line x1="{x:.1f}" x2="{x + 16:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="{1.3 if kind == "dash" else 2}"{da}/>'
            )
        p.append(f'<text class="leg" x="{x + 20:.1f}" y="{yy + 3.5:.1f}">{esc(label)}</text>')
    return p


def ylabel_svg(text: str, ymid: float, x: float = 11) -> str:
    return f'<text class="axl" transform="translate({x},{ymid:.1f}) rotate(-90)" text-anchor="middle">{esc(text)}</text>'


def fmt_small(v: float) -> str:
    if v >= 0.01:
        return fmt(v)
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}"


def bars(
    groups: list[dict],
    width: int = 1060,
    height: int = 250,
    ylabel: str = "",
    legend: list[tuple[str, str, str, str]] | None = None,
) -> str:
    """Grouped columns on a log scale: groups of {name, values: [(label, value, color)], limit}, with each
    column's label under it, its value above it, and a legend below."""
    ml, mr, mt = 64 if ylabel else 52, 10, 18
    leg, extra = legend_layout(legend or [], ml, width - mr)
    mb = 46 + extra
    height += extra
    vals = [v for g in groups for _, v, _ in g["values"] if v and v > 0] + [
        g["limit"] for g in groups if g.get("limit")
    ]
    lo, hi = math.log10(min(vals)) - 0.25, math.log10(max(vals)) + 0.35
    x0, x1, y0, y1 = ml, width - mr, mt, height - mb

    def y(v: float) -> float:
        return y1 - (math.log10(max(v, 1e-12)) - lo) / (hi - lo) * (y1 - y0)

    p = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">'
    ]
    if ylabel:
        p.append(ylabel_svg(ylabel, (y0 + y1) / 2))
    for k in range(math.floor(lo), math.ceil(hi) + 1):
        if lo <= k <= hi:
            yy = y(10**k)
            p.append(
                f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}"/><text class="tick" x="{x0 - 5}" y="{yy + 3:.1f}" text-anchor="end">1e{k}</text>'
            )
    p.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="{AXIS}"/>')
    gw = (x1 - x0) / max(len(groups), 1)
    for i, g in enumerate(groups):
        gx = x0 + i * gw
        n = len(g["values"])
        gap = 10
        bw = min(84, (gw - 40 - gap * (n - 1)) / n)
        start = gx + (gw - bw * n - gap * (n - 1)) / 2
        for j, (lab, v, col) in enumerate(g["values"]):
            bx = start + j * (bw + gap)
            yy = y(v)
            p.append(
                f'<rect x="{bx:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{y1 - yy:.1f}" fill="{col}"><title>{esc(lab)}: {v:.3g} g²</title></rect>'
            )
            p.append(
                f'<text class="lbl" x="{bx + bw / 2:.1f}" y="{yy - 4:.1f}" text-anchor="middle" fill="{INK}">{fmt_small(v)}</text>'
            )
            p.append(
                f'<text class="tick" x="{bx + bw / 2:.1f}" y="{y1 + 13}" text-anchor="middle">{esc(lab)}</text>'
            )
        if g.get("limit"):
            yl = y(g["limit"])
            p.append(
                f'<line x1="{start - 6:.1f}" x2="{start + n * (bw + gap) - gap + 6:.1f}" y1="{yl:.1f}" y2="{yl:.1f}" stroke="{AMBER}" stroke-dasharray="4 3" stroke-width="1.2"/>'
            )
        p.append(
            f'<text class="gname" x="{gx + gw / 2:.1f}" y="{y1 + 31}" text-anchor="middle">{esc(g["name"])}</text>'
        )
    p.extend(legend_svg(leg, y1 + 44))
    p.append("</svg>")
    return "".join(p)


def spectrum(
    f: np.ndarray,
    amp: np.ndarray,
    ref_f: np.ndarray,
    ref_amp: np.ndarray,
    fmax: float,
    markers: list[tuple[float, str, str, bool]],
    width: int = 520,
    height: int = 200,
    unit: str = "Hz",
    ylabel: str = "",
    legend: list[tuple[str, str, str, str]] | None = None,
) -> str:
    """Envelope spectrum (blue) over the healthy reference (gray), with fault-frequency markers
    (frequency, label, color, labeled), the x-axis label centered under the plot and a legend below."""
    ml, mr, mt = 64 if ylabel else 52, 10, 8
    leg, extra = legend_layout(legend or [], ml, width - mr)
    mb = 34 + extra
    height += extra + 8
    placer = Placer()
    lab = [m for m in markers if m[3]]
    rows = 1 if not lab else 2
    mt += rows * ROW
    x0, x1, y0, y1 = ml, width - mr, mt, height - mb
    m = (f > 0) & (f <= fmax)
    mr_ = (ref_f > 0) & (ref_f <= fmax)
    top = (
        max(float(amp[m].max()) if m.any() else 1e-6, float(ref_amp[mr_].max()) if mr_.any() else 1e-6) * 1.08
    )

    def X(v: np.ndarray | float) -> np.ndarray | float:
        return x0 + np.asarray(v) / fmax * (x1 - x0)

    def Y(v: np.ndarray | float) -> np.ndarray | float:
        return y1 - np.asarray(v) / top * (y1 - y0)

    p = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">'
    ]
    for tv in nice_ticks(0, top, 3):
        yy = float(Y(tv))
        p.append(
            f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}"/><text class="tick" x="{x0 - 5}" y="{yy + 3:.1f}" text-anchor="end">{fmt(tv)}</text>'
        )
    for tv in nice_ticks(0, fmax, 5):
        xx = float(X(tv))
        p.append(f'<text class="tick" x="{xx:.1f}" y="{y1 + 13}" text-anchor="middle">{fmt(tv)}</text>')
    p.append(
        f'<text class="axl" x="{(x0 + x1) / 2:.1f}" y="{y1 + 27}" text-anchor="middle">{esc(unit)}</text>'
    )
    if ylabel:
        p.append(ylabel_svg(ylabel, (y0 + y1) / 2))
    p.extend(legend_svg(leg, y1 + 36))
    p.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="{AXIS}"/>')
    p.append(
        '<path d="M'
        + "L".join(f"{a:.1f},{b:.1f}" for a, b in zip(X(ref_f[mr_]), Y(ref_amp[mr_]), strict=True))
        + f'" fill="none" stroke="{REF}" stroke-width="1"/>'
    )
    p.append(
        '<path d="M'
        + "L".join(f"{a:.1f},{b:.1f}" for a, b in zip(X(f[m]), Y(amp[m]), strict=True))
        + f'" fill="none" stroke="{BLUE}" stroke-width="1"/>'
    )
    for fr, text, col, labeled in sorted(markers, key=lambda t: t[0]):
        if fr > fmax:
            continue
        xx = float(X(fr))
        p.append(
            f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{y0}" y2="{y1}" stroke="{col}" stroke-width="0.9" stroke-dasharray="3 2"/>'
        )
        if labeled:
            wv = text_w(text)
            for r in range(rows):
                yy = 8 + r * ROW + 10
                for ax, anchor in ((xx + 2, "start"), (xx - 2 - wv, "end")):
                    b_ = (ax, yy - 11, ax + wv, yy + 3)
                    if ax >= x0 - 30 and ax + wv <= width and placer.free(b_):
                        placer.take(b_)
                        tx = xx + 2 if anchor == "start" else xx - 2
                        p.append(
                            f'<text class="lbl" x="{tx:.1f}" y="{yy:.1f}" text-anchor="{anchor}" fill="{col}">{esc(text)}</text>'
                        )
                        break
                else:
                    continue
                break
    p.append("</svg>")
    return "".join(p)

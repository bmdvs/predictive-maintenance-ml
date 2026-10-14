"""Loads docs/index.html in the installed Chrome at 1,440 px, clicks every queue row and checks the page; then
takes the README screenshots.

Checks: no detail section renders empty; no chart has overlapping labels; every status chip matches the prod
levels at the snapshot (from build/queue_cache/level_episodes.parquet); every suspected condition matches its
prod work request (build/queue_cache/work_requests.parquet); no horizontal scroll at 1,440 px.

Needs Playwright (any environment that has it) and pandas:
python tools/check_queue_page.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
CACHE = ROOT / "build" / "queue_cache"
SHOTS = ROOT / "docs" / "screenshots"
WIDTH = 1440

OVERLAP_JS = """
(det) => {
  const out = [];
  det.querySelectorAll('svg.chart').forEach((svg, i) => {
    const lab = [...svg.querySelectorAll('text.lbl')].map(t => ({t: t.textContent, b: t.getBBox()}));
    const all = lab.concat([...svg.querySelectorAll('text.tick')].map(t => ({t: t.textContent, b: t.getBBox(), tick: true})));
    for (let a = 0; a < lab.length; a++) for (let c = a + 1; c < all.length; c++) {
      const p = lab[a].b, q = all[c].b;
      const ox = Math.min(p.x + p.width, q.x + q.width) - Math.max(p.x, q.x);
      const oy = Math.min(p.y + p.height, q.y + q.height) - Math.max(p.y, q.y);
      if (ox > 0.5 && oy > 0.5) out.push([i, lab[a].t, all[c].t]);
    }
  });
  return out;
}
"""

EMPTY_JS = """
(det) => [...det.querySelectorAll('.sec')].map((s, i) => {
  const svgs = [...s.querySelectorAll('svg.chart')];
  const emptySvg = svgs.filter(v => v.querySelectorAll('path, circle, rect[height]').length < 2).length;
  const rows = s.querySelectorAll('tbody tr').length;
  return {i, title: (s.querySelector('.sech') || {}).textContent || '', svgs: svgs.length, emptySvg, rows};
}).filter(x => (x.svgs === 0 && x.rows === 0) || x.emptySvg > 0)
"""

HEAD_JS = """
(det) => {
  const chips = [...det.querySelectorAll('.hchip')].map(c => ({name: c.querySelector('.hl').textContent.replace('Shadow','').trim(), level: c.querySelector('.chip').textContent}));
  const kv = Object.fromEntries([...det.querySelectorAll('table.kv tr')].map(r => [r.cells[0].textContent, r.cells[1].textContent]));
  return {chips, suspected: kv['Suspected condition']};
}
"""


def expected() -> dict[str, dict[str, str]]:
    facts = json.loads((CACHE / "page.json").read_text(encoding="utf-8"))
    T = pd.Timestamp(facts["snapshot_utc"])
    e = pd.read_parquet(CACHE / "level_episodes.parquet")
    for c in ("opened", "closed", "alarm_ts"):
        e[c] = pd.to_datetime(e[c], utc=True)
    o = e[(e["opened"] <= T) & (e["closed"] > T)]
    wr = pd.read_parquet(CACHE / "work_requests.parquet").set_index("case_id")
    out: dict[str, dict[str, str]] = {}
    for r in o.itertuples():
        d = out.setdefault(r.asset_id, {"Vendor Rules": "OK", "ML Model": "OK", "suspected": None})
        lvl = "ALARM" if pd.notna(r.alarm_ts) and r.alarm_ts <= T else "ALERT"
        d["Vendor Rules" if r.method == "vendor_rules" else "ML Model"] = lvl
        if r.method == "ml_model":
            cid = f"{r.asset_id}|{r.opened:%Y%m%d%H}"
            if cid in wr.index:
                fam, det = wr.at[cid, "suspected_condition"], str(wr.at[cid, "suspected_detail"] or "")
                loc = next(
                    (x for x in ("outer race", "inner race", "ball", "cage") if det.startswith(x)), None
                )
                d["suspected"] = fam
                d["cell"] = (
                    f"bearing, {loc or 'location not determined'}" if fam == "bearing" else fam.lower()
                )
            else:
                d["suspected"] = d["cell"] = "missing"
    return out


def main() -> int:
    want = expected()
    facts = json.loads((CACHE / "page.json").read_text(encoding="utf-8"))
    res: dict = {"assets": {}, "problems": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": WIDTH, "height": 900}, device_scale_factor=1)
        page.goto(PAGE.as_uri())
        page.wait_for_load_state("load")
        sw = page.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
        res["scroll_width"] = sw
        if sw[0] > sw[1]:
            res["problems"].append(f"horizontal scroll at {WIDTH}: {sw}")
        text = page.evaluate("document.body.innerText") + page.content()
        if "WATCH" in text:
            res["problems"].append("WATCH appears on the page")
        rows = page.locator("tr.qrow")
        assets = [rows.nth(i).get_attribute("data-asset") for i in range(rows.count())]
        res["rows"] = assets
        if set(assets) != set(want):
            res["problems"].append(
                f"active list differs from prod levels: page {sorted(assets)} prod {sorted(want)}"
            )
        for a in assets:
            rows.filter(has=page.locator(f"xpath=self::*[@data-asset='{a}']")).first.click(
                position={"x": 30, "y": 8}
            )
            det = page.locator(f"#d-{a}")
            if not det.is_visible():
                res["problems"].append(f"{a}: detail not shown on click")
                continue
            sel = page.locator("tr.qrow.sel").get_attribute("data-asset")
            el = det.element_handle()
            empty = page.evaluate(EMPTY_JS, el)
            overl = page.evaluate(OVERLAP_JS, el)
            head = page.evaluate(HEAD_JS, el)
            sw2 = page.evaluate(
                "[document.documentElement.scrollWidth, document.documentElement.clientWidth]"
            )
            row_chips = (
                rows.filter(has=page.locator(f"xpath=self::*[@data-asset='{a}']"))
                .first.locator("td")
                .nth(1)
                .inner_text()
                .split()[0],
                rows.filter(has=page.locator(f"xpath=self::*[@data-asset='{a}']"))
                .first.locator("td")
                .nth(2)
                .inner_text()
                .split()[0],
            )
            w = want.get(a, {})
            chips = {c["name"]: c["level"] for c in head["chips"]}
            sus = (head["suspected"] or "").split("·")[0].strip().lower()
            cell = (
                rows.filter(has=page.locator(f"xpath=self::*[@data-asset='{a}']"))
                .first.locator("td")
                .nth(4)
                .inner_text()
                .strip()
                .lower()
            )
            insp = (
                rows.filter(has=page.locator(f"xpath=self::*[@data-asset='{a}']"))
                .first.locator("td")
                .nth(5)
                .inner_text()
                .strip()
            )
            ok_chips = chips.get("Vendor Rules") == w.get("Vendor Rules") and chips.get("ML Model") == w.get(
                "ML Model"
            )
            ok_rows = row_chips == (w.get("Vendor Rules"), w.get("ML Model"))
            want_cell = w.get("cell", "-") if w.get("ML Model") != "OK" else "-"
            ok_sus = sus == want_cell and cell == want_cell
            if not insp or insp.lower().startswith("not "):
                res["problems"].append(f"{a}: recommended inspection empty: '{insp}'")
            sections = page.evaluate(
                "(d) => [...d.querySelectorAll('.sech')].map(x => x.childNodes[0].textContent.trim())", el
            )
            res["assets"][a] = {
                "highlighted": sel == a,
                "sections": ["Header", *sections],
                "empty_sections": empty,
                "label_overlaps": overl,
                "chips": chips,
                "row_chips": list(row_chips),
                "prod": w,
                "chips_match": ok_chips and ok_rows,
                "suspected_page": sus,
                "suspected_cell": cell,
                "inspection": insp,
                "suspected_match": ok_sus,
                "scroll": sw2,
            }
            for k, bad in (("empty section", empty), ("label overlap", overl)):
                if bad:
                    res["problems"].append(f"{a}: {k}: {bad}")
            if not (ok_chips and ok_rows):
                res["problems"].append(f"{a}: chips {chips} row {row_chips} prod {w}")
            if not ok_sus:
                res["problems"].append(f"{a}: suspected header '{sus}' cell '{cell}' prod '{want_cell}'")
            if sel != a:
                res["problems"].append(f"{a}: row not highlighted")
            if sw2[0] > sw2[1]:
                res["problems"].append(f"{a}: horizontal scroll {sw2}")
        if "--shots" in sys.argv:
            a = facts["default"]
            rows.filter(has=page.locator(f"xpath=self::*[@data-asset='{a}']")).first.click(
                position={"x": 30, "y": 8}
            )
            page.evaluate("window.scrollTo(0, 0)")
            SHOTS.mkdir(parents=True, exist_ok=True)
            det = page.locator(f"#d-{a}")
            first_chart = det.locator(".chartbox").first.bounding_box()
            frame = page.locator(".frame").bounding_box()
            page.screenshot(
                path=str(SHOTS / "queue.png"),
                full_page=True,
                clip={
                    "x": 0,
                    "y": 0,
                    "width": WIDTH,
                    "height": first_chart["y"] + first_chart["height"] + 12,
                },
            )
            det.screenshot(path=str(SHOTS / "queue_detail.png"))
            res["screenshots"] = {
                "asset": a,
                "queue_height": first_chart["y"] + first_chart["height"] + 12,
                "frame_width": frame["width"],
            }
        browser.close()
    (CACHE / "page_check.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(
        json.dumps(
            {"rows": len(res["rows"]), "problems": res["problems"], "scroll": res["scroll_width"]}, indent=1
        )
    )
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())

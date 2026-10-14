"""Checkpoint (b) rev2 (diag25): assembles the report from the section files. Writes the markdown and the
timeline CSV to Downloads."""

import shutil
from pathlib import Path

import pandas as pd

from cb_common import D, md, note, sec

OUT = Path.home() / "Downloads" / "predictive-maintenance-ml-rev2-cpb_model_report_tables.md"
cv = pd.read_parquet(D / "curves.parquet")
curves = sec("2.3 Operating curves (item 8)") + md(cv) + note(
    "E2 to E7 on prod gold.level_episodes rebuilt at each sweep threshold (ML Model: the prod scores' thr_ columns, 84th to 99.99th percentile, ALERT episodes only; operating point the 99.5th); Vendor Rules: persisted exceedances at each alert offset (alarm at twice it) with the ISO zone rule, through ml_tasks.vendor_levels; operating point 25% of the zone B width"
)
head = (
    "# Checkpoint (b), revision 2: model report results\n\n"
    "Version 13 (pdm.ml.nbm_point_models, autoencoder_vendor [16-4-16], 200 epochs, single threshold) with D2 diagnosis; prod tables read 2026-11-01 (read only). "
    "Evaluation year July 2025 to June 2026. Headline definitions E2 to E7; second set E2 to E4; original + E2, E3. Every table states the rule as coded beneath it.\n\n"
    "Out of scope (2026-11-01): the fleet survival model, the remaining-time model and the supervised layer (not computed; the pipeline's prognosis table is unchanged).\n\n"
)
parts = [head]
for f in ("s1.md", "s2a.md"):
    parts.append((D / f).read_text(encoding="utf-8"))
parts.append((D / "s2b.md").read_text(encoding="utf-8"))
parts.append(curves)
parts.append((D / "s2c.md").read_text(encoding="utf-8"))
parts.append((D / "s3.md").read_text(encoding="utf-8"))
parts.append((D / "s4.md").read_text(encoding="utf-8"))
txt = "".join(parts)
txt = txt.replace("\u2014", ", ").replace("\u2013", "-")
OUT.write_text(txt, encoding="utf-8")
shutil.copy(D / "timeline.csv", Path.home() / "Downloads" / "predictive-maintenance-ml-rev2-cpb_timeline.csv")
print(OUT, len(txt), txt.count("\n"))

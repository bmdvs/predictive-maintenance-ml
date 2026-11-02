"""Checkpoint (b) rev2 (diag25): condition onsets (own start, else truth onset) for the signature panels."""

import json

import pandas as pd

from cb_common import REPO, core

OUT = REPO / "build" / "report_data"
K = core()
B = json.loads((OUT / "bundle.json").read_text())
ct = K["res"][("E7", "ML Model")]["ct"].set_index("wo_id")
for wo in B["signature_cases"]:
    B["signature_cases"][wo]["onset"] = str(ct.at[wo, "onset_ref"]) if wo in ct.index else None
(OUT / "bundle.json").write_text(json.dumps(B, indent=1, default=str))
print(B["signature_cases"])

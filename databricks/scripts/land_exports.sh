#!/usr/bin/env bash
# Copy the local source exports to the landing volume, keeping the folder structure: every system
# folder under the local export root goes to /Volumes/<catalog>/landing/exports/<system>/. Files are
# overwritten, so a rerun lands the same content again. --check compares file counts and bytes per
# system folder between the local exports and the volume, read-only.
#
# Requires: Databricks CLI authenticated to the workspace (profile in ~/.databrickscfg, not in the repo).
# Usage: databricks/scripts/land_exports.sh [PROFILE]          copies the exports
#        databricks/scripts/land_exports.sh [PROFILE] --check  read-only count and byte comparison
set -euo pipefail

PROFILE="${1:-${DATABRICKS_CONFIG_PROFILE:-DEFAULT}}"
MODE="${2:-}"
CATALOG="${PDM_CATALOG:-pdm}"
LOCAL_ROOT="${PDM_LOCAL_EXPORTS:-data/bronze}"
VOLUME="/Volumes/${CATALOG}/landing/exports"

if [[ "${MODE}" == "--check" ]]; then
  python - "${PROFILE}" "${LOCAL_ROOT}" "dbfs:${VOLUME}" <<'PY'
import json
import pathlib
import subprocess
import sys

profile, local_root, volume = sys.argv[1:4]
root = pathlib.Path(local_root)


def remote(path: str) -> tuple[int, int]:
    """Files and bytes under a volume folder, by the Files API listing."""
    files = size = 0
    stack = [path]
    while stack:
        p = stack.pop()
        out = subprocess.run(
            ["databricks", "--profile", profile, "fs", "ls", p, "-l", "-o", "json"],
            capture_output=True, text=True, check=True,
        ).stdout
        for e in json.loads(out or "[]"):
            name = e["name"].rstrip("/")
            full = f"{p}/{name.split('/')[-1]}"
            if e.get("is_directory"):
                stack.append(full)
            else:
                files += 1
                size += int(e.get("size", 0))
    return files, size


ok = True
print(f"{'system':<16}{'local files':>12}{'volume files':>14}{'local bytes':>16}{'volume bytes':>16}  result")
for d in sorted(x for x in root.iterdir() if x.is_dir()):
    lf = [f for f in d.rglob("*") if f.is_file()]
    lb = sum(f.stat().st_size for f in lf)
    rf, rb = remote(f"{volume}/{d.name}")
    match = rf == len(lf) and rb == lb
    ok &= match
    print(f"{d.name:<16}{len(lf):>12}{rf:>14}{lb:>16}{rb:>16}  {'ok' if match else 'DIFF'}")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

for dir in "${LOCAL_ROOT}"/*/; do
  system="$(basename "${dir}")"
  echo "landing ${system}"
  databricks --profile "${PROFILE}" fs cp -r --overwrite "${LOCAL_ROOT}/${system}" "dbfs:${VOLUME}/${system}"
done
echo "done: ${VOLUME}"

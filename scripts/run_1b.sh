#!/bin/bash
# Spec 0010 on the Mac mini: wait for the downloads, build and tokenize the new shelves, check their sizes, add the
# new probes to the v0017 and 0009 evaluations, then train 1B tokens (arm bil). Run from the repo root:
#   nohup caffeinate -dimsu bash scripts/run_1b.sh > runs/run_1b.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/../brain"
UV=${UV:-$HOME/.local/bin/uv}
echo "$(date +%T) waiting for downloads"
until grep -q '^DONE' ../runs/download_1b.log 2>/dev/null; do sleep 60; done
grep FAILED ../runs/download_1b.log && { echo "a download failed"; exit 1; }
while pgrep -f "cerbeto library build" > /dev/null; do sleep 30; done
echo "$(date +%T) building shelves"
$UV run cerbeto library build --shelf s0_dialogue --shelf s2_textbooks --shelf s2_wiki_plus --shelf s3_edu_plus
echo "$(date +%T) tokenizing"
$UV run cerbeto library tokenize
$UV run python - <<'PY'
import json
from cerbeto import paths
need = {"s0_dialogue": 100e6, "s2_textbooks": 50e6, "s2_wiki_plus": 5e6, "s3_edu_plus": 200e6}
ok = True
for s, n in need.items():
    m = json.loads((paths.SHELVES / s / "manifest.json").read_text())
    t = m["tokens"]["train"]
    print(f"{s}: {t / 1e6:.0f}M tokens, valid {m['tokens']['valid'] / 1e6:.1f}M, leaks dropped {m.get('exam_leaks_dropped')}")
    ok &= t >= n
raise SystemExit(0 if ok else "a shelf is smaller than expected — stopping before training")
PY
echo "$(date +%T) probes for v0017 and mid"
$UV run python -m cerbeto.train.final_test probes v0017
$UV run python -m cerbeto.train.final_test probes mid
echo "$(date +%T) training bil"
$UV run python -m cerbeto.train.final_test train bil --tokens 1e9

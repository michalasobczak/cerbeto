#!/bin/bash
# Downloads the extra library sources of spec 0010 (one billion tokens). Resumable: curl -C -
# Usage: ./scripts/download_library_1b.sh [probe]   — "probe" fetches one file per source to measure yields first.
set -u
HF=https://huggingface.co/datasets
D=${LIBRARY_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/library}
MODE=${1:-all}
get(){ mkdir -p "$(dirname "$2")"; echo ">> $2"; curl -sSL --retry 5 -C - -o "$2" "$1" || echo "FAILED $1"; }

# SODA — social conversations of ~8 turns (CC BY 4.0)
get "$HF/allenai/soda/resolve/main/train.parquet" "$D/soda/train.parquet"
get "$HF/allenai/soda/resolve/main/README.md" "$D/soda/README.md"

# Cosmopedia — synthetic stories, filtered later to audience young_children (Apache 2.0); web_samples_v2 has no
# young_children audience (measured 2026-11-07), so only the stories subset
get "$HF/HuggingFaceTB/cosmopedia/resolve/main/README.md" "$D/cosmopedia/README.md"
n_stories=43
[ "$MODE" = probe ] && n_stories=1
for i in $(seq 0 $((n_stories - 1))); do
  f=$(printf "train-%05d-of-00043.parquet" "$i")
  get "$HF/HuggingFaceTB/cosmopedia/resolve/main/data/stories/$f" "$D/cosmopedia/stories/$f"
done

# FineWeb-Edu — the other 13 shards of the 10BT sample (ODC-By 1.0)
[ "$MODE" = probe ] || for i in $(seq 1 13); do
  f=$(printf "%03d_00000.parquet" "$i")
  get "$HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/$f" "$D/fineweb_edu/sample10BT_$f"
done
echo DONE; du -sh "$D"/*

#!/bin/bash
# Pobieranie półek biblioteki (poza gitem). Wznawialne: curl -C -
set -u
HF=https://huggingface.co/datasets
D=${LIBRARY_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/library}
get(){ echo ">> $2"; curl -sSL --retry 5 -C - -o "$2" "$1" || echo "FAILED $1"; }
for f in bnc_spoken childes gutenberg open_subtitles simple_wiki switchboard; do
  get "$HF/BabyLM-community/BabyLM-2026-Strict/resolve/main/$f.train.txt" "$D/babylm/$f.train.txt"
done
get "$HF/BabyLM-community/BabyLM-2026-Strict/resolve/main/README.md" "$D/babylm/README.md"
get "$HF/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" "$D/tinystories/TinyStoriesV2-GPT4-train.txt"
get "$HF/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" "$D/tinystories/TinyStoriesV2-GPT4-valid.txt"
get "$HF/roneneldan/TinyStories/resolve/main/README.md" "$D/tinystories/README.md"
get "$HF/wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet" "$D/simplewiki/simplewiki-20231101.parquet"
get "$HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet" "$D/fineweb_edu/sample10BT_000_00000.parquet"
echo DONE; ls -la "$D"/*

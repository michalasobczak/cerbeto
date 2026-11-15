#!/bin/bash
# Downloads the library sources of spec 0016 (ten billion tokens) straight onto the training machine. Resumable: curl -C -
# Usage: LIBRARY_DIR=... ./scripts/download_library_10b.sh   (default: data/library next to the repo's brain/)
set -u
HF=https://huggingface.co/datasets
D=${LIBRARY_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data/library}
WEB_SHARDS=${WEB_SHARDS:-24}          # of web_samples_v2's 118 files (~0.28 GB each)
get(){ mkdir -p "$(dirname "$2")"; [ -s "$2" ] && [ ! -f "$2.part" ] && return; echo ">> $2"; touch "$2.part";
       curl -sSL --retry 8 -C - -o "$2" "$1" && rm -f "$2.part" || echo "FAILED $1"; }
list(){ curl -s "https://huggingface.co/api/datasets/HuggingFaceTB/cosmopedia/tree/main/data/$1" |
        python3 -c "import json,sys; print('\n'.join(sorted(x['path'] for x in json.load(sys.stdin) if x['path'].endswith('.parquet'))))"; }

# FineWeb-Edu — the 14 shards of the 10BT sample (ODC-By 1.0); 0014 used their paragraphs up to grade 8
for i in $(seq 0 13); do
  f=$(printf "%03d_00000.parquet" "$i")
  get "$HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/$f" "$D/fineweb_edu/sample10BT_$f"
done
# Cosmopedia (Apache 2.0): every audience of the stories, the textbook subsets and part of web_samples_v2
get "$HF/HuggingFaceTB/cosmopedia/resolve/main/README.md" "$D/cosmopedia/README.md"
for s in stories stanford openstax khanacademy wikihow auto_math_text; do
  for p in $(list $s); do get "$HF/HuggingFaceTB/cosmopedia/resolve/main/$p" "$D/cosmopedia/${p#data/}"; done
done
for p in $(list web_samples_v2 | head -n "$WEB_SHARDS"); do
  get "$HF/HuggingFaceTB/cosmopedia/resolve/main/$p" "$D/cosmopedia/${p#data/}"
done
echo DONE; du -sh "$D"/*/ "$D"/cosmopedia/*/

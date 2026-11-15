#!/bin/bash
# Spec 0016 on the training machine: wait for the downloads, build the three new shelves, cut them with the 16k
# tokenizer, print their sizes. Run in the background: nohup scripts/final/bil10_data.sh > ~/cerbeto/logs/bil10_data.log 2>&1 &
set -e
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../../brain"
LOG=${DOWNLOAD_LOG:-$HOME/cerbeto/logs/download10b.log}
LIB=../data/library
fineweb_ready() { [ "$(ls $LIB/fineweb_edu/sample10BT_*.parquet 2>/dev/null | wc -l)" -ge 14 ] && ! ls $LIB/fineweb_edu/*.part > /dev/null 2>&1; }
until fineweb_ready; do sleep 60; done
echo "fineweb ready $(date)"
uv run cerbeto library build --shelf s4_edu_12
uv run python -c "from cerbeto.library.tokenize import tokenize_all; tokenize_all(vocab=16384, shelves=['s4_edu_12'])"
until grep -q '^DONE' "$LOG"; do sleep 60; done
echo "downloads done $(date)"
for s in s4_stories s4_textbooks; do
  uv run cerbeto library build --shelf $s
  uv run python -c "from cerbeto.library.tokenize import tokenize_all; tokenize_all(vocab=16384, shelves=['$s'])"
done
for s in s4_edu_12 s4_stories s4_textbooks; do
  ls -l ../data/shelves/$s/
  python3 -c "import os; print('$s', 'train tokens', os.path.getsize('../data/shelves/$s/train_16384.bin') // 2)"
done
echo "shelves ready $(date)"

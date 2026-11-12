#!/bin/bash
# spec 0014 on the M6: wait for vext, build the grade 7-8 edu shelf, cut it with the 16k tokenizer, then 1B tokens of bil2.
set -e
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd "$(dirname "$0")/../../brain"
while pgrep -f "final_test train vext" > /dev/null; do sleep 60; done
echo "vext done $(date)"
uv run cerbeto library build --shelf s3_edu_8
uv run python -c "from cerbeto.library.tokenize import tokenize_all; tokenize_all(vocab=16384, shelves=['s3_edu_8'])"
ls -l ../data/shelves/s3_edu_8/
echo "shelf ready $(date)"
caffeinate -dimsu uv run python -m cerbeto.train.final_test train bil2 --tokens 1e9

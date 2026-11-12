#!/bin/bash
# spec 0012 vext on the M6: re-cut the 1B shelves with tokenizer_12288.json, then 60M tokens from v6 (talk2/final).
set -e
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd "$(dirname "$0")/../../brain"
uv run python -m cerbeto.tokenizer.vocab tokenize --size 12288
caffeinate -dimsu uv run python -m cerbeto.train.final_test train vext --tokens 60e6

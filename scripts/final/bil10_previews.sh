#!/usr/bin/env bash
# Spec 0016, from the MacBook (the only machine that reaches both the training machines and the Mac mini): every 2B tokens of the
# bil10 run, copy the coordinator's checkpoint to the mini and add it to the chat Space as a preview (to_llama → ONNX int8 →
# publish_space). The mini holds the HF token; a preview already on the mini (checkpoints/final/bil10/preview-<N>B) is skipped.
#   scripts/final/bil10_previews.sh        (machines from scripts/gpu/hosts.env, the mini as MINI=user@host)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../gpu/hosts.env"
c="$GPU_USER@$COORD"; m=${MINI:?MINI=user@host of the Mac mini}
run=cerbeto/model/checkpoints/final/bil10 repo=Work/indatify/platformy/model
START=2485                                          # million tokens read before bil10 (v6 1423 + v16k 60 + bil2-avg 1002)
for n in 2 4 6 8 10; do
  dst="$repo/checkpoints/final/bil10/preview-${n}B"
  ssh "$m" "test -f $dst/model.safetensors" && { echo "${n}B already there"; continue; }
  until t=$(ssh -o ConnectTimeout=15 "$c" "python3 -c \"import json; print(json.load(open('cerbeto/model/runs/final/bil10/state.json'))['tokens'])\"") \
        && [ "$t" -ge $((n * 1000000000)) ]; do sleep 600; done
  src=$run/latest; [ "$n" = 10 ] && src=$run/final
  ssh "$m" "mkdir -p $dst"
  for try in 1 2 3; do                             # latest is replaced every 30 minutes: weights and metadata from one save
    scp -3 -q "$c:$src/architecture.json" "$m:$dst/a1.json" && scp -3 -q "$c:$src/model.safetensors" "$m:$dst/" \
      && scp -3 -q "$c:$src/architecture.json" "$m:$dst/architecture.json" && ssh "$m" "cmp -s $dst/a1.json $dst/architecture.json" && break
    sleep 60
  done
  life=$(ssh "$m" "python3 -c \"import json; print(round(($START * 1e6 + json.load(open('$dst/architecture.json'))['final_test']['tokens']) / 1e6))\"")
  echo "preview ${n}B: ${life}M tokens in all $(date)"
  ssh "$m" "cd $repo && export PATH=\$HOME/.local/bin:\$PATH && rm -rf /tmp/web/cerbeto_hf \
    && uv run --no-project --python 3.12 --with torch --with transformers --with safetensors --with tokenizers --with huggingface_hub python scripts/hf/to_llama.py /tmp/web/cerbeto_hf $dst \
    && uv run --no-project --python 3.12 --with optimum-onnx --with onnxruntime --with torch python scripts/hf/export_web.py \
    && cd brain && uv run --with huggingface_hub python ../scripts/hf/publish_space.py bil10-${life}M --short preview --tokens ${life}e6 \
       --label 'preview - $(echo "scale=1; $life / 1000" | bc)B tokens (10B run, ${n}B of 10B)' \
       --note 'Preview, not a release: the run of ten billion tokens of new text (FineWeb-Edu grade 8-12, Cosmopedia), ${n}B of 10B'" 2>&1 \
    | grep -E "logit|identical|int8|PUBLISHED|Error|error" || { echo "preview ${n}B failed"; ssh "$m" "rm -rf $dst"; exit 1; }
done
echo "all previews published"

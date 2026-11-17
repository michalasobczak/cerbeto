#!/usr/bin/env bash
# Spec 0016, from the MacBook (the only machine that reaches both the training machines and the Mac mini): at each threshold of the
# bil10 run (million tokens), copy the coordinator's checkpoint to the mini and add it to the chat Space as a preview (to_llama →
# ONNX int8 → publish_space). The mini holds the HF token; a preview already on the mini (checkpoints/final/bil10/preview-<N>) is skipped.
#   scripts/final/bil10_previews.sh [MTOKENS ...]     (default 2000 4000 6000 8000 10000; machines from scripts/gpu/hosts.env, MINI=user@host)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../gpu/hosts.env"
c="$GPU_USER@$COORD"; m=${MINI:?MINI=user@host of the Mac mini}
run=cerbeto/model/checkpoints/final/bil10 repo=Work/indatify/platformy/model
START=2485                                          # million tokens read before bil10 (v6 1423 + v16k 60 + bil2-avg 1002)
for n in ${@:-2000 4000 6000 8000 10000}; do
  name=$([ $((n % 1000)) = 0 ] && echo "$((n / 1000))B" || echo "${n}M")
  dst="$repo/checkpoints/final/bil10/preview-$name"
  ssh "$m" "test -f $dst/model.safetensors" && { echo "$name already there"; continue; }
  src=$run/latest; [ "$n" -ge 10000 ] && src=$run/final
  # wait for a saved checkpoint (not just the live state) past the threshold
  until t=$(ssh -o ConnectTimeout=15 "$c" "python3 -c \"import json; print(json.load(open('$src/architecture.json'))['final_test']['tokens'])\"" 2>/dev/null) \
        && [ "$t" -ge $((n * 1000000)) ]; do sleep 300; done
  ssh "$m" "mkdir -p $dst"
  for try in 1 2 3; do                             # latest is replaced every 30 minutes: weights and metadata from one save
    scp -3 -q "$c:$src/architecture.json" "$m:$dst/a1.json" && scp -3 -q "$c:$src/model.safetensors" "$m:$dst/" \
      && scp -3 -q "$c:$src/architecture.json" "$m:$dst/architecture.json" && ssh "$m" "cmp -s $dst/a1.json $dst/architecture.json" && break
    sleep 60
  done
  life=$(ssh "$m" "python3 -c \"import json; print(round(($START * 1e6 + json.load(open('$dst/architecture.json'))['final_test']['tokens']) / 1e6))\"")
  ran=$(printf %.1f "$(echo "($life - $START) / 1000" | bc -l)")                # the run's own tokens: latest may be past the threshold
  echo "preview $name: ${ran}B of the run, ${life}M tokens in all $(date)"
  ssh "$m" "cd $repo && export PATH=\$HOME/.local/bin:\$PATH && rm -rf /tmp/web/cerbeto_hf \
    && uv run --no-project --python 3.12 --with torch --with transformers --with safetensors --with tokenizers --with huggingface_hub python scripts/hf/to_llama.py /tmp/web/cerbeto_hf \$HOME/$dst \
    && uv run --no-project --python 3.12 --with optimum-onnx --with onnxruntime --with torch python scripts/hf/export_web.py \
    && cd brain && uv run --with huggingface_hub python ../scripts/hf/publish_space.py bil10-${life}M --short preview --tokens ${life}e6 --replace \
       --label 'preview - $(printf %.1f "$(echo "$life / 1000" | bc -l)")B tokens (10B run, ${ran}B of 10B)' \
       --note 'Preview, not a release: the run of ten billion tokens of new text (FineWeb-Edu grade 8-12, Cosmopedia), ${ran}B of 10B'" 2>&1 \
    | grep -E "logit|identical|int8|PUBLISHED|Error|error" || { echo "preview $name failed"; ssh "$m" "rm -rf $dst"; exit 1; }
done
echo "all previews published"

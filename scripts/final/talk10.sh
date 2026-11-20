#!/usr/bin/env bash
# Spec 0017, from the MacBook: when bil10 has ended, copy bil10/final from the coordinator to the data host (shelves and
# data/talk are there), tune it there on one card (arm talk10 or talk10b, 50M tokens), copy <arm>/final to the Mac mini and run the
# quick benchmark there against v6 and bil10/final. Machines from scripts/gpu/hosts.env (COORD, DATA_HOST, MINI).
#   scripts/final/talk10.sh [TOKENS] [ARM]     (default 50e6 talk10; GPU=0; e.g. 50e6 talk10b)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../gpu/hosts.env"
u=$GPU_USER c="$GPU_USER@$COORD" d="$GPU_USER@$DATA_HOST" m=${MINI:?MINI=user@host of the Mac mini}
tokens=${1:-50e6} arm=${2:-talk10} gpu=${GPU:-0}
ck=cerbeto/model/checkpoints/final repo=Work/indatify/platformy/model
s(){ for try in 1 2 3 4 5; do ssh -o ConnectTimeout=15 "$1" "$2" && return 0; sleep 30; done; return 1; }

# 1. bil10 has ended: its final checkpoint is saved and no DiLoCo process is left on the data host
until s "$c" "test -f $ck/bil10/final/model.safetensors" && ! s "$d" 'pgrep -f "[c]erbeto.train.diloco" > /dev/null'; do sleep 300; done
echo "bil10 ended $(date)"

# 2. the start checkpoint on the data host, the code with arm talk10
s "$d" "mkdir -p $ck/bil10 ~/cerbeto/logs && rm -f ~/cerbeto/logs/$arm.log && cd ~/cerbeto/model && git fetch -q origin && git reset -q --hard origin/main" || { echo "data host unreachable"; exit 1; }
s "$d" "test -s $ck/bil10/final/model.safetensors" || scp -3 -rq "$c:$ck/bil10/final" "$d:$ck/bil10/"; s "$d" "ls -l $ck/bil10/final && test -s $ck/bil10/final/model.safetensors" \
  || { echo "copy of bil10/final failed"; exit 1; }

# 3. tune on one card
s "$d" "cd ~/cerbeto/model/brain && CUDA_VISIBLE_DEVICES=$gpu nohup ~/.local/bin/uv run python -m cerbeto.train.final_test train $arm --tokens $tokens --no-recompute > ~/cerbeto/logs/$arm.log 2>&1 < /dev/null &"
echo "$arm started $(date)"
until s "$d" "grep -q -E '$arm: finished|Traceback|interrupted' ~/cerbeto/logs/$arm.log"; do sleep 300; done
s "$d" "grep -q '$arm: finished' ~/cerbeto/logs/$arm.log" || { echo "$arm failed"; s "$d" "tail -20 ~/cerbeto/logs/$arm.log"; exit 1; }
s "$d" "tail -3 ~/cerbeto/logs/$arm.log"

# 4. to the mini (bil10/final too, for the comparison) and the quick benchmark
for run in bil10 $arm; do
  ssh "$m" "test -s $repo/checkpoints/final/$run/final/model.safetensors" && [ $run = bil10 ] && continue
  ssh "$m" "mkdir -p $repo/checkpoints/final/$run"
  scp -3 -rq "$d:$ck/$run/final" "$m:$repo/checkpoints/final/$run/" || { echo "copy of $run to the mini failed"; exit 1; }
done
echo "on the mini $(date)"
ssh "$m" "cd $repo/brain && export PATH=\$HOME/.local/bin:\$PATH && caffeinate -i uv run --with huggingface_hub python ../scripts/hf/bench.py \
  v6-1032M checkpoints/final/bil10/final checkpoints/final/$arm/final 2>&1 | grep -v Warning | tail -9"

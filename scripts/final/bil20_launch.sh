#!/usr/bin/env bash
# Spec 0018, from the MacBook: give every GPU machine the start checkpoint (bil10/final, on the coordinator and the
# data host since 0017), then start the DiLoCo run `bil20` as bil10 was started (0016). Machines from
# scripts/gpu/hosts.env (COORD, DATA_HOST, GPUS, SLOTS, VALIDATOR).
#   scripts/final/bil20_launch.sh [TOKENS]     (default 2e10)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../gpu/hosts.env"
u=$GPU_USER tokens=${1:-2e10} ck=cerbeto/model/checkpoints/final
s(){ for try in 1 2 3 4 5; do ssh -o ConnectTimeout=15 "$u@$1" "$2" && return 0; sleep 20; done; return 1; }
for h in $COORD $GPUS; do
  s "$h" "test -s $ck/bil10/final/model.safetensors" && continue
  s "$h" "mkdir -p $ck/bil10" || { echo "$h unreachable"; exit 1; }
  scp -3 -rq "$u@$DATA_HOST:$ck/bil10/final" "$u@$h:$ck/bil10/" || { echo "copy of bil10/final to $h failed"; exit 1; }
  s "$h" "ls -l $ck/bil10/final" || exit 1
done
echo "bil10/final on every machine $(date)"
cd "$here/../.." && EVERY=500 OUTER_LR=1 OUTER_M=0 VALIDATOR="$VALIDATOR" scripts/gpu/diloco.sh bil20 bil20 "$tokens" --no-recompute --no-validate

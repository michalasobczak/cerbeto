#!/usr/bin/env bash
# Spec 0015: start a DiLoCo run from the MacBook — a coordinator (no GPU needed) and one worker per GPU slot. The
# machines are in scripts/gpu/hosts.env (not in git): GPU_USER, COORD (ssh address), COORD_LAN (the address workers
# reach it at), GPUS ("host host"), SLOTS ("host:gpu ..."). Logs: ~/cerbeto/logs/<name>-*.log on each host.
#   [EVERY=500] [SLOTS="host:gpu ..."] scripts/gpu/diloco.sh bil2 d1 1e8 [extra worker flags]
#   scripts/gpu/diloco.sh stop
# OUTER_LR / OUTER_M: the outer step (1 / 0 = plain averaging; 2026-11-14: 0.7 / 0.9 overshot with 1-3 workers).
# SLOTS: host:gpu per worker. VALIDATOR=host:gpu runs the validation on its own card and
# the workers skip it (pass --no-validate among the worker flags).
set -euo pipefail
source "$(dirname "$0")/hosts.env"
user=$GPU_USER
coord=$COORD
coord_lan=$COORD_LAN
read -r -a gpus <<< "$GPUS"
read -r -a slots <<< "$SLOTS"
if [ "${1:-}" = "stop" ]; then
  for h in "${gpus[@]}" "$coord"; do ssh "$user@$h" 'pkill -INT -f "[c]erbeto.train.diloco" || true'; done
  exit 0
fi
on() {                                              # on HOST COMMAND: start in the background, three tries
  for try in 1 2 3; do ssh -o ConnectTimeout=10 -f "$user@$1" "$2" && return 0; sleep 10; done
  echo "cannot start on $1"; return 1
}
arm=$1; name=$2; tokens=$3; shift 3
extra="$*"
for h in "$coord" "${gpus[@]}"; do                 # the VPN drops now and then: three tries per host
  for try in 1 2 3; do
    ssh -o ConnectTimeout=10 "$user@$h" 'cd ~/cerbeto/model && git fetch -q origin && git reset -q --hard origin/main && mkdir -p ~/cerbeto/logs' && break
    [ "$try" = 3 ] && { echo "cannot reach $h"; exit 1; }
    sleep 10
  done
done
on "$coord" "cd ~/cerbeto/model/brain && nohup ~/.local/bin/uv run python -m cerbeto.train.diloco coordinator $arm --name $name --tokens $tokens --workers ${#slots[@]} --every ${EVERY:-500} --outer-lr ${OUTER_LR:-0.7} --outer-momentum ${OUTER_M:-0.9} > ~/cerbeto/logs/$name-coordinator.log 2>&1 < /dev/null &"
sleep 20                                        # the coordinator loads the start checkpoint
rank=0
for slot in "${slots[@]}"; do
  h=${slot%%:*}; gpu=${slot##*:}
  on "$h" "cd ~/cerbeto/model/brain && CUDA_VISIBLE_DEVICES=$gpu nohup ~/.local/bin/uv run python -m cerbeto.train.diloco worker $arm --coordinator $coord_lan --rank $rank $extra > ~/cerbeto/logs/$name-w$rank.log 2>&1 < /dev/null &"
  rank=$((rank + 1))
done
if [ -n "${VALIDATOR:-}" ]; then
  h=${VALIDATOR%%:*}; gpu=${VALIDATOR##*:}
  on "$h" "cd ~/cerbeto/model/brain && CUDA_VISIBLE_DEVICES=$gpu nohup ~/.local/bin/uv run python -m cerbeto.train.diloco validator $arm --coordinator $coord_lan > ~/cerbeto/logs/$name-validator.log 2>&1 < /dev/null &"
fi
echo "started $name: coordinator on $coord, workers 0-$((rank - 1))"

#!/usr/bin/env bash
# Spec 0015: start a DiLoCo run from the MacBook — the coordinator on .13, two workers (one per RTX 3060) on .14 and
# .15. Logs: ~/cerbeto/logs/<name>-*.log on each host. Stop: scripts/gpu/diloco.sh stop.
#   [EVERY=500] [SLOTS="10.10.12.14:1 10.10.12.15:0"] scripts/gpu/diloco.sh bil2 d1 1e8 [extra worker flags]
# SLOTS: host:gpu per worker (default all four cards).
set -euo pipefail
user=sobczam
coord=10.10.12.13
coord_lan=172.31.115.13
gpus=(10.10.12.14 10.10.12.15)
read -r -a slots <<< "${SLOTS:-10.10.12.14:0 10.10.12.14:1 10.10.12.15:0 10.10.12.15:1}"
if [ "${1:-}" = "stop" ]; then
  for h in "${gpus[@]}" "$coord"; do ssh "$user@$h" 'pkill -INT -f "[c]erbeto.train.diloco" || true'; done
  exit 0
fi
arm=$1; name=$2; tokens=$3; shift 3
extra="$*"
for h in "$coord" "${gpus[@]}"; do ssh "$user@$h" 'cd ~/cerbeto/model && git pull -q --ff-only && mkdir -p ~/cerbeto/logs'; done
ssh -f "$user@$coord" "cd ~/cerbeto/model/brain && nohup ~/.local/bin/uv run python -m cerbeto.train.diloco coordinator $arm --name $name --tokens $tokens --workers ${#slots[@]} --every ${EVERY:-500} > ~/cerbeto/logs/$name-coordinator.log 2>&1 < /dev/null &"
sleep 20                                        # the coordinator loads the start checkpoint
rank=0
for slot in "${slots[@]}"; do
  h=${slot%%:*}; gpu=${slot##*:}
  ssh -f "$user@$h" "cd ~/cerbeto/model/brain && CUDA_VISIBLE_DEVICES=$gpu nohup ~/.local/bin/uv run python -m cerbeto.train.diloco worker $arm --coordinator $coord_lan --rank $rank $extra > ~/cerbeto/logs/$name-w$rank.log 2>&1 < /dev/null &"
  rank=$((rank + 1))
done
echo "started $name: coordinator on $coord, workers 0-$((rank - 1))"

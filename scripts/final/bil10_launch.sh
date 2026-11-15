#!/usr/bin/env bash
# Spec 0016, from the MacBook: wait until the training machine has built the new shelves (bil10_data.sh), send them
# to the second GPU machine over their LAN (tar | nc), give it the start checkpoint, then start the DiLoCo run.
# Machines from scripts/gpu/hosts.env plus DATA_HOST (where the shelves are built), PEERS ("ssh-host:lan-ip ...").
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../gpu/hosts.env"
u=$GPU_USER
s(){ for try in 1 2 3 4 5; do ssh -o ConnectTimeout=15 "$u@$1" "$2" && return 0; sleep 20; done; return 1; }
until s "$DATA_HOST" 'grep -q "^shelves ready" ~/cerbeto/logs/bil10_data.log'; do
  s "$DATA_HOST" 'grep -q -E "Traceback|Error" ~/cerbeto/logs/bil10_data.log' && { echo "shelf build failed"; exit 1; }
  sleep 300
done
echo "shelves ready $(date)"
s "$DATA_HOST" 'tail -8 ~/cerbeto/logs/bil10_data.log'
for peer in $PEERS; do
  h=${peer%%:*}; ip=${peer##*:}
  s "$h" "mkdir -p ~/cerbeto/model/data/shelves ~/cerbeto/model/checkpoints/final/bil2-avg; pkill -f 'nc -l 5999' ; (nc -l 5999 | tar x -C ~/cerbeto/model/data/shelves) > /dev/null 2>&1 < /dev/null &"
  sleep 5
  s "$DATA_HOST" "cd ~/cerbeto/model/data/shelves && tar c --exclude=train.jsonl s4_edu_12 s4_stories s4_textbooks | nc -N $ip 5999" || { echo "copy to $h failed"; exit 1; }
  s "$h" 'du -sh ~/cerbeto/model/data/shelves/s4_*'
  scp -3 -rq "$u@$DATA_HOST:cerbeto/model/checkpoints/final/bil2-avg/final" "$u@$h:cerbeto/model/checkpoints/final/bil2-avg/"
done
echo "copied $(date)"
cd "$here/../.." && EVERY=500 OUTER_LR=1 OUTER_M=0 VALIDATOR="$VALIDATOR" scripts/gpu/diloco.sh bil10 bil10 1e10 --no-recompute --no-validate

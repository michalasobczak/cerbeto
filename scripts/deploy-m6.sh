#!/bin/sh
# Copy the code (not data, checkpoints or runs) from this dev copy to the Mac mini M6, the main machine.
set -e
HOST=${CERBETO_HOST:-michalsobczak@10.10.11.71}
cd "$(dirname "$0")/.."
rsync -a --exclude .venv --exclude __pycache__ --exclude .remember --exclude .git \
  --exclude /data --exclude /checkpoints --exclude /runs ./ "$HOST:Work/indatify/platformy/model/"
echo "deployed to $HOST"

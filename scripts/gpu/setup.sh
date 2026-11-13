#!/usr/bin/env bash
# Spec 0015: install uv, clone the repo and its Python environment on a GPU (or coordinator) machine. Run from the
# MacBook: scripts/gpu/setup.sh 10.10.12.15 [user]. Idempotent; installs nothing outside ~ (no sudo there).
set -euo pipefail
host=$1
user=${2:-sobczam}
ssh "$user@$host" bash -s <<'EOF'
set -euo pipefail
if [ ! -x ~/.local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh > /tmp/uvinst.log 2>&1
fi
mkdir -p ~/cerbeto
cd ~/cerbeto
if [ ! -d model ]; then
  git clone -q https://github.com/michalasobczak/cerbeto.git model
fi
cd model && git pull -q --ff-only
cd brain && ~/.local/bin/uv sync -q
~/.local/bin/uv run python -c "import torch; print(torch.__version__, 'cuda devices:', torch.cuda.device_count())"
EOF

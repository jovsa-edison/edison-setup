#!/usr/bin/env bash
# voltage_park_setup.sh — first-time per-user setup on the Voltage Park cluster.
#
# Installs:
#   - uv (Python package manager)        → ~/.local/bin/uv
#   - claude (Claude Code CLI)           → ~/.local/bin/claude
#
# Both binaries land under ~/.local/bin, which sits on the /data NFS via the
# admin-managed /home/<you> -> /data/users/<you> symlink — so any worker node
# Slurm hands you sees them too, no per-host installs needed.
#
# Idempotent: re-running skips tools that are already installed.
#
# Usage (run on the cluster after first ssh):
#   bash voltage_park_setup.sh

set -euo pipefail

bin_dir="$HOME/.local/bin"
mkdir -p "$bin_dir"
export PATH="$bin_dir:$PATH"

step() { printf '\n=== %s ===\n' "$*"; }

# --- uv ---------------------------------------------------------------------
step "uv"
if command -v uv >/dev/null 2>&1; then
  echo "[skip] uv already installed: $(uv --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  hash -r
  echo "[ok] uv installed: $(uv --version)"
fi

# --- claude code ------------------------------------------------------------
step "claude code"
if command -v claude >/dev/null 2>&1; then
  echo "[skip] claude already installed: $(claude --version 2>&1 | head -1)"
else
  curl -fsSL https://claude.ai/install.sh | bash
  hash -r
  echo "[ok] claude installed: $(claude --version 2>&1 | head -1)"
fi

# --- summary ----------------------------------------------------------------
step "done"
cat <<EOF
Installed under: $bin_dir  (on /data NFS, visible from every worker pod)

If a future shell doesn't see the new binaries, the installers usually patch
~/.bashrc themselves; if not, add:
  export PATH="\$HOME/.local/bin:\$PATH"

Next:
  - Re-ssh (or 'source ~/.bashrc') to refresh PATH in this shell.
  - In the edison-setup repo on the cluster: 'uv sync --locked' to
    materialize .venv from the committed lockfile.
  - Run 'claude' once to log in.
EOF

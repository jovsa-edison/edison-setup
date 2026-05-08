#!/usr/bin/env bash
# voltage_park_setup.sh — first-time per-user setup on the Voltage Park cluster.
#
# Installs:
#   - uv (Python package manager)        → ~/.local/bin/uv
#   - claude (Claude Code CLI)           → ~/.local/bin/claude
#
# Configures shell:
#   - ~/.bashrc          → puts ~/.local/bin on PATH for interactive shells
#   - ~/.bash_profile    → sources ~/.bashrc on login (ssh)
#
# Both binaries land under ~/.local/bin, which sits on the /data NFS via the
# admin-managed /home/<you> -> /data/users/<you> symlink — so any worker node
# Slurm hands you sees them too, no per-host installs needed.
#
# Idempotent: re-running skips tools and rc-file lines that are already in place.
#
# Usage (run on the cluster after first ssh):
#   bash voltage_park_setup.sh
#   source ~/.bashrc       # or open a new shell — `claude` and `uv` should now work

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

# --- shell init -------------------------------------------------------------
# Make sure ~/.local/bin is on PATH for every future shell.
#
# Two files matter:
#   ~/.bashrc        → sourced for interactive shells; we put PATH there.
#   ~/.bash_profile  → sourced for login shells (ssh); just sources .bashrc.
#
# Each line is appended only if missing — safe to re-run.
step "shell init"

ensure_line() {
  # Append a line to a file iff that exact line isn't already present.
  local line="$1" file="$2"
  if [ -f "$file" ] && grep -qxF -- "$line" "$file"; then
    echo "[skip] $file already contains: $line"
  else
    [ -f "$file" ] || touch "$file"
    printf '%s\n' "$line" >> "$file"
    echo "[ok]  appended to $file: $line"
  fi
}

ensure_line 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"
ensure_line '[[ -f ~/.bashrc ]] && source ~/.bashrc' "$HOME/.bash_profile"

# --- summary ----------------------------------------------------------------
step "done"
cat <<EOF
Installed under: $bin_dir  (on /data NFS, visible from every worker pod)
Shell rc files:  ~/.bashrc, ~/.bash_profile

To use 'uv' and 'claude' in THIS shell right now:
  source ~/.bashrc

Future ssh sessions pick up PATH automatically.

Next:
  - In the edison-setup repo on the cluster: 'uv sync --locked' to
    materialize .venv from the committed lockfile.
  - Run 'claude' once to log in.
EOF

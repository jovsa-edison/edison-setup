#!/usr/bin/env bash
# Voltage Park cluster end-to-end setup smoke test.
# Usage: ./voltage_park_smoke.sh <host-alias>      e.g. ./voltage_park_smoke.sh g0287
#        ./voltage_park_smoke.sh g0287 --skip-srun  to skip the GPU/worker check

set -u

HOST="${1:-}"
SKIP_SRUN="${2:-}"

if [[ -z "$HOST" ]]; then
  echo "usage: $0 <ssh-host-alias> [--skip-srun]" >&2
  exit 2
fi

if [[ -t 1 ]]; then
  G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; D=$'\e[2m'; N=$'\e[0m'
else
  G=""; R=""; Y=""; D=""; N=""
fi

PASS=0
FAIL=0
SKIP=0

pass() { echo "${G}[PASS]${N} $1"; PASS=$((PASS+1)); }
fail() { echo "${R}[FAIL]${N} $1${2:+ ${D}- $2${N}}"; FAIL=$((FAIL+1)); }
skip() { echo "${Y}[SKIP]${N} $1${2:+ ${D}- $2${N}}"; SKIP=$((SKIP+1)); }

echo "=== client checks ==="

# 1. ssh config has the host
if grep -qE "^[Hh]ost[[:space:]]+.*\b${HOST}\b" "$HOME/.ssh/config" 2>/dev/null; then
  pass "~/.ssh/config has an entry matching '$HOST'"
else
  fail "no '$HOST' entry in ~/.ssh/config"
fi

# 2. some private key exists (runbook suggests voltage-park_id_ed25519, but accept any)
if compgen -G "$HOME/.ssh/voltage-park_id_*" > /dev/null || compgen -G "$HOME/.ssh/id_*" > /dev/null; then
  pass "ssh private key present in ~/.ssh"
else
  fail "no ssh private key found in ~/.ssh"
fi

# 3. non-interactive ssh actually works
if ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" true 2>/dev/null; then
  pass "ssh $HOST connects non-interactively"
else
  fail "cannot ssh $HOST" "key not added by Voltage Park, or wrong host?"
  echo
  echo "${R}aborting - remaining checks require working ssh${N}"
  echo "${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
  exit 1
fi

echo
echo "=== server checks (on $HOST) ==="

# Run remote checks in a single ssh session. Each line of output is one check
# in the form "STATUS|description|detail" - parsed below.
REMOTE_OUTPUT=$(ssh -o BatchMode=yes "$HOST" bash -s <<'REMOTE_EOF'
emit() { printf '%s|%s|%s\n' "$1" "$2" "${3:-}"; }

# default shell is bash
case "$(getent passwd "$USER" | cut -d: -f7)" in
  */bash) emit PASS "default shell is bash" ;;
  *)      emit FAIL "default shell is not bash" "$(getent passwd "$USER" | cut -d: -f7)" ;;
esac

# /data/$USER exists
[[ -d "/data/$USER" ]] \
  && emit PASS "/data/$USER exists" \
  || emit FAIL "/data/$USER missing" "run the symlink block from the runbook"

# expected symlinks resolve into /data
check_symlink() {
  local link="$1"
  if [[ -L "$link" ]]; then
    local target
    target="$(readlink -f "$link" 2>/dev/null || true)"
    if [[ "$target" == /data/* ]]; then
      emit PASS "$link -> $target"
    else
      emit FAIL "$link does not point into /data" "target=$target"
    fi
  else
    emit FAIL "$link is not a symlink" "expected symlink into /data"
  fi
}
check_symlink "$HOME/aviary_data"
check_symlink "$HOME/logs"

# caches must be REAL dirs (NFS-cross symlinks tank uv sync etc.)
check_real_dir() {
  local p="$1"
  if [[ -L "$p" ]]; then
    emit FAIL "$p is a symlink" "runbook warns ~/.cache must not cross NFS"
  elif [[ -d "$p" ]]; then
    emit PASS "$p is a real directory"
  else
    emit FAIL "$p missing" "create it locally, do not symlink"
  fi
}
check_real_dir "$HOME/.cache/huggingface"
check_real_dir "$HOME/.cache/uv"

# interactive sbatch helper present
[[ -f "$HOME/confs/interactive.sbatch" ]] \
  && emit PASS "~/confs/interactive.sbatch present" \
  || emit FAIL "~/confs/interactive.sbatch missing"

# tmux installed
command -v tmux >/dev/null \
  && emit PASS "tmux installed ($(tmux -V))" \
  || emit FAIL "tmux not installed" "sudo apt install tmux"

# slurm reachable from login
if command -v sinfo >/dev/null && sinfo -h >/dev/null 2>&1; then
  emit PASS "slurm reachable (sinfo ok)"
else
  emit FAIL "sinfo failed" "controller unreachable?"
fi

# 'main' partition exists (used by interactive.sbatch)
if [[ -n "$(sinfo -h -p main 2>/dev/null)" ]]; then
  emit PASS "partition 'main' exists"
else
  emit FAIL "partition 'main' not visible to this user"
fi

# login egress IP in 159.26.85.0/24
LOGIN_EGRESS="$(curl -4 -s --max-time 8 https://api.ipify.org || true)"
if [[ "$LOGIN_EGRESS" == 159.26.85.* ]]; then
  emit PASS "login egress in 159.26.85.0/24" "$LOGIN_EGRESS"
else
  emit FAIL "unexpected login egress IP" "got '$LOGIN_EGRESS', expected 159.26.85.0/24"
fi

# molr1 venv (best effort - path is conventional, not guaranteed)
if [[ -f "$HOME/code/molr1/.venv/bin/activate" ]]; then
  emit PASS "molr1 venv at ~/code/molr1/.venv"
else
  emit SKIP "molr1 venv not at ~/code/molr1/.venv" "may live elsewhere"
fi
REMOTE_EOF
)

REMOTE_RC=$?

if [[ $REMOTE_RC -ne 0 && -z "$REMOTE_OUTPUT" ]]; then
  fail "remote check session failed" "ssh exited $REMOTE_RC with no output"
else
  while IFS='|' read -r status desc detail; do
    [[ -z "$status" ]] && continue
    case "$status" in
      PASS) pass "$desc${detail:+ ($detail)}" ;;
      FAIL) fail "$desc" "$detail" ;;
      SKIP) skip "$desc" "$detail" ;;
    esac
  done <<< "$REMOTE_OUTPUT"
fi

echo
echo "=== worker checks (via srun on $HOST) ==="

if [[ "$SKIP_SRUN" == "--skip-srun" ]]; then
  skip "GPU + worker egress check" "--skip-srun passed"
else
  WORKER_OUTPUT=$(ssh -o BatchMode=yes "$HOST" bash -s <<'WORKER_EOF' 2>&1
set -u
# Reserve briefly: 1 GPU, 2 min, partition main.
srun --quiet --partition=main --time=00:02:00 --gpus=1 --cpus-per-task=1 \
  bash -c '
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    EGRESS=$(curl -4 -s --max-time 8 https://api.ipify.org || echo "")
    echo "GPU_COUNT=$GPU_COUNT"
    echo "GPU_NAME=$GPU_NAME"
    echo "EGRESS=$EGRESS"
  '
WORKER_EOF
)
  WORKER_RC=$?

  if [[ $WORKER_RC -ne 0 ]]; then
    fail "srun allocation failed" "rc=$WORKER_RC; queue full or partition denied?"
    echo "${D}${WORKER_OUTPUT}${N}"
  else
    GPU_COUNT=$(echo "$WORKER_OUTPUT" | grep -E '^GPU_COUNT=' | cut -d= -f2)
    GPU_NAME=$(echo  "$WORKER_OUTPUT" | grep -E '^GPU_NAME='  | cut -d= -f2-)
    EGRESS=$(echo    "$WORKER_OUTPUT" | grep -E '^EGRESS='    | cut -d= -f2)

    if [[ "${GPU_COUNT:-0}" -ge 1 ]] 2>/dev/null; then
      pass "worker sees $GPU_COUNT GPU(s) ($GPU_NAME)"
    else
      fail "worker sees no GPUs" "nvidia-smi returned 0 rows"
    fi

    if [[ "$GPU_NAME" == *H100* ]]; then
      pass "GPU is H100"
    else
      fail "expected H100, got '$GPU_NAME'"
    fi

    if [[ "$EGRESS" == 159.26.85.* ]]; then
      pass "worker egress in 159.26.85.0/24" "$EGRESS"
    else
      fail "unexpected worker egress" "got '$EGRESS'"
    fi
  fi
fi

echo
echo "=== summary ==="
echo "${G}${PASS} passed${N}, ${R}${FAIL} failed${N}, ${Y}${SKIP} skipped${N}"
[[ $FAIL -eq 0 ]]

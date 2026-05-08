# voltage_park_smoke.sh — assumptions

Companion notes for `voltage_park_smoke.sh`. Source of truth for the cluster
setup this script tests against:
[Voltage Park Cluster (Notion)](https://www.notion.so/Voltage-Park-Cluster-201c9764b95980bba06bf368937f8a68).

## Local (laptop) assumptions

- macOS with **bash 4+** (uses `compgen`, `[[ ]]`, `${var:+...}`).
- A host alias matching the script's first arg (e.g. `g0287`) exists in
  `~/.ssh/config`. The runbook's example block is the expected shape:
  ```
  Host g0287
    HostName 159.26.85.33
    Port 30022
    User <cluster-username>
    IdentityFile ~/.ssh/voltage-park_id_ed25519
  ```
- A private key exists in `~/.ssh` (the runbook suggests
  `voltage-park_id_ed25519`, but the script accepts any `id_*` or
  `voltage-park_id_*`).
- The public key has already been added on the cluster side by Voltage Park —
  i.e. `ssh -o BatchMode=yes <host> true` succeeds with no prompt.
- The cluster username comes from the ssh config's `User` field, not from
  `$USER` on the laptop.

## Cluster (login + worker) assumptions

Observed on `g0287` on **2026-05-07**. The Notion runbook describes an
older layout; what's actually on the cluster differs in important ways
(see "Drift from the Notion runbook" below).

- Slurm reachable from the login node; `sinfo` works; partition `main` is
  visible to the user and accepts `--gpus=1 --cpus-per-task=1` allocations.
- **Storage: `/home` and `/data` are the same place.** `/home/<you>` is an
  admin-managed symlink to `/data/users/<you>`, so `~` already lives on the
  200T `/data` NFS. The `/home` directory itself sits on the login pod's
  local overlay and contains nothing but these per-user symlinks; your
  actual files are on NFS via the redirection. Because there is no
  `/home`↔`/data` boundary to cross, the runbook's "symlink subdirs out of
  `/home` into `/data`" recipe is unnecessary, and its warning against
  symlinking `~/.cache` across NFS doesn't apply. Just `mkdir` what you
  need under `~`:
  ```bash
  mkdir -p ~/aviary_data ~/logs ~/.cache/huggingface ~/.cache/uv
  ```
  New users land under `/data/users/`; some legacy accounts (alex, conor,
  emoss, james, michael, sid, ...) live directly under `/data/`.
- Default login shell is `bash` (the cluster ships `sh` by default; the
  runbook documents the switch).
- `~/confs/interactive.sbatch` exists (the helper from the runbook).
- `tmux` is installed (`sudo apt install tmux`).
- `getent` is available (Linux NSS) — used to read the user's login shell.
- `nvidia-smi` is on `$PATH` on workers and reports H100s in CSV form.
- Outbound HTTPS to `https://api.ipify.org` works from both the login pod
  and a worker — used to observe the egress IP.
- Observed egress range is **`159.26.85.0/24`**. The runbook lists
  per-node egress IPs (e.g. `159.26.85.43` for `g0287`) but explicitly
  notes the NAT pool is not 1:1, so only the `/24` is load-bearing. If
  Voltage Park changes their NAT pool, update this range in both the
  script and this doc.
- A 2-minute `srun` allocation is long enough to run `nvidia-smi` + one
  `curl`. If the queue is busy, the worker phase blocks until allocated;
  use `--skip-srun` to bypass it.

## Best-effort (skipped, not failed)

- `~/code/molr1/.venv/bin/activate` — the runbook's `.bashrc` snippet
  activates the molr1 venv from this path, but the path itself isn't
  guaranteed. Reported as `[SKIP]` if missing.

## Drift from the Notion runbook

The runbook is the canonical source but is out of date in two places. The
script's expectations above match the cluster, not the runbook:

- **Per-user data path.** Runbook says `mkdir /data/$(whoami)`. Reality:
  `/data/users/$(whoami)` is auto-provisioned and `/home/$(whoami)` is
  already a symlink to it.
- **Symlink block + cache warning.** Runbook tells you to symlink
  `~/aviary_data`, `~/logs`, `~/.cache/huggingface`, `~/.cache/uv` into
  `/data`, and elsewhere warns that `~/.cache` must NOT be symlinked
  across NFS. Both are obsolete: `~` is already on `/data` NFS, so these
  should just be plain `mkdir`s.

## Useful commands

### From your laptop

```bash
# Full smoke test against a host alias.
./voltage_park_smoke.sh g0287

# Skip the srun/GPU phase (faster; useful when the queue is busy).
./voltage_park_smoke.sh g0287 --skip-srun

# Drop into the cluster.
ssh g0287
```

### On the login node

```bash
# First-time per-user setup. `~` is already on the /data NFS — no symlinks needed.
mkdir -p ~/aviary_data ~/logs ~/.cache/huggingface ~/.cache/uv

# Pretty squeue (alias from the runbook's .bashrc).
sq

# Submit the 12-hr 1-GPU/1-CPU reservation helper (needs ~/confs/interactive.sbatch
# and the `interactive()` function from the runbook's .bashrc).
interactive my-job-name

# One-shot interactive shell on a worker without the helper.
srun --partition=main --time=02:00:00 --gpus=1 --cpus-per-task=1 --pty bash

# Cluster state.
sinfo
sinfo -p main
squeue -u "$USER"
scancel <jobid>

# Confirm `~` is actually on /data.
readlink /home/"$USER"          # → /data/users/<you>
ls -la /data/users/"$USER"
```

### On a worker (inside an srun / interactive job)

```bash
nvidia-smi                                       # GPU inventory
curl -4 -s https://api.ipify.org; echo          # egress IP (expect 159.26.85.0/24)
```

### Admin (sudo)

```bash
# Drain a bad node, un-drain a repaired one.
sudo scontrol update nodename=worker-18 state=drain reason="bad NIC"
sudo scontrol update nodename=worker-18 state=resume
```

## Out of scope

- Validating the molr1 install itself (Python deps, CUDA-Python compat).
  See [molr1 README](https://github.com/Future-House/molr1#ubuntu).
- Multi-node or NCCL/InfiniBand checks — this is a per-user smoke test, not
  a cluster-wide health check.
- Storage capacity / NFS performance.
- Anything in the "Outstanding cluster improvements" section of the runbook
  (Ansible, Docker, Docker-for-Slurm).

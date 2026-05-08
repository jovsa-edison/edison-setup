# edison-setup

Onboarding scripts and a tiny MNIST DDP job for the Voltage Park cluster.
Two halves: laptop-side SSH/preflight, then cluster-side env + training.

## Files

### Laptop side (run first)

1. `nodes_v2.py` — adds the `vp` SSH alias to `~/.ssh/config` (round-robin ProxyCommand over the cluster's ingress IPs).
2. `voltage_park_smoke.sh` — end-to-end smoke test against an ssh host alias; verifies login, Slurm, GPUs, egress IP.
3. `voltage_park_smoke.md` — companion notes for the smoke script: assumptions, drift from the Notion runbook, useful commands.
4. `ssh_setup.txt` — scratchpad log of ssh setup attempts.

### Cluster side — one-time setup

5. `voltage_park_setup.sh` — first-time per-user install on the login pod: `uv`, `claude`, `codex`, and PATH wiring in `~/.bashrc`.

### Cluster side — training

6. `pyproject.toml` + `uv.lock` — Python deps (torch, torchvision) pinned for `uv sync --locked`.
7. `train.py` — tiny MNIST CNN; single-process or DDP via `torchrun`; rank 0 writes a per-run log to `~/logs`.
8. `train.sbatch` — Slurm wrapper for `train.py`; calls `uv sync` then `srun … torchrun … train.py`. Scales by bumping `--nodes=N`.

## Order to run

```bash
# 1. Laptop: set up ssh alias and verify the cluster is reachable.
python3 nodes_v2.py --user <cluster-username>
./voltage_park_smoke.sh vp

# 2. Cluster: one-time per-user install.
ssh vp
bash voltage_park_setup.sh
source ~/.bashrc

# 3. Cluster: materialize the venv and train.
cd edison-setup
uv sync --locked
sbatch --gpus-per-node=8 --cpus-per-task=8 train.sbatch                # 1 node x 8 GPUs
sbatch --nodes=2 --gpus-per-node=8 --cpus-per-task=8 train.sbatch      # 2 nodes x 8 GPUs
```

Logs land in `/home/<you>/logs/mnist-ddp-<jobid>.out`.

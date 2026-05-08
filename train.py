"""Tiny MNIST CNN.

Same script supports four launch modes:

  python train.py                                  # CPU or single GPU
  srun --gpus=1 --pty python train.py              # 1 GPU under Slurm
  srun --gpus=8 --ntasks=1 --pty \\
      torchrun --standalone --nproc-per-node=8 train.py
  sbatch [--nodes=N --gpus-per-node=8] train.sbatch

The script reads RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
from the environment (set by torchrun) and inits NCCL+DDP when present;
otherwise it runs single-process.

Rank 0 also writes a per-run log to --log-dir (default ~/logs).
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return rank, world_size, local_rank, device, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device, False


class Logger:
    """Write to stdout and (optionally) to a file. Stdout always flushes."""

    def __init__(self, file=None):
        self.file = file

    def __call__(self, msg):
        print(msg, flush=True)
        if self.file is not None:
            self.file.write(msg + "\n")
            self.file.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default=os.path.expanduser("~/aviary_data/mnist"))
    p.add_argument("--log-dir", default=os.path.expanduser("~/logs"))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rank, world_size, local_rank, device, distributed = setup_distributed()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    log_file = None
    log_path = None
    if is_main:
        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        run_id = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(args.log_dir, f"mnist-{run_id}.log")
        log_file = open(log_path, "a")
    log = Logger(log_file)

    if is_main:
        log(f"[rank {rank}/{world_size}] device={device} distributed={distributed} "
            f"local_rank={local_rank} log_path={log_path}")

    # Rank 0 downloads first, others wait, then everyone reads from NFS.
    if distributed and rank != 0:
        dist.barrier()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if distributed and rank == 0:
        dist.barrier()

    train_sampler = DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(test_set, batch_size=512, num_workers=args.num_workers)

    model = CNN().to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        total, total_loss = 0, 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += xb.size(0)
            total_loss += loss.item() * xb.size(0)
        if is_main:
            log(f"[rank 0] epoch {epoch + 1}/{args.epochs} "
                f"train_loss={total_loss / max(total, 1):.4f} "
                f"time={time.time() - t0:.1f}s")

    if is_main:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = (model.module if distributed else model)(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        log(f"[rank 0] test_acc={correct / total:.4f}")
        if log_file is not None:
            log_file.close()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

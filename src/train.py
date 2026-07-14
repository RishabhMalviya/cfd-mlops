#!/usr/bin/env python
"""
Transolver training on DrivAerML — DDP-ready, fault-tolerant.

Single GPU:
    python cfd-mlops/src/train.py --data_dir drivaer_data/ --cache_dir drivaer_data/cache/

Multi-GPU (e.g. 2):
    torchrun --nproc_per_node=2 cfd-mlops/src/train.py --data_dir drivaer_data/ ...

Resume from checkpoint:
    python cfd-mlops/src/train.py ... --resume checkpoints/epoch_0020.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Transolver" / "Car-Design-ShapeNetCar"))
sys.path.insert(0, str(ROOT / "cfd-mlops" / "src"))

from models.Transolver import Model  # noqa: E402
from drivaer_dataset import DrivAerDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="drivaer_data/")
    p.add_argument("--cache_dir",  default="drivaer_data/cache/")
    p.add_argument("--ckpt_dir",   default="checkpoints/")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--reg",        type=float, default=0.5,  help="pressure loss weight")
    p.add_argument("--val_every",  type=int,   default=10)
    p.add_argument("--ckpt_every", type=int,   default=20)
    p.add_argument("--resume",     default="",  help="path to checkpoint .pt to resume from")
    return p.parse_args()


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world_size = 0, 1
    return rank, world_size


def cleanup_ddp(world_size):
    if world_size > 1:
        dist.destroy_process_group()


def build_datasets(args, rank):
    all_ids = list(range(1, 51))
    avail = [
        i for i in all_ids
        if os.path.exists(os.path.join(args.data_dir, f"run_{i}", f"boundary_{i}.vtp"))
    ]
    if not avail:
        raise RuntimeError(f"No DrivAerML runs found under {args.data_dir}")

    n_train   = max(1, int(len(avail) * 0.8))
    train_ids = avail[:n_train]
    # Guarantee at least one val sample even when dataset is tiny
    val_ids   = avail[n_train:] if len(avail) > n_train else avail[-1:]

    if rank == 0:
        print(f"[data] {len(avail)} runs  |  train={len(train_ids)}  val={len(val_ids)}")

    train_ds = DrivAerDataset(args.data_dir, run_ids=train_ids, cache_dir=args.cache_dir)
    val_ds   = DrivAerDataset(args.data_dir, run_ids=val_ids,   cache_dir=args.cache_dir,
                               norm_coef=train_ds.norm_coef)
    return train_ds, val_ds


def build_loaders(train_ds, val_ds, args, world_size):
    sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=True)
    return train_loader, val_loader, sampler


def save_ckpt(path, epoch, model, optimizer, scheduler, best_val):
    state = {
        "epoch":     epoch,
        "model":     (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val":  best_val,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic write avoids corrupted checkpoints on crash


def load_ckpt(path, model, optimizer, scheduler, device):
    state     = torch.load(path, map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"], state.get("best_val", float("inf"))


def train_one_epoch(device, model, loader, optimizer, scheduler, sampler, epoch, reg):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    crit   = nn.MSELoss(reduction="none")
    losses = []
    for cfd_data, geom in loader:
        cfd_data = cfd_data.to(device)
        geom     = geom.to(device)
        optimizer.zero_grad(set_to_none=True)

        out     = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = crit(out[cfd_data.surf, -1],  targets[cfd_data.surf, -1]).mean()
        loss_velo  = crit(out[:, :-1],             targets[:, :-1]).mean()
        loss       = loss_velo + reg * loss_press

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(device, model, loader, reg):
    model.eval()
    crit   = nn.MSELoss(reduction="none")
    losses = []
    for cfd_data, geom in loader:
        cfd_data = cfd_data.to(device)
        geom     = geom.to(device)
        out      = model((cfd_data, geom))
        targets  = cfd_data.y

        loss_press = crit(out[cfd_data.surf, -1],  targets[cfd_data.surf, -1]).mean()
        loss_velo  = crit(out[:, :-1],             targets[:, :-1]).mean()
        losses.append((loss_velo + reg * loss_press).item())

    return float(np.mean(losses))


def main():
    args             = parse_args()
    rank, world_size = setup_ddp()
    device           = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_ds, val_ds                 = build_datasets(args, rank)
    train_loader, val_loader, sampler = build_loaders(train_ds, val_ds, args, world_size)

    model = Model(
        space_dim=7, fun_dim=0, out_dim=4,
        n_hidden=256, n_layers=8, n_head=8,
        mlp_ratio=2, slice_num=32, unified_pos=0,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # total_steps computed after loader is built so drop_last accounting is exact
    total_steps = len(train_loader) * args.epochs
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(total_steps, 1),
        final_div_factor=1000.0,
    )

    start_epoch = 0
    best_val    = float("inf")
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_val = load_ckpt(args.resume, model, optimizer, scheduler, device)
        if rank == 0:
            print(f"[ckpt] resumed from {args.resume}  (start epoch {start_epoch + 1})")
        start_epoch += 1

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] {n_params:,} trainable parameters")
        print(f"[train] {args.epochs} epochs  |  {len(train_loader)} steps/epoch  |  device={device}")

    for epoch in range(start_epoch, args.epochs):
        t0         = time.time()
        train_loss = train_one_epoch(
            device, model, train_loader, optimizer, scheduler, sampler, epoch, args.reg
        )

        val_loss = float("nan")
        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss = evaluate(device, model, val_loader, args.reg)
            if rank == 0 and val_loss < best_val:
                best_val = val_loss
                save_ckpt(
                    os.path.join(args.ckpt_dir, "best.pt"),
                    epoch, model, optimizer, scheduler, best_val,
                )

        if rank == 0:
            print(
                f"epoch {epoch + 1:>4d}/{args.epochs}"
                f"  train={train_loss:.4f}"
                f"  val={val_loss:.4f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
                f"  t={time.time() - t0:.1f}s"
            )
            if (epoch + 1) % args.ckpt_every == 0:
                save_ckpt(
                    os.path.join(args.ckpt_dir, f"epoch_{epoch + 1:04d}.pt"),
                    epoch, model, optimizer, scheduler, best_val,
                )

    if rank == 0:
        save_ckpt(
            os.path.join(args.ckpt_dir, f"final_epoch_{args.epochs}.pt"),
            args.epochs - 1, model, optimizer, scheduler, best_val,
        )
        print(f"[done] best val loss: {best_val:.4f}")

    cleanup_ddp(world_size)


if __name__ == "__main__":
    main()

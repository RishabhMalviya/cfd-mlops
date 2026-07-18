# CLAUDE.md — Scaling strategies

This repo trains **Transolver** to predict surface CFD fields on **DrivAerML** car meshes.
The raw surface mesh is enormous — `boundary_*.vtp` has **~8.8M cells** (`run_1`: 8,828,095) —
and Transolver's memory cost is **linear in the number of nodes N**, dominated by *activation*
memory, not parameters. The model itself is ~2M params (a few MB); Adam states are negligible.
Everything below is about fitting/scaling those activations.

> Sizing anchor: one `(N, hidden=256)` fp32 activation at N=8.8M is **8.42 GiB**.
> A single `PhysicsAttention.forward` holds ~3 of these live at peak.
> → forward-only peak ≈ **25 GiB**; a training step (activations retained for backward,
> ~5–7 such tensors/block × 5 blocks) ≈ **250 GiB**. On a 12 GB card this OOMs immediately.

There are two distinct ways to scale, along two different axes. **Do not conflate them.**
DDP scales *throughput over samples*; it does **not** make a single sample fit. Spatial
parallelism scales *a single sample across GPUs*; it does **not** by itself increase throughput.

---

## Approach 1 — Decimation + DDP  (the headline / practical path)

Make each sample small enough to fit on one GPU, then use **data parallelism** to go fast
across many car runs. This is the portfolio story: **DDP + torchrun + fault-tolerant
checkpoint/resume**. DDP (not FSDP) is deliberate — the bottleneck is activation memory, not
parameter memory, so sharding parameters would be the wrong tool.

**Shrink the sample — decimate the mesh.**
`triangulate().decimate(0.99)` takes 8.8M → ~88k nodes, dropping activation cost ~100×
(8.42 GiB → ~90 MiB per `(N,256)` tensor). Decimation builds a *new* mesh whose vertices no
longer carry the CFD fields, so labels are recovered with a **KDTree remap**:

```
pv.read()                          # 8.8M cells, HAS pMeanTrim + wallShearStressMeanTrim
  → cell_data_to_point_data()      # move fields onto points
  → triangulate().decimate(0.99)   # ~88k points, geometry only — labels lost
  → compute_normals()              # normals for the survivors  → x = [pos, normals]
  → KDTree(orig_points).query(deci_points)   # nearest-original index per survivor
  → y = [wss[idx], pressure[idx]]  # gather labels back onto the 88k points
```

Nearest-neighbor is valid because surface fields are spatially smooth: a decimated vertex sits
sub-millimeter from an original one. Keep cell-vs-point space consistent on both sides of the
tree (hence `cell_data_to_point_data()` first). A fixed-budget random subsample (e.g. 32k–65k
nodes) can cap memory further and doubles as augmentation across epochs.

**Scale out — DDP.** Each rank gets a *full replica* of the model and a *different* decimated
sample (batch sharded across ranks). More GPUs = more samples/sec. Checkpoints bundle
`{epoch, model (unwrapped from DDP), optimizer, scheduler, best_val}`, written atomically
(`torch.save` to `.tmp` then `os.replace`); `--resume` restores all and continues from `epoch+1`.

**Memory:** trivial once decimated — a full step fits in <2 GiB even on one 12 GB card.

**Limitation:** DDP replicates the sample. It can **never** let you train on the *undecimated*
8.8M mesh — each rank would still need the full ~250 GiB. That is Approach 2's job.

---

## Approach 2 — Model/spatial parallelism + gradient checkpointing  (full-mesh path)

Train on the **full 8.8M-node mesh** by sharding the **N (node) dimension** across GPUs, so
each GPU holds only N/G nodes' worth of activations. This is *tensor/sequence parallelism*, a
different axis from DDP.

**Why Transolver shards cleanly.** The per-node work is nearly embarrassingly parallel; the only
cross-GPU coupling is tiny:

- **Slice** — `softmax` over slices, per node → **local**, no comms.
- **Slice-token formation** `einsum("bhnc,bhng->bhgc")` — a **reduction over N**. Each GPU sums
  over its shard, then **all-reduce** `slice_token` + `slice_norm`, both shape `(B,H,G,C)=(1,8,32,32)`
  → a few KB. Negligible traffic.
- **Attention over the G=32 slice tokens** — tiny; just replicate on every GPU.
- **Deslice** `einsum("bhgc,bhng->bhnc")` — per-node weighted combine → **local** again.

So the entire cross-GPU cost is **two KB-sized all-reduces per block**. This is what makes the
full-mesh route attractive rather than painful.

**Gradient checkpointing** on each `TransolverBlock` (`torch.utils.checkpoint`): store only block
*inputs* and recompute activations during backward — trades ~1.3× compute for a large activation
saving. Essential here; without it even the sharded step overflows.

**Napkin memory on 8×16 GB** (per-GPU `(N/8,256)` tensor = 1.05 GiB):

| configuration                              | per-GPU activations | fits 16 GiB? |
|--------------------------------------------|---------------------|--------------|
| inference (forward only)                   | ~3 GiB              | ✅ trivially |
| training, naive                            | ~30–35 GiB          | ❌ over      |
| training + gradient checkpointing          | ~9–12 GiB           | ✅           |
| training + grad-checkpoint + bf16          | ~5–6 GiB            | ✅ comfortable |

bf16 halves activation memory again; the RTX Blackwell / most 16 GB datacenter cards support it.

**Limitation:** more implementation complexity (custom sharded `PhysicsAttention` with partial
sums + all-reduce, careful autograd through the collectives) and it parallelizes *one* sample —
combine with DDP (2D parallelism) if you also want throughput.

---

## Which to use

- **Default / demo:** Approach 1. It fits a single 12 GB card, exercises the DDP + checkpoint/
  resume skills this project is meant to showcase, and is far simpler.
- **Full-fidelity / "cool" track:** Approach 2, when you specifically want to train on the
  undecimated mesh and have a multi-GPU box. It is feasible precisely because Transolver's
  slice-token reduction keeps cross-GPU comms to KB per block.

## Reference facts

- `pyproject.toml`: `uv` project, Python 3.12, torch pinned to the cu128 wheel index.
- Dev box: WSL2, single RTX 5070 Ti (12 GB, Blackwell sm_120).
- The OOM message's `"...17179869184.00 GiB memory in use"` is a **units bug**:
  17,179,869,184 = 2^34 bytes = **16.00 GiB** (raw byte count mislabeled `GiB`). Trust the
  `"17.05 GiB allocated"` / `"11.94 GiB capacity"` figures instead.

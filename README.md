# CFD-MLOps — Distributed Training of a Surrogate CFD Model for Vehicle Aerodynamics

Predicting surface **pressure** and **wall shear stress** fields over car meshes with a neural PDE surrogate — trained in a distributed manner with **PyTorch DDP + `torchrun`**.

<p align="center">
  <img src="readme_assets/Prediction Outputs.png" alt="Inputs and outputs for this project" width="80%">
</p>

---

## The problem

Automotive aerodynamics is traditionally evaluated with Computational Fluid Dynamics (CFD) simulation. While these are highly accurate, they can take hours to days of solver time per design. The goal here is to train a surrogate neural network model to directly predict the quantities a CFD solver would produce given a car's surface mesh, in a single forward pass. 

Neural network are do a great of job of generalizing to inputs that are nearby in continuous space. Since the variability in car geometries is not too great, I expect neural surrogate models to generarlize to unseen car geometries quite well. 

Concretely, this project trains a model to predict, at every node of a vehicle's surface mesh:

- **Pressure** (`p`) — the dominant contributor to aerodynamic drag and lift
- **Wall shear stress** (3 components) — the surface friction that makes up the rest of the drag budget

## The model — Transolver

Training uses [Transolver](https://github.com/thuml/Transolver), a Transformer-based PDE solver that learns physical states over irregular meshes via a "physics-attention" mechanism.

The [`Car-Design-ShapeNetCar` variant](https://github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar) is adapted here to the surface-only DrivAerML data without modifying the model itself. Each mesh node is fed 7 geometric input channels `[position(3), sdf(1), normals(3)]` and the model predicts 4 output channels `[wall shear stress(3), pressure(1)]`. This is the same sub-problem tackled by the paper [GA-Field: Geometry Aware Vehicle Aerodynamics Field Prediction](https://arxiv.org/pdf/2602.20609).

Since Transolver's memory bottleneck is activation memory from large mesh inputs, I use DDP to parallelize the training across multiple GPUs.

## The data — DrivAerML

[DrivAerML](https://caemldatasets.org/drivaerml/) is an open, high-fidelity CFD dataset of 500 parametrically-morphed DrivAer road-car geometries, generated with OpenFOAM using industrial-standard, validated workflows:

<p align="center">
  <img src="https://caemldatasets.org/assets/img/drivaer1.png" alt="DrivAerML — high-fidelity CFD over a morphed DrivAer car geometry" width="80%">
</p>

It provides surface (boundary) and volume flow-field data — pressure, wall shear stress, velocity, forces and moments — released under CC BY-SA 4.0. This project uses the surface (`boundary_*.vtp`) data

## Tech stack

| Concern | Tooling |
|---|---|
| Model | Transolver (physics-attention Transformer) |
| Training | PyTorch 2.11 (CUDA 12.8), DDP via `torchrun` |
| Data | DrivAerML surface meshes · PyVista · PyTorch Geometric |
| Orchestration | Kubeflow Pipelines |
| Environment | `uv` · Python 3.12 |

## Quickstart

Environment is managed with [`uv`](https://docs.astral.sh/uv/). Run from the working root.

```bash
# Dataset smoke test — builds/caches a few runs and asserts tensor shapes
python src/dataset/drivaer_dataset.py data/drivaer_data <cache_dir>

# Single-GPU training
python src/train.py --data_dir data/drivaer_data --cache_dir data/drivaer_data/cache

# Multi-GPU training (DDP)
torchrun --nproc_per_node=2 src/train.py --data_dir data/drivaer_data ...

# Resume from a checkpoint
python src/train.py ... --resume checkpoints/epoch_0020.pt
```

The DrivAerML data can be fetched with `data/download_drivaer_data.sh`.

## Project layout

```
cfd-mlops/
├── src/
│   ├── train.py                    # DDP training entrypoint (torchrun-aware)
│   └── dataset/drivaer_dataset.py  # mesh preprocessing, caching, normalization
├── data/drivaer_data/              # DrivAerML runs (gitignored)
└── pyproject.toml                  # uv project, Python 3.12, torch cu128
```

## Acknowledgements

- **Transolver** — Wu et al., [thuml/Transolver](https://github.com/thuml/Transolver)
- **DrivAerML** — [caemldatasets.org/drivaerml](https://caemldatasets.org/drivaerml/) (CC BY-SA 4.0)

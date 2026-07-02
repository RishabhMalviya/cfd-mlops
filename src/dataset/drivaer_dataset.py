import os
from typing import cast

import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class DrivAerDataset(Dataset):
    def __init__(self, data_dir, run_ids, target_reduction=0.99, target_geom_tensor_size=8192, cache_dir=None, norm_coef=None, norm_cache=None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, '/cache/')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.target_reduction = target_reduction
        self.target_geom_tensor_size = target_geom_tensor_size

        self.run_ids = [
            i for i in run_ids
            if os.path.exists(os.path.join(data_dir, f"run_{i}", f"boundary_{i}.vtp"))
        ]

        for idx in range(len(self.run_ids)):
            print(f'Processing and caching run {idx+1}/{len(self.run_ids)}...', end='\r')
            _, _ = self._process_and_cache(idx)

        self.norm_cache = norm_cache if norm_cache else os.path.join(self.cache_dir, "norm_coef.pt")
        if os.path.exists(self.norm_cache):
            self.norm_coef = torch.load(self.norm_cache, weights_only=False)
        else:
            self.norm_coef = norm_coef if norm_coef else self._compute_norm_coef()
            torch.save(self.norm_coef, self.norm_cache)

    def __len__(self):
        return len(self.run_ids)

    def __getitem__(self, idx):
        data, geom = self._process_and_cache(idx)
        data, geom = self._apply_norm(data, geom)

        return data, geom

    def _process_and_cache(self, idx):
        run_id = self.run_ids[idx]

        cache_path = os.path.join(self.cache_dir, f"run_{run_id}.pt")

        cache_exists = os.path.exists(cache_path)
        if cache_exists:
            data, geom = torch.load(cache_path, weights_only=False)
        else:
            data, geom = self._process(run_id)
            torch.save((data, geom), cache_path)

        return data, geom

    def _process(self, run_id):
        vtp_path = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        mesh_full = pv.read(vtp_path).cell_data_to_point_data()

        # Decimate for geometry-aware point reduction, then compute normals on reduced mesh
        mesh_dec = (
            mesh_full
            .triangulate()
            .decimate(self.target_reduction)
            .compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
        )

        pos     = mesh_dec.points.astype(np.float32)                          # (N, 3)
        normals = mesh_dec.point_data["Normals"].astype(np.float32)           # (N, 3)
        N = pos.shape[0]

        # Map decimated points back to original mesh via nearest-neighbour lookup.
        # vtkDecimatePro only removes vertices, so each decimated point is an exact
        # original vertex — distances should be ~0.
        _, nn_idx = KDTree(mesh_full.points).query(pos, workers=-1)          # (N,)
        pressure = mesh_full.point_data["pMeanTrim"][nn_idx].astype(np.float32)             # (N,)
        wss      = mesh_full.point_data["wallShearStressMeanTrim"][nn_idx].astype(np.float32)  # (N, 3)

        sdf = np.zeros((N, 1), dtype=np.float32)
        x = np.concatenate([pos, sdf, normals], axis=1)           # (N, 7)
        y = np.concatenate([wss, pressure[:, None]], axis=1)       # (N, 4)

        # Vectorised undirected edge construction from triangle faces
        faces    = mesh_dec.faces.reshape(-1, 4)[:, 1:]            # (N_tri, 3)
        pair_idx = np.array([[0, 1], [1, 2], [0, 2]])              # (3, 2)
        src = faces[:, pair_idx[:, 0]].reshape(-1)                 # (N_tri*3,)
        dst = faces[:, pair_idx[:, 1]].reshape(-1)                 # (N_tri*3,)
        edge_pairs = np.unique(
            np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0),
            axis=1,
        )                                                           # (2, E)

        # Geometry point cloud for Transolver's shape encoder
        n_sub    = min(self.target_geom_tensor_size, N)
        sub_idx  = np.random.choice(N, n_sub, replace=False)
        geom_pts = pos[sub_idx].copy()
        geom_pts -= geom_pts.mean(axis=0)
        max_r = np.linalg.norm(geom_pts, axis=1).max()
        if max_r > 0:
            geom_pts /= max_r

        data = Data(
            x=torch.from_numpy(x),
            y=torch.from_numpy(y),
            pos=torch.from_numpy(pos),
            edge_index=torch.from_numpy(edge_pairs).to(torch.int64),
            surf=torch.ones(N, dtype=torch.bool),
        )
        geom = torch.from_numpy(geom_pts)                          # (<=8192, 3)

        return data, geom

    def _apply_norm(self, data, geom):
        mean_x, std_x, mean_y, std_y = self.norm_coef
        data = data.clone()
        data.x = (data.x - mean_x) / std_x
        data.y = (data.y - mean_y) / std_y
        return data, geom

    def _compute_norm_coef(self):
        # Calculate Means
        sum_x = torch.zeros(7)
        sum_y = torch.zeros(4)
        count = 0
        for idx in range(len(self.run_ids)):
            data, _ = self._process_and_cache(idx)
            sum_x += data.x.sum(dim=0)
            sum_y += cast(torch.Tensor, data.y).sum(dim=0)
            count += data.x.shape[0]
        mean_x = sum_x / count
        mean_y = sum_y / count

        # Calculate Variances
        sum_sq_x = torch.zeros(7)
        sum_sq_y = torch.zeros(4)
        for idx in range(len(self.run_ids)):
            data, _ = self._process_and_cache(idx)
            sum_sq_x += ((data.x - mean_x) ** 2).sum(dim=0)
            sum_sq_y += ((data.y - mean_y) ** 2).sum(dim=0)
        std_x = (sum_sq_x / count).clamp(min=1e-8).sqrt()
        std_y = (sum_sq_y / count).clamp(min=1e-8).sqrt()

        return mean_x, std_x, mean_y, std_y

    @classmethod
    def compute_norm_coef(cls, dataset):
        return dataset._compute_norm_coef()


if __name__ == "__main__":
    import sys

    data_dir  = sys.argv[1] if len(sys.argv) > 1 else "./data/drivaer_data"
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_dir, "cache_smoke_test")

    all_ids = list(range(1, 51))
    avail   = [i for i in all_ids
               if os.path.exists(os.path.join(data_dir, f"run_{i}", f"boundary_{i}.vtp"))]
    print(f"Available runs : {len(avail)}/50  {avail}")

    # Use only what's downloaded for smoke test
    n_train = max(1, int(len(avail) * 0.8))
    train_ids, val_ids = avail[:n_train], avail[n_train:]

    print(f"\n--- Building train dataset ({len(train_ids)} samples) ---")
    train_ds = DrivAerDataset(data_dir, run_ids=train_ids, cache_dir=cache_dir)

    if val_ids:
        print(f"\n--- Building val dataset ({len(val_ids)} samples, shared norm) ---")
        val_ds = DrivAerDataset(data_dir, run_ids=val_ids,
                                cache_dir=cache_dir, norm_coef=train_ds.norm_coef)
    else:
        print("(skipping val dataset — not enough samples)")

    # --- Shape checks ---
    print("\n--- Shape checks (train[0]) ---")
    data, geom = train_ds[0]
    N = data.x.shape[0]

    assert data.x.shape          == (N, 7),          f"x shape wrong: {data.x.shape}"
    assert data.y.shape          == (N, 4),          f"y shape wrong: {data.y.shape}"
    assert data.pos.shape        == (N, 3),          f"pos shape wrong: {data.pos.shape}"
    assert data.edge_index.shape[0] == 2,            f"edge_index row dim wrong"
    assert data.surf.shape       == (N,),            f"surf shape wrong: {data.surf.shape}"
    assert data.surf.all(),                          "surf should be all True"
    assert geom.shape[1]         == 3,               f"geom col dim wrong: {geom.shape}"
    assert geom.shape[0]         <= 8192,            f"geom too large: {geom.shape}"
    assert data.x.dtype          == torch.float32,   f"x dtype wrong: {data.x.dtype}"
    assert data.y.dtype          == torch.float32,   f"y dtype wrong: {data.y.dtype}"
    assert data.edge_index.dtype == torch.int64,     f"edge_index dtype wrong"

    print(f"  data.x        : {tuple(data.x.shape)}")
    print(f"  data.y        : {tuple(data.y.shape)}")
    print(f"  data.pos      : {tuple(data.pos.shape)}")
    print(f"  edge_index    : {tuple(data.edge_index.shape)}")
    print(f"  surf          : all={data.surf.all().item()}")
    print(f"  geom          : {tuple(geom.shape)}")

    # --- Normalisation checks ---
    print("\n--- Normalisation checks ---")
    # After normalisation, x and y should be roughly zero-mean unit-variance
    # (not exact per-sample, but close across the dataset)
    x_mean = data.x.mean(dim=0)
    y_mean = data.y.mean(dim=0)
    print(f"  x per-channel mean (should be ~0): {x_mean.tolist()}")
    print(f"  y per-channel mean (should be ~0): {y_mean.tolist()}")

    # --- Edge index validity ---
    print("\n--- Edge index checks ---")
    src, dst = data.edge_index
    assert src.min() >= 0 and src.max() < N, "src indices out of range"
    assert dst.min() >= 0 and dst.max() < N, "dst indices out of range"
    # Undirected: every (u,v) should have a corresponding (v,u)
    edges_set  = set(zip(src.tolist(), dst.tolist()))
    n_missing  = sum(1 for u, v in edges_set if (v, u) not in edges_set)
    assert n_missing == 0, f"{n_missing} edges missing reverse direction"
    print(f"  {data.edge_index.shape[1]} edges, all bidirectional: OK")

    # --- NN mapping sanity: y values should be finite and in original data range ---
    print("\n--- y value checks ---")
    assert torch.isfinite(data.y).all(), "y contains non-finite values"
    print(f"  y (normalised) min={data.y.min():.3f}  max={data.y.max():.3f}  — finite: OK")

    # --- Cache round-trip ---
    print("\n--- Cache round-trip check ---")
    data2, geom2 = train_ds[0]
    assert torch.allclose(data.x, data2.x),     "cached x differs"
    assert torch.allclose(data.y, data2.y),     "cached y differs"
    assert torch.allclose(geom,  geom2),        "cached geom differs"
    print("  Cache round-trip: OK")

    # --- Norm coef not recomputed for val ---
    if val_ids:
        print("\n--- Val norm coef matches train ---")
        for a, b in zip(train_ds.norm_coef, val_ds.norm_coef):
            assert torch.allclose(a, b), "norm_coef mismatch between train and val"
        print("  OK")

    print("\nAll smoke tests passed.")

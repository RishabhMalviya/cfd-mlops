import argparse
import os
import time
from typing import cast

import numpy as np
from tqdm import tqdm
import pyvista as pv
from pyvista.core.pointset import PolyData

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


class DrivAerDataset(Dataset):
    def __init__(self, data_dir, run_ids=None, decimate=False, target_reduction=0.99, cache_dir=None, norm_coef=None, norm_cache=None):
        def _get_run_ids_from_data_dir(data_dir):
            inferred = []
            # look for directories starting with 'run_'
            for entry in os.listdir(data_dir):
                entry_path = os.path.join(data_dir, entry)
                if not os.path.isdir(entry_path):
                    continue
                if entry.startswith('run_'):
                    suffix = entry.split('run_', 1)[1]
                    # check for boundary file
                    if os.path.exists(os.path.join(data_dir, entry, f"boundary_{suffix}.vtp")):
                        try:
                            inferred.append(int(suffix))
                        except ValueError:
                            inferred.append(suffix)
            inferred.sort()
            return inferred

        self.data_dir = data_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, 'cache/')
        os.makedirs(self.cache_dir, exist_ok=True)
        if run_ids is None:
            self.run_ids = _get_run_ids_from_data_dir(data_dir)
        else:
            self.run_ids = [
                i for i in run_ids
                if os.path.exists(os.path.join(data_dir, f"run_{i}", f"boundary_{i}.vtp"))
            ]

        self.decimate = decimate
        self.decimate_reduction = target_reduction
        for idx in tqdm(range(len(self.run_ids)), desc="Processing and caching runs", unit="run"):
            _ = self._process_and_cache(idx)

        self.norm_cache = norm_cache if norm_cache else os.path.join(self.cache_dir, "norm_coef.pt")
        if os.path.exists(self.norm_cache):
            self.norm_coef = torch.load(self.norm_cache, weights_only=False)
        else:
            self.norm_coef = norm_coef if norm_coef else self._compute_norm_coef()
            torch.save(self.norm_coef, self.norm_cache)

    def __len__(self):
        return len(self.run_ids)

    def __getitem__(self, idx):
        def _apply_norm(data):
            mean_x, std_x, mean_y, std_y = self.norm_coef

            data = data.clone()

            data.x = (data.x - mean_x) / std_x
            data.y = (data.y - mean_y) / std_y

            return data
        
        data = self._process_and_cache(idx)
        data = _apply_norm(data)

        return data

    def _process_and_cache(self, idx):
        run_id = self.run_ids[idx]

        cache_path = os.path.join(self.cache_dir, f"run_{run_id}.pt")

        cache_exists = os.path.exists(cache_path)
        if cache_exists:
            data = torch.load(cache_path, weights_only=False)
        else:
            data = self._process(run_id)
            torch.save(data, cache_path)

        return data

    def _process(self, run_id):
        vtp_path = os.path.join(self.data_dir, f"run_{run_id}", f"boundary_{run_id}.vtp")
        mesh: PolyData = pv.read(vtp_path) # pyright: ignore[reportAssignmentType]

        # --- Build `x` ---
        if self.decimate:
            original_mesh_kd_tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(mesh.cell_centers().points)

            triangulated_mesh = mesh.triangulate()
            decimated_mesh = triangulated_mesh.decimate_pro(reduction=self.decimate_reduction, preserve_topology=True)

            pos = decimated_mesh.cell_centers().points.astype(np.float32)     # (N, 3)
            normals = decimated_mesh.cell_normals.astype(np.float32)          # (N, 3)
        else:
            pos     = mesh.cell_centers().points.astype(np.float32)     # (N, 3)
            normals = mesh.cell_normals.astype(np.float32)              # (N, 3)

        x = np.concatenate([pos, normals], axis=1)                  # (N, 6)

        # --- Build `y` ---
        if self.decimate:
            # Find closest original mesh cell for each decimated mesh cell
            dists, original_indices = original_mesh_kd_tree.kneighbors(pos)
            assert max(dists.flatten()) < 5e-2, f"Decimated mesh cell is too far from original mesh cell: {max(dists.flatten())} (run_id={run_id})"

            # Get cell values from original mesh using the indices of the closest cells
            pressure = mesh.cell_data['pMeanTrim'][original_indices.flatten()].astype(np.float32)                  # (N,)
            wss      = mesh.cell_data['wallShearStressMeanTrim'][original_indices.flatten()].astype(np.float32)    # (N, 3)
        else:
            pressure = mesh["pMeanTrim"].astype(np.float32)                  # (N,)
            wss      = mesh["wallShearStressMeanTrim"].astype(np.float32)    # (N, 3)

        y = np.concatenate([wss, pressure[:, np.newaxis]], axis=1)                  # (N, 4) = [wx, wy, wz, p]

        # --- Encapsulate in PyG Data object ---
        data = Data(
            x=torch.from_numpy(x),
            y=torch.from_numpy(y),
            pos=torch.from_numpy(pos)
        )

        return data

    def _compute_norm_coef(self):
        # Calculate Means
        sum_x = torch.zeros(6)
        sum_y = torch.zeros(4)
        count = 0
        for idx in range(len(self.run_ids)):
            data = self._process_and_cache(idx)
            sum_x += data.x.sum(dim=0)
            sum_y += cast(torch.Tensor, data.y).sum(dim=0)
            count += data.x.shape[0]
        mean_x = sum_x / count
        mean_y = sum_y / count

        # Calculate Variances
        sum_sq_x = torch.zeros(6)
        sum_sq_y = torch.zeros(4)
        for idx in range(len(self.run_ids)):
            data = self._process_and_cache(idx)
            sum_sq_x += ((data.x - mean_x) ** 2).sum(dim=0)
            sum_sq_y += ((data.y - mean_y) ** 2).sum(dim=0)
        std_x = (sum_sq_x / count).clamp(min=1e-8).sqrt()
        std_y = (sum_sq_y / count).clamp(min=1e-8).sqrt()

        return mean_x, std_x, mean_y, std_y

    @classmethod
    def compute_norm_coef(cls, dataset):
        return dataset._compute_norm_coef()


def _check_shapes(ds):
    print("\n--- Shape Checks ---")
    data: Data = ds[0]
    N = data.x.shape[0]

    assert data.x is not None, "data.x is None"
    x: torch.Tensor = data.x
    assert data.x.shape          == (N, 6),          f"x shape wrong: {data.x.shape}"
    assert data.x.dtype          == torch.float32,   f"x dtype wrong: {data.x.dtype}"
    print(f"data.x           : {tuple(data.x.shape)}")
    assert torch.isfinite(data.x).all(), "x contains non-finite values"
    print(f"x (Normalized)   : min={data.x.min():.3f}  max={data.x.max():.3f}")

    assert type(data.y) is torch.Tensor, "data.y is not a torch.Tensor"
    y: torch.Tensor = data.y
    assert data.y.shape          == (N, 4),          f"y shape wrong: {data.y.shape}"
    assert data.y.dtype          == torch.float32,   f"y dtype wrong: {data.y.dtype}"
    print(f"data.y           : {tuple(data.y.shape)}")
    assert torch.isfinite(data.y).all(), "y contains non-finite values"
    print(f"y (Normalized)   : min={data.y.min():.3f}  max={data.y.max():.3f}")

    assert data.pos is not None, "data.pos is None"
    pos: torch.Tensor = data.pos
    assert data.pos.shape        == (N, 3),          f"pos shape wrong: {data.pos.shape}"
    print(f"data.pos         : {tuple(data.pos.shape)}")

    print('Shape Checks: OK')

    return data


def _check_normalization(ds):
    # Checks *dataset-wide* mean ~= 0 and variance ~= 1
    print("\n--- Normalization Checks (over full dataset) ---")

    sum_x   = torch.zeros(6, dtype=torch.float64)
    sumsq_x = torch.zeros(6, dtype=torch.float64)
    sum_y   = torch.zeros(4, dtype=torch.float64)
    sumsq_y = torch.zeros(4, dtype=torch.float64)
    n_pts   = 0
    for i in tqdm(range(len(ds)), desc="Normalization check (over entire dataset)", unit="run"):
        d  = ds[i]                       # normalised sample
        xx = d.x.double()
        yy = cast(torch.Tensor, d.y).double()
        sum_x   += xx.sum(dim=0)
        sumsq_x += (xx * xx).sum(dim=0)
        sum_y   += yy.sum(dim=0)
        sumsq_y += (yy * yy).sum(dim=0)
        n_pts   += xx.shape[0]

    pooled_mean_x = sum_x / n_pts
    pooled_var_x  = sumsq_x / n_pts - pooled_mean_x ** 2
    pooled_mean_y = sum_y / n_pts
    pooled_var_y  = sumsq_y / n_pts - pooled_mean_y ** 2

    print(f"  x mean (should be ~0): {pooled_mean_x.tolist()}")
    print(f"  x var  (should be ~1): {pooled_var_x.tolist()}")
    print(f"  y mean (should be ~0): {pooled_mean_y.tolist()}")
    print(f"  y var  (should be ~1): {pooled_var_y.tolist()}")

    assert torch.allclose(pooled_mean_x, torch.zeros(6, dtype=torch.float64), atol=1e-3), "pooled x mean not ~0"
    assert torch.allclose(pooled_mean_y, torch.zeros(4, dtype=torch.float64), atol=1e-3), "pooled y mean not ~0"
    assert torch.allclose(pooled_var_x,  torch.ones(6,  dtype=torch.float64), atol=1e-2), "pooled x variance not ~1"
    assert torch.allclose(pooled_var_y,  torch.ones(4,  dtype=torch.float64), atol=1e-2), "pooled y variance not ~1"

    print("Normalization Checks: OK")


def _check_cache(ds, data, data_dir, cache_dir):
    print("\n--- Cache Checks ---")
    ds2 = DrivAerDataset(
        data_dir,
        cache_dir=cache_dir,
    )
    data2: Data = ds2[0]

    print('  Checking that `x` and `y` data matches between successive dataset instantiations...')
    assert data2.x is not None, "data2.x is None"
    assert torch.allclose(data.x, data2.x),     "cached x differs"
    assert type(data2.y) is torch.Tensor, "data2.y is not a torch.Tensor"
    assert torch.allclose(data.y, data2.y),     "cached y differs"

    print('  Checking that norm_coef matches between successive dataset instantiations...')
    for a, b in zip(ds.norm_coef, ds2.norm_coef):
        assert torch.allclose(a, b), "norm_coef mismatch between successive dataset instances"
    
    print("Cache Checks: OK")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess & cache the DrivAerML runs, then run dataset smoke tests."
    )
    parser.add_argument("--data-dir", default="./data/drivaer_data",
                        help="Directory holding run_*/boundary_*.vtp")
    parser.add_argument("--cache-dir", default=None,
                        help="Where to write the per-run .pt cache (default: <data-dir>/cache)")
    parser.add_argument("--decimate", action=argparse.BooleanOptionalAction, default=False,
                        help="Decimate the mesh before extracting features")
    parser.add_argument("--target-reduction", type=float, default=0.99,
                        help="Decimation reduction fraction (only used with --decimate)")
    parser.add_argument("--max-runs", type=int, default=50,
                        help="Highest run id to scan for")
    args = parser.parse_args()

    print(f"Cache dir      : {args.cache_dir or os.path.join(args.data_dir, 'cache/')}")
    print(f"Decimation     : {f'on (reduction={args.target_reduction})' if args.decimate else 'off'}")

    ds = DrivAerDataset(
        data_dir=args.data_dir,
        decimate=args.decimate,
        target_reduction=args.target_reduction,
        cache_dir=args.cache_dir,
    )

    data = _check_shapes(ds)
    _check_normalization(ds)
    _check_cache(ds, data, args.data_dir, args.cache_dir)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()

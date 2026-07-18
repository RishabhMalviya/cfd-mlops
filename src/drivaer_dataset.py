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


class DrivAerDataset(Dataset):
    def __init__(self, data_dir, run_ids, decimate=False, target_reduction=0.99, cache_dir=None, norm_coef=None, norm_cache=None):
        self.data_dir = data_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, 'cache/')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.target_reduction = target_reduction

        self.run_ids = [
            i for i in run_ids
            if os.path.exists(os.path.join(data_dir, f"run_{i}", f"boundary_{i}.vtp"))
        ]

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

        pos     = mesh.cell_centers().points.astype(np.float32)     # (N, 3)
        normals = mesh.cell_normals.astype(np.float32)              # (N, 3)
        x = np.concatenate([pos, normals], axis=1)                  # (N, 6)

        pressure = mesh["pMeanTrim"].astype(np.float32)                  # (N,)
        wss      = mesh["wallShearStressMeanTrim"].astype(np.float32)    # (N, 3)
        y = np.concatenate([wss, pressure[:, np.newaxis]], axis=1)                  # (N, 4) = [wx, wy, wz, p]

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


if __name__ == "__main__":
    # --- Run Preprocessing & Caching ---
    data_dir = "./data/drivaer_data"

    all_run_ids = list(range(1, 51))
    avail_run_ids   = [i for i in all_run_ids if os.path.exists(os.path.join(data_dir, f"run_{i}", f"boundary_{i}.vtp"))]
    print(f"Available runs : {len(avail_run_ids)}/50  {avail_run_ids}")

    ds = DrivAerDataset(data_dir=data_dir, run_ids=avail_run_ids)


    # --- Shape Checks ---
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


    # --- Normalization Checks (checks *dataset-wide* mean ~= 0 and variance ~= 1) ---
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


    # --- Cache Checks ---
    print("\n--- Cache Checks ---")
    ds2 = DrivAerDataset(data_dir, run_ids=avail_run_ids)
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


    print("\nAll smoke tests passed.")

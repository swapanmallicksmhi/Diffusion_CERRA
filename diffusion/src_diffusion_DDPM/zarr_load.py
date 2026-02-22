"""
Zarr Data Loader for Diffusion Training
ERA5 -> CERRA (0h forecast pairing from same timestamp)

Author: Swapan Mallick
"""

import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# ZARR PAIRED DATASET (0h pairing via same index)
# ============================================================

class ZarrPairedDataset0h(Dataset):
    """
    Pairs ERA5 and CERRA fields at the same timestamp index.
    """

    def __init__(
        self,
        zarr_path: str,
        era5_var: str,
        cerra_var: str,
        indices: list,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ):
        super().__init__()
        self.store = zarr.open(zarr_path, mode="r")

        self.era5_var = era5_var
        self.cerra_var = cerra_var
        self.indices = indices

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]

        era5 = self.store[self.era5_var][global_idx].astype(np.float32)
        cerra = self.store[self.cerra_var][global_idx].astype(np.float32)

        era5 = torch.from_numpy(era5).unsqueeze(0)
        cerra = torch.from_numpy(cerra).unsqueeze(0)

        if self.mean is not None and self.std is not None:
            era5 = (era5 - self.mean) / self.std
            cerra = (cerra - self.mean) / self.std

        # Match diffusion pipeline scaling [-1,1]
        era5 = torch.clamp(era5, -10, 10)
        cerra = torch.clamp(cerra, -10, 10)

        return era5, cerra


# ============================================================
# LOAD FUNCTION (same API style as your PNG loader)
# ============================================================

def load_data(
    zarr_path: str,
    batch_size: int,
    era5_var: str = "t2m_era5_mean",
    cerra_var: str = "t2m_cerra_mean",
    shuffle: bool = True,
):
    """
    Creates diffusion-ready DataLoader from Zarr store.
    """

    store = zarr.open(zarr_path, mode="r")

    n_samples = store[era5_var].shape[0]
    indices = list(range(n_samples))

    # --------------------------------------------------
    # Load correct stats (match preprocessing order!)
    # --------------------------------------------------
    full_variable_order = [
        "t2m_cerra_mean",
        "t2m_cerra_members",
        "t2m_cerra_std",
        "t2m_era5_mean",
        "t2m_era5_members",
        "t2m_era5_std",
    ]

    all_mean = store["mean"][:]
    all_std = store["std"][:]

    era5_idx = full_variable_order.index(era5_var)
    cerra_idx = full_variable_order.index(cerra_var)

    mean = torch.tensor(all_mean[era5_idx], dtype=torch.float32).view(1, 1, 1)
    std = torch.tensor(all_std[era5_idx], dtype=torch.float32).view(1, 1, 1)

    dataset = ZarrPairedDataset0h(
        zarr_path=zarr_path,
        era5_var=era5_var,
        cerra_var=cerra_var,
        indices=indices,
        mean=mean,
        std=std,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    return loader

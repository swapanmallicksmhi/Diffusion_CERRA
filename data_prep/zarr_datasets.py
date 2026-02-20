#!/usr/bin/env python3
"""
Zarr Weather Dataset Loader
• Date-based splitting
• Correct subset normalization
• PyTorch Dataset wrapper
• Normalization verification
"""

import argparse
import zarr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# ============================================================
# DATE SPLITTER
# ============================================================

class DateRangeSplitter:
    def __init__(self, zarr_path, train_range, val_range, test_range):
        self.store = zarr.open(zarr_path, mode="r")
        self.ranges = {
            "train": train_range,
            "val": val_range,
            "test": test_range
        }

    def get_splits(self):
        time_arr = self.store["valid_time"][:]

        # Convert assuming seconds since epoch
        timestamps = pd.to_datetime(time_arr, unit="s", origin="unix", utc=True)

        splits = {}
        for name, (start, end) in self.ranges.items():
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts   = pd.Timestamp(end, tz="UTC")

            mask = (timestamps >= start_ts) & (timestamps <= end_ts)
            indices = np.where(mask)[0]
            splits[name] = indices.tolist()

            print(f"✔ {name}: {len(indices)} samples")

        return splits


# ============================================================
# NORMALIZER (Z-score)
# ============================================================

class ZarrNormalizer:
    def __init__(self, mean, std):
        self.mean = mean.view(-1, 1, 1)
        self.std  = std.view(-1, 1, 1)
        self.std[self.std == 0] = 1.0

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


# ============================================================
# DATASET
# ============================================================

class ZarrWeatherDataset(Dataset):
    def __init__(self, zarr_path, indices, variables, normalizer=None):
        self.store = zarr.open(zarr_path, mode="r")
        self.indices = indices
        self.variables = variables
        self.normalizer = normalizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]

        data_list = []
        for var in self.variables:
            data = self.store[var][global_idx].astype(np.float32)
            data_list.append(data)

        tensor = torch.from_numpy(np.stack(data_list, axis=0))

        if self.normalizer:
            tensor = self.normalizer.normalize(tensor)

        return tensor


# ============================================================
# VISUAL CHECK
# ============================================================

def plot_check(original, normalized, denormalized, var_names):
    orig = original.numpy()
    norm = normalized.numpy()
    denorm = denormalized.numpy()

    C = orig.shape[0]

    fig, axes = plt.subplots(C, 3, figsize=(15, 4*C), constrained_layout=True)

    if C == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(C):
        vmin, vmax = orig[i].min(), orig[i].max()

        im0 = axes[i,0].imshow(orig[i], vmin=vmin, vmax=vmax)
        axes[i,0].set_title(f"{var_names[i]} Original")
        plt.colorbar(im0, ax=axes[i,0])

        im1 = axes[i,1].imshow(norm[i], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[i,1].set_title("Normalized")
        plt.colorbar(im1, ax=axes[i,1])

        im2 = axes[i,2].imshow(denorm[i], vmin=vmin, vmax=vmax)
        axes[i,2].set_title("Denormalized")
        plt.colorbar(im2, ax=axes[i,2])

        for j in range(3):
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])

    plt.savefig("normalization_report.png", dpi=200)
    print("✔ normalization_report.png saved")


# ============================================================
# MAIN
# ============================================================

def main(zarr_path):

    print("Opening Zarr:", zarr_path)
    store = zarr.open(zarr_path, mode="r")

    # --------------------------------------------------
    #  DATE SPLITS
    # --------------------------------------------------
    splitter = DateRangeSplitter(
        train_range=("2020-01-01", "2020-05-30"),
        val_range=("2020-06-01", "2020-06-30"),
        test_range=("2020-07-01", "2020-07-31"),
        zarr_path=zarr_path
    )

    splits = splitter.get_splits()

    # --------------------------------------------------
    #  FULL VARIABLE ORDER (CRITICAL)
    # --------------------------------------------------
    full_variable_order = [
        "t2m_cerra_mean",
        "t2m_cerra_members",
        "t2m_cerra_std",
        "t2m_era5_mean",
        "t2m_era5_members",
        "t2m_era5_std",
    ]

    # Variables you want to train on
    target_vars = [
        "t2m_cerra_mean",
        "t2m_cerra_std",
        "t2m_era5_mean",
        "t2m_era5_std",
    ]

    # --------------------------------------------------
    #  LOAD & SUBSET STATS CORRECTLY
    # --------------------------------------------------
    all_mean = store["mean"][:]
    all_std  = store["std"][:]

    var_indices = [full_variable_order.index(v) for v in target_vars]

    mean_stats = torch.from_numpy(all_mean[var_indices].astype(np.float32))
    std_stats  = torch.from_numpy(all_std[var_indices].astype(np.float32))

    normalizer = ZarrNormalizer(mean_stats, std_stats)

    # --------------------------------------------------
    #  BUILD DATASETS
    # --------------------------------------------------
    train_raw = ZarrWeatherDataset(
        zarr_path, splits["train"], target_vars, normalizer=None
    )

    train_norm = ZarrWeatherDataset(
        zarr_path, splits["train"], target_vars, normalizer=normalizer
    )

    # --------------------------------------------------
    #  TEST SAMPLE
    # --------------------------------------------------
    idx = 0
    original = train_raw[idx]
    normalized = train_norm[idx]
    denormalized = normalizer.denormalize(normalized)

    print("Sample shape:", original.shape)

    plot_check(original, normalized, denormalized, target_vars)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", type=str, required=True)
    args = parser.parse_args()

    main(args.zarr_path)

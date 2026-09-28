#!/usr/bin/env python3

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
ZARR_PATH = "/cfs/klemming/scratch/y/yarongc/all_zarr-out_2024.zar"
#OUTPUT_FILE = "cerra_vs_era5_std_Jan_Jun_2024.png"
OUTPUT_FILE = "cerra_vs_era5_std_test.png"

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

DPI = 100

# -------------------------------------------------
# OPEN DATASET
# -------------------------------------------------
print("Opening dataset...")
ds = xr.open_dataset(ZARR_PATH, engine="zarr")

# Sort time (important)
ds = ds.sortby("valid_time")

# Select time range
ds = ds.sel(valid_time=slice(START_DATE, END_DATE))

print("Time range:",
      ds.valid_time.values[0],
      "to",
      ds.valid_time.values[-1])

# -------------------------------------------------
# COMPUTE DOMAIN MEAN PER MEMBER
# -------------------------------------------------
def domain_mean_members(data_array):
    """Compute spatial mean over y,x for each ensemble member"""
    return data_array.mean(dim=("y", "x"))

cerra = domain_mean_members(ds["t2m_cerra_members"]).compute()
era5  = domain_mean_members(ds["t2m_era5_members"]).compute()

# Convert to pandas
cerra_df = cerra.to_pandas().T
era5_df  = era5.to_pandas().T

cerra_df["month"] = cerra_df.index.month
era5_df["month"]  = era5_df.index.month

# -------------------------------------------------
# SELECT JANUARY (1) AND JUNE (6)
# -------------------------------------------------
months = [1, 6]
month_names = {1: "January", 6: "June"}

cerra_stats = {}
era5_stats = {}

for m in months:
    data_cerra = cerra_df[cerra_df["month"] == m].iloc[:, :-1].values
    data_era5  = era5_df[era5_df["month"] == m].iloc[:, :-1].values

    cerra_stats[m] = {
        "mean": np.mean(data_cerra, axis=1),
        "std": np.std(data_cerra, axis=1)
    }
    era5_stats[m] = {
        "mean": np.mean(data_era5, axis=1),
        "std": np.std(data_era5, axis=1)
    }

# -------------------------------------------------
# PLOT MEAN STD
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

for ax, m in zip(axes, months):

    # CERRA (green)
    mean_c = cerra_stats[m]["mean"]
    std_c  = cerra_stats[m]["std"]
    time_c = np.arange(len(mean_c))
    ax.fill_between(time_c, mean_c - std_c, mean_c + std_c, color="green", alpha=0.5)
    ax.plot(time_c, mean_c, color="green", label="CERRA Mean")

    # ERA5 (red)
    mean_e = era5_stats[m]["mean"]
    std_e  = era5_stats[m]["std"]
    time_e = np.arange(len(mean_e))
    ax.fill_between(time_e, mean_e - std_e, mean_e + std_e, color="red", alpha=0.5)
    ax.plot(time_e, mean_e, color="red", label="ERA5 Mean")

    ax.set_title(month_names[m])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Domain Mean Temperature")
    ax.legend()

fig.suptitle("CERRA vs ERA5 Ensemble: Mean  Std Deviation (Jan & Jun 2024)")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=DPI)
plt.close()

print("Saved:", OUTPUT_FILE)

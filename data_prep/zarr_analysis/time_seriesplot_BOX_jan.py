#!/usr/bin/env python3

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
ZARR_PATH = "./zarr_in"
OUTPUT_FILE = "cerra_vs_era5_boxplot_Jan_Jun_2024.png"

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

DPI = 150

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

cerra_data = {}
era5_data  = {}

for m in months:
    cerra_data[m] = cerra_df[cerra_df["month"] == m].iloc[:, :-1].values.flatten()
    era5_data[m]  = era5_df[era5_df["month"] == m].iloc[:, :-1].values.flatten()

# -------------------------------------------------
# PLOT
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

for ax, m in zip(axes, months):

    # Plot CERRA (green)
    bp1 = ax.boxplot(
        cerra_data[m],
        positions=[1],
        widths=0.4,
        patch_artist=True
    )
    for patch in bp1['boxes']:
        patch.set_facecolor("green")

    # Plot ERA5 (red)
    bp2 = ax.boxplot(
        era5_data[m],
        positions=[2],
        widths=0.4,
        patch_artist=True
    )
    for patch in bp2['boxes']:
        patch.set_facecolor("red")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["CERRA", "ERA5"])
    ax.set_title(month_names[m])
    ax.set_ylabel("Domain Mean Temperature")

fig.suptitle("CERRA vs ERA5 Ensemble Distribution\n(January and June 2024)")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=DPI)
plt.close()

print("Saved:", OUTPUT_FILE)

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
OUTPUT_FILE = "cerra_vs_era5_boxplot_2024.png"

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

DPI = 100

# -------------------------------------------------
# OPEN DATASET
# -------------------------------------------------
print("Opening dataset...")
ds = xr.open_dataset(ZARR_PATH, engine="zarr")

# Fix slicing issue
ds = ds.sortby("valid_time")

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

# -------------------------------------------------
# CONVERT TO PANDAS FOR GROUPING
# -------------------------------------------------
cerra_df = cerra.to_pandas().T   # index=time, columns=members
era5_df  = era5.to_pandas().T

# Add month column
cerra_df["month"] = cerra_df.index.month
era5_df["month"]  = era5_df.index.month

# -------------------------------------------------
# PREPARE MONTHLY BOXPLOT DATA
# -------------------------------------------------
cerra_monthly = [cerra_df[cerra_df["month"] == m].iloc[:, :-1].values.flatten()
                 for m in range(1, 13)]

era5_monthly  = [era5_df[era5_df["month"] == m].iloc[:, :-1].values.flatten()
                 for m in range(1, 13)]

# -------------------------------------------------
# PLOT
# -------------------------------------------------
plt.figure(figsize=(14, 6))

positions_cerra = np.arange(1, 13) - 0.2
positions_era5  = np.arange(1, 13) + 0.2

plt.boxplot(cerra_monthly,
            positions=positions_cerra,
            widths=0.3,
            patch_artist=True)

plt.boxplot(era5_monthly,
            positions=positions_era5,
            widths=0.3,
            patch_artist=True)

plt.xticks(np.arange(1, 13),
           ['Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec'])

plt.xlabel("Month (2024)")
plt.ylabel("Domain Mean Temperature")
plt.title("CERRA vs ERA5 Ensemble Distribution (Monthly Boxplots, 2024)")

plt.legend(["CERRA", "ERA5"])

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=DPI)
plt.close()

print("Saved:", OUTPUT_FILE)

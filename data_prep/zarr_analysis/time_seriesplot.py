#!/usr/bin/env python3

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
ZARR_PATH = "./zarr_in"
OUTPUT_DIR = "./ensemble_timeseries_2024"

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

DPI = 150

# -------------------------------------------------
# PREPARE OUTPUT DIRECTORY
# -------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# OPEN DATASET
# -------------------------------------------------
print("Opening dataset...")
ds = xr.open_dataset(ZARR_PATH, engine="zarr")

# Sort time to avoid slicing error
ds = ds.sortby("valid_time")

# Select time range
ds = ds.sel(valid_time=slice(START_DATE, END_DATE))

print("Selected time range:")
print(ds.valid_time.values[0], "to", ds.valid_time.values[-1])
print("Number of timesteps:", ds.sizes["valid_time"])

# -------------------------------------------------
# FUNCTION TO COMPUTE DOMAIN MEAN
# -------------------------------------------------
def compute_domain_mean(data_array):
    """
    Compute spatial mean over y and x dimensions.
    """
    return data_array.mean(dim=("y", "x"))

# -------------------------------------------------
# FUNCTION TO PLOT ENSEMBLE TIMESERIES
# -------------------------------------------------
def plot_ensemble_timeseries(data_array, dataset_name, output_file):

    print(f"Processing {dataset_name} ...")

    # Spatial mean (per member, per time)
    domain_mean = compute_domain_mean(data_array)

    # Compute explicitly (important for Dask)
    domain_mean = domain_mean.compute()

    time = domain_mean.valid_time.values
    members = domain_mean.number.values

    plt.figure(figsize=(12, 6))

    for m in members:
        plt.plot(
            time,
            domain_mean.sel(number=m),
            label=f"Member {m}",
            linewidth=1
        )

    plt.xlabel("Time")
    plt.ylabel("Domain Mean Temperature")
    plt.title(f"{dataset_name}\nDomain Mean Time Series ({START_DATE} to {END_DATE})")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    plt.savefig(output_file, dpi=DPI)
    plt.close()

    print("Saved:", output_file)

# -------------------------------------------------
# PROCESS CERRA ENSEMBLE
# -------------------------------------------------
if "t2m_cerra_members" in ds:
    plot_ensemble_timeseries(
        ds["t2m_cerra_members"],
        "CERRA Ensemble Members",
        os.path.join(OUTPUT_DIR, "cerra_ensemble_timeseries.png")
    )

# -------------------------------------------------
# PROCESS ERA5 ENSEMBLE
# -------------------------------------------------
if "t2m_era5_members" in ds:
    plot_ensemble_timeseries(
        ds["t2m_era5_members"],
        "ERA5 Ensemble Members",
        os.path.join(OUTPUT_DIR, "era5_ensemble_timeseries.png")
    )

print("\nAll ensemble time series plots generated successfully.")

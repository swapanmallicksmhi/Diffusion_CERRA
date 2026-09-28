#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
ZARR_PATH = "./zarr_in"
OUTPUT_DIR = "./spatial_means_2024"

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"

DPI = 150

# -------------------------------------------------
# CREATE OUTPUT DIRECTORY
# -------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# OPEN DATASET
# -------------------------------------------------
print("Opening dataset...")
ds = xr.open_dataset(ZARR_PATH, engine="zarr")

# IMPORTANT: sort time to avoid slicing error
if "valid_time" in ds.coords:
    ds = ds.sortby("valid_time")

print("Dataset opened successfully.")
print("Time range in dataset:")
print("First:", ds.valid_time.values[0])
print("Last :", ds.valid_time.values[-1])

# -------------------------------------------------
# SELECT TIME RANGE
# -------------------------------------------------
print(f"\nSelecting time slice {START_DATE} to {END_DATE} ...")
ds_sel = ds.sel(valid_time=slice(START_DATE, END_DATE))

print("Selected time range:")
print("First:", ds_sel.valid_time.values[0])
print("Last :", ds_sel.valid_time.values[-1])
print("Number of timesteps:", ds_sel.sizes["valid_time"])

# -------------------------------------------------
# GET LAT/LON
# -------------------------------------------------
lat = ds["latitude"]
lon = ds["longitude"]

# -------------------------------------------------
# FUNCTION TO PLOT MAP
# -------------------------------------------------
def plot_spatial_map(data_array, title, output_file):

    plt.figure(figsize=(10, 6))

    # compute explicitly (important for Dask-backed arrays)
    data_array = data_array.compute()

    plt.pcolormesh(lon, lat, data_array, shading="auto")
    plt.colorbar(label=data_array.name)

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    plt.savefig(output_file, dpi=DPI)
    plt.close()

    print("Saved:", output_file)

# -------------------------------------------------
# VARIABLES WITHOUT ENSEMBLE DIMENSION
# -------------------------------------------------
variables_to_plot = [
    "t2m_cerra_mean",
    "t2m_cerra_std",
    "t2m_era5_mean",
    "t2m_era5_std",
]

for var in variables_to_plot:

    if var not in ds_sel:
        continue

    print(f"\nProcessing {var} ...")

    spatial_mean = ds_sel[var].mean(dim="valid_time")

    title = f"{var}\nSpatial Mean ({START_DATE} to {END_DATE})"
    output_file = os.path.join(
        OUTPUT_DIR,
        f"{var}_spatial_mean_{START_DATE}_{END_DATE}.png"
    )

    plot_spatial_map(spatial_mean, title, output_file)

# -------------------------------------------------
# ENSEMBLE VARIABLES
# -------------------------------------------------
ensemble_vars = [
    "t2m_cerra_members",
    "t2m_era5_members",
]

for var in ensemble_vars:

    if var not in ds_sel:
        continue

    print(f"\nProcessing {var} (ensemble mean) ...")

    spatial_mean = (
        ds_sel[var]
        .mean(dim="number")        # ensemble mean
        .mean(dim="valid_time")    # time mean
    )

    title = f"{var} (Ensemble Mean)\nSpatial Mean ({START_DATE} to {END_DATE})"
    output_file = os.path.join(
        OUTPUT_DIR,
        f"{var}_ensemble_spatial_mean_{START_DATE}_{END_DATE}.png"
    )

    plot_spatial_map(spatial_mean, title, output_file)

print("\nAll spatial mean maps generated successfully.")

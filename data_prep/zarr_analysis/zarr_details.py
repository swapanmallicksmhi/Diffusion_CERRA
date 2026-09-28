import xarray as xr
import numpy as np
import pandas as pd

# Open Zarr dataset
ds = xr.open_dataset("./zarr_in", engine="zarr")

print("\n================ DATASET SUMMARY ================")
print(ds)

print("\n================ DIMENSIONS ================")
for dim, size in ds.sizes.items():
    print(f"{dim}: {size}")

print("\n================ COORDINATES ================")
for coord in ds.coords:
    print(f"{coord}: shape={ds[coord].shape}, dtype={ds[coord].dtype}")

print("\n================ DATA VARIABLES ================")
for var in ds.data_vars:
    print(f"{var}: shape={ds[var].shape}, dtype={ds[var].dtype}")

print("\n================ GLOBAL ATTRIBUTES ================")
for attr, value in ds.attrs.items():
    print(f"{attr}: {value}")

print("\n================ CHUNKING INFO ================")
for var in ds.data_vars:
    print(f"{var}: chunks={ds[var].chunks}")

# --------------------------------------------------
# Time information (if valid_time exists)
# --------------------------------------------------
if "valid_time" in ds:
    vt = ds["valid_time"].values
    print("\n================ TIME INFORMATION ================")
    print("n_times:", ds.sizes.get("valid_time", None))
    print("first:", vt[0])
    print("last :", vt[-1])

    years = pd.DatetimeIndex(vt).year
    print("year min/max:", years.min(), years.max())
    print("unique years:", sorted(set(years)))

# --------------------------------------------------
# Variable statistics
# --------------------------------------------------
print("\n================ VARIABLE STATISTICS ================")

for var in ds.data_vars:
    data = ds[var]

    print(f"\n--- {var} ---")

    try:
        print("min :", data.min().values)
        print("max :", data.max().values)
        print("mean:", data.mean().values)
        print("std :", data.std().values)
        print("NaNs:", np.isnan(data).sum().values)
    except Exception as e:
        print("Could not compute stats:", e)

# --------------------------------------------------
# Dataset memory size estimate
# --------------------------------------------------
print("\n================ MEMORY ESTIMATE ================")
try:
    size_bytes = ds.nbytes
    print(f"Approx dataset size in memory: {size_bytes / 1e9:.3f} GB")
except Exception:
    print("Could not compute dataset size.")

print("\n================ END OF REPORT ================")

import xarray as xr
import numpy as np
import pandas as pd

#ds = xr.open_dataset("/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/zarr-out_2015-2024", engine="zarr")
ds = xr.open_dataset("./zarr_in", engine="zarr")

vt = ds["valid_time"].values
print("n_times:", ds.sizes.get("valid_time", None))
print("first:", vt[0])
print("last :", vt[-1])

print(ds["t2m_cerra_mean"].min().values)
print(ds["t2m_cerra_mean"].max().values)

print(np.isnan(ds["t2m_cerra_mean"]).sum().values)
vt = ds["valid_time"].values
print("n_times:", ds.sizes.get("valid_time", None))
print("first:", vt[0])
print("last :", vt[-1])

years = pd.DatetimeIndex(vt).year
print("year min/max:", years.min(), years.max())
print("years:", sorted(set(years))[:5], "...", sorted(set(years))[-5:])

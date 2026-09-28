import xarray as xr

path = "/cfs/klemming/scratch/y/yarongc/2015-2023-no2019.zarr"
ds = xr.open_zarr(path)

print(ds)
print(ds["t2m_cerra_mean"])
print(ds["t2m_cerra_mean"].dims)
print(ds["t2m_cerra_mean"].shape)
print(ds["valid_time"])
print(ds["latitude"].shape, ds["longitude"].shape)

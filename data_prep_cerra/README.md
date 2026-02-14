# Data prep for CERRA

import xarray as xr

# Open the Zarr store
#ds = xr.open_zarr('/ec/res4/scratch/smcd/output/yarong_data/cera_era5_128x128', consolidated=True)
ds = xr.open_zarr('/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/zarr-out_2015-2024', consolidated=True)
ds = xr.open_zarr('/ec/res4/scratch/smcd/output/yarong_data/cera_era5_256x256', consolidated=True, zarr_format=2)


print(ds)
print("\nVariables:", list(ds.data_vars))
#print("Time steps:", ds.valid_time.values)
print(ds["mean"])
temp_ds = ds['t2m_cerra_mean'].isel(valid_time=0)

## Files needed in stage 1 of the project 
- era5-eda ensemble on 
- scripts 
  - fetch_era5eda_t2m_nc_ens.py /lus/h2resw01/scratch/swe4281/CERRA_DATA2026/CERRA_DATA
  - fetch_cerra_t2m_nc_ens.py   /lus/h2resw01/scratch/swe4281/CERRA_DATA2026/ERA5EDA_DATA


## Create zarr dataset
```bash
cd transform
conda activate mamba-env
python3 main.py \
    --grid-size 256 \
    --cerra-dir "/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/CERRA_DATA/" \
    --era5-dir  "/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/ERA5EDA_DATA/" \
    --zarr-path "/ec/res4/scratch/smcd/output/yarong_data/cera_era5_256x256"
```
  


## Ernvironment
### ON ECMWF

- Activate the env 
```bash
export TMPDIR=~/tmp
conda activate mamba-env
```


- Setup The env
```bash
module load conda/24.11.3-2

conda create -n mamba-env -c conda-forge mamba
conda activate mamba-env
export TMPDIR=~/tmp
mamba install -y -c conda-forge \
    xarray \
    netcdf4 \
    h5netcdf \
    scipy \
    numpy \
    pandas \
    matplotlib \
    cartopy \
    proj \
    pyproj \
    shapely \
    geos \
    cfgrib \
    eccodes \
    dask \
    zarr \
    pytorch-lightning
```
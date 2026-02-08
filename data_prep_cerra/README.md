# Data prep for CERRA



## Files needed in stage 1 of the project 
- era5-eda ensemble on 
- scripts 
  - fetch_era5eda_t2m_nc_ens.py /lus/h2resw01/scratch/swe4281/CERRA_DATA2026/CERRA_DATA
  - fetch_cerra_t2m_nc_ens.py   /lus/h2resw01/scratch/swe4281/CERRA_DATA2026/ERA5EDA_DATA






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
    zarr
```
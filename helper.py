import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.colors import BoundaryNorm
import xesmf as xe

# ==========================================
# 1. FILE FINDER
# ==========================================
def find_matching_files(era5land_dir, variables):
    import glob

    matched_files = []

    for var in variables:

        pattern = os.path.join(
            era5land_dir,
            f"{var}_12km_6hr_*.nc"
        )

        files = glob.glob(pattern)
        print(files)

        if files:
            for f in files:
                matched_files.append(f)
                print(f"[FOUND] {os.path.basename(f)}")
        else:
            print(f"[MISSING] {var}")

    return sorted(matched_files)


# ==========================================
# 2. HIGH-RESOLUTION FILE FINDER
# ==========================================
def find_highres_file(era5land_dir):
    import glob

    pattern = os.path.join(
        era5land_dir,
        "tas_3km_6hr_*.nc"
    )

    files = sorted(glob.glob(pattern))
    print(files)

    if files:
        print(f"[FOUND 3KM] {os.path.basename(files[0])}")
        return files[0]
    print("[MISSING] 3-km tas file")
    return None

# ==========================================
# 3. INTERPOLATION
# ==========================================
def regrid_to_highres(ds, ds_highres, regridder=None, source_lat=None, source_lon=None):

    data_vars = list(ds.data_vars)

    if not data_vars:
        raise ValueError("No data variables found in source dataset")

    ds_in = ds.copy()

    for var in data_vars:
        ds_in[var] = ds_in[var].where(np.isfinite(ds_in[var]))
        ds_in[var] = ds_in[var].where(np.abs(ds_in[var]) < 1.0e19)

    source_grid = xr.Dataset(
        {
            'lat': (('y', 'x'), ds_in['lat'].values),
            'lon': (('y', 'x'), ds_in['lon'].values)
        }
    )

    target_grid = xr.Dataset(
        {
            'lat': (('y', 'x'), ds_highres['lat'].values),
            'lon': (('y', 'x'), ds_highres['lon'].values)
        }
    )

    if regridder is None:

        print("Creating bilinear interpolation weights...")

        regridder = xe.Regridder(
            source_grid,
            target_grid,
            'bilinear',
            periodic=False,
            reuse_weights=False
        )

        source_lat = ds_in['lat'].values.copy()
        source_lon = ds_in['lon'].values.copy()

    else:

        if not np.allclose(ds_in['lat'].values, source_lat, equal_nan=True):
            raise ValueError("12-km latitude grid differs between files")

        if not np.allclose(ds_in['lon'].values, source_lon, equal_nan=True):
            raise ValueError("12-km longitude grid differs between files")

        print("Reusing interpolation weights...")

    ds_interp = regridder(ds_in, keep_attrs=True)

    source_coverage = xr.DataArray(
        np.ones((ds_in.sizes['y'], ds_in.sizes['x']), dtype=np.float32),
        dims=('y', 'x'),
        coords={
            'y': ds_in['y'],
            'x': ds_in['x'],
            'lat': (('y', 'x'), ds_in['lat'].values),
            'lon': (('y', 'x'), ds_in['lon'].values)
        }
    )

    target_coverage = regridder(source_coverage)
    valid_mask = target_coverage > 0.5

    ds_interp = ds_interp.assign_coords(
        time=ds_in['time'],
        y=ds_highres['y'],
        x=ds_highres['x'],
        lat=(('y', 'x'), ds_highres['lat'].values),
        lon=(('y', 'x'), ds_highres['lon'].values)
    )

    valid_mask = valid_mask.assign_coords(
        y=ds_highres['y'],
        x=ds_highres['x']
    )

    for var in ds_interp.data_vars:
        if var in ds_in.data_vars:
            original_attrs = ds[var].attrs.copy()
            ds_interp[var] = ds_interp[var].where(valid_mask)
            ds_interp[var].attrs = original_attrs

    return ds_interp, regridder, source_lat, source_lon


# ==========================================
# 4. SUMMARY STATS
# ==========================================
def calculate_summary_stats(zarr_path):

    ds = xr.open_dataset(zarr_path, engine='zarr', chunks={})

    stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    var_names = []

    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.number) and 'time' in ds[var].dims:
            d = ds[var]

            stats['min'].append(float(d.min(skipna=True)))
            stats['max'].append(float(d.max(skipna=True)))
            stats['mean'].append(float(d.mean(skipna=True)))
            stats['std'].append(float(d.std(skipna=True)))

            var_names.append(var)

    stats_ds = xr.Dataset(
        {k: (('variable',), v) for k, v in stats.items()},
        coords={'variable': var_names}
    )

    stats_ds.to_zarr(zarr_path, mode='a', zarr_format=2)

    ds.close()


# ==========================================
# 5. ZARR SAVING
# ==========================================
def save_to_zarr(zarr_path, ds_era5land):

    ds_to_save = ds_era5land.sortby('time')

    ds_to_save = ds_to_save.chunk({
        'time': 1,
        'y': -1,
        'x': -1
    })

    zarr_group = os.path.join(zarr_path, '.zgroup')

    if not os.path.exists(zarr_group):
        print(f"Initializing Zarr store at {zarr_path}")
        ds_to_save.to_zarr(zarr_path, mode='w', zarr_format=2)
    else:
        print(f"Zarr store already exists at {zarr_path}")


# ==========================================
# 6. CROPPING
# ==========================================
def crop_era5land(ds, grid_size=106):

    x_size = ds.sizes['x']
    y_size = ds.sizes['y']

    ds_sub = ds.isel(
        y=slice(0, y_size),
        x=slice(0, x_size)
    )

    print(f"Grid: y[0:{y_size}], x[0:{x_size}]")
    print(f"Actual high-resolution grid size: y={y_size}, x={x_size}")

    return ds_sub


# ==========================================
# 7. COLORMAP
# ==========================================
def get_colormap(var):
    v = var.lower()

    if v in ["ta500", "ta700", "ta850", "ta950", "tas_3km", "tas_12km_interp"]:
        levels = np.arange(242, 312, 2)
        cmap = cmocean.cm.thermal.resampled(len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)
        return cmap, norm, levels
    else:
        return "viridis", None, None


# ==========================================
# 8. PLOTTING
# ==========================================
def plot_all(ds_era5land, idx, varname='tas_3km', title_suffix="", save_path=None):

    variable_info = {
        "ta500":          ("Air Temperature 500 hPa", "K"),
        "ta700":          ("Air Temperature 700 hPa", "K"),
        "ta850":          ("Air Temperature 850 hPa", "K"),
        "ta950":          ("Air Temperature 950 hPa", "K"),
        "tas_3km":        ("Near-Surface Air Temperature Native 3 km", "K"),
        "tas_12km_interp":("Near-Surface Air Temperature 12 km Interpolated to 3 km", "K"),
    }

    long_name, units = variable_info.get(
        varname,
        (varname, "")
    )

    data = ds_era5land[varname].isel(**idx)
    data = data.where(np.isfinite(data))

    cmap, norm, levels = get_colormap(varname)

    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
        cmap.set_bad(color='white')

    fig, ax = plt.subplots(
        1, 1,
        figsize=(10, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    if norm is not None:
        mesh = ax.pcolormesh(
            data['lon'],
            data['lat'],
            data,
            transform=ccrs.PlateCarree(),
            shading='auto',
            cmap=cmap,
            norm=norm
        )
    else:
        mesh = ax.pcolormesh(
            data['lon'],
            data['lat'],
            data,
            transform=ccrs.PlateCarree(),
            shading='auto',
            cmap=cmap
        )

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        linestyle='--'
    )

    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"{long_name} {title_suffix}")

    if levels is not None:
        cbar = plt.colorbar(
            mesh,
            ax=ax,
            shrink=0.8,
            pad=0.03,
            boundaries=levels,
            ticks=levels
        )
    else:
        cbar = plt.colorbar(
            mesh,
            ax=ax,
            shrink=0.8,
            pad=0.03
        )

    cbar.set_label(f"{long_name} ({units})")

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close(fig)

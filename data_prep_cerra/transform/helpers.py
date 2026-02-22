import os
import glob
import random
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 1. FILE FINDER
# ==========================================
def find_matching_files(cerra_dir, era5_dir, years, months, cycles):
    matched_pairs = []
    
    for year in years:
        for month in months:
            mm = f"{int(month):02d}"
            for cycle in cycles:
                yyyy = str(year)
                # Construct filenames
                cerra_filename = f"cerra_t2m_{yyyy}{mm}_{cycle}.nc"
                era5_filename = f"era5eda_t2m_{yyyy}{mm}_{cycle}.nc" # Note: 'erra5eda' typo handled

                cerra_path = os.path.join(cerra_dir, cerra_filename)
                era5_path = os.path.join(era5_dir, era5_filename)

                if os.path.exists(cerra_path) and os.path.exists(era5_path):
                    matched_pairs.append((cerra_path, era5_path))
                    print(f"[FOUND] {cycle}: {cerra_filename}")
                else:
                    print(f"[MISSING] Cycle {cycle} incomplete.")

    return matched_pairs

def calculate_summary_stats(zarr_path):
    """
    Calculates global min, max, mean, and std for all variables in the dataset.
    
    Args:
        file_path : Path to input dataset (Zarr).
        
    Returns:
        xr.Dataset: The original dataset merged with a new section where:
            - Dimension 'variable' contains the names of original variables.
            - Data variables are 'min', 'max', 'mean', 'std'.
    """
    stats = {
        'min': [],
        'max': [],
        'mean': [],
        'std': []
    }
    
    # Open dataset (assuming Zarr engine based on context)
    ds = xr.open_dataset(zarr_path, engine='zarr', chunks={})

 
    if  check_for_nans(ds):
        print("ERROR [calculate_summary_stats]: NaN values detected in the dataset to be saved. Please investigate.")
        exit(1)


    if  check_for_nans(ds.longitude) or check_for_nans(ds.latitude):
        print("ERROR [calculate_summary_stats]: NaN values detected in coordinates. Cannot save dataset with invalid coordinates.")
        exit(1)

    # List to store the variable names as we process them
    processed_var_names = []

    # Iterate over the actual data variables in the dataset
    for var in ds.data_vars:
        # Check if the variable is numeric before calculating stats
        if np.issubdtype(ds[var].dtype, np.number):
            d_var = ds[var]
            
            # Compute stats (float() ensures computation if lazy/dask)
            stats['min'].append(float(d_var.min()))
            stats['max'].append(float(d_var.max()))
            stats['mean'].append(float(d_var.mean()))
            stats['std'].append(float(d_var.std()))
            
            processed_var_names.append(var)

    # Construct the summary Dataset
    stats_ds = xr.Dataset(
        data_vars={
            stat_name: (('variable',), values) 
            for stat_name, values in stats.items()
        },
        coords={
            'variable': processed_var_names
        }
    )
    stats_ds.to_zarr(zarr_path, mode='a', zarr_format=2)
 

def check_for_nans(obj):
    """
    Checks if an Xarray Dataset or DataArray contains any NaN values.
    Returns a dictionary with results per variable if it's a Dataset.
    """
    results = None
    if isinstance(obj, xr.DataArray):
        nan_count = obj.isnull().sum().compute().item()
        has_nans = nan_count > 0
        print(f"Variable '{obj.name}': {nan_count} NaNs found. (Has NaNs: {has_nans})")
        if has_nans:
            return {obj.name: True}
        results
    
    elif isinstance(obj, xr.Dataset):
        results = {}
        print("Checking Dataset variables for NaNs...")
        for var in obj.data_vars:
            nan_count = obj[var].isnull().sum().compute().item()
            if results is None:
                results = {}
            if   nan_count > 0:
                results[var] = True
            print(f" - {var}: {nan_count} NaNs")
        return results


def save_to_zarr(zarr_path, cerra_mean, cerra_std, era5_mean, era5_std, 
                 ds_cerra=None, ds_era5=None, 
                 save_cerra_members=False, save_era5_members=False):
    """
    Merges statistical and member data, chunks it, and saves/appends to Zarr.
    """

    def rename_vars(ds, suffix):
        """Internal helper to rename variables with a suffix to avoid collisions."""
        return ds.rename({v: f"{v}_{suffix}" for v in ds.data_vars})
    
    
    ds_save_list = []

    # 1. Always add Mean and Std
    ds_save_list.append(rename_vars(cerra_mean, "cerra_mean"))
    ds_save_list.append(rename_vars(cerra_std, "cerra_std"))
    ds_save_list.append(rename_vars(era5_mean, "era5_mean"))
    ds_save_list.append(rename_vars(era5_std, "era5_std"))

    # 2. Conditionally add Full Members
    if save_cerra_members and ds_cerra is not None:
        ds_save_list.append(rename_vars(ds_cerra, "cerra_members"))
    
    if save_era5_members and ds_era5 is not None:
        ds_save_list.append(rename_vars(ds_era5, "era5_members"))

    # 3. Merge
    # compat='no_conflicts' preserves metadata and allows merging 
    # even if some coords are slightly different (within tolerance)
    ds_to_save = xr.merge(ds_save_list, compat='no_conflicts')
    if  check_for_nans(ds_to_save):
        print("ERROR: NaN values detected in the dataset to be saved. Please investigate.")
        exit(1)


    if  check_for_nans(ds_to_save.longitude) or check_for_nans(ds_to_save.latitude):
        print("ERROR: NaN values detected in coordinates. Cannot save dataset with invalid coordinates.")
        exit(1)
    # 4. Sort
    ds_to_save = ds_to_save.sortby('valid_time')
    
    # 5. Chunk on 'y' and 'x' dimensions
    # valid_time=1 allows efficient appending over time
    ds_to_save = ds_to_save.chunk({'valid_time': 1, 'y': -1, 'x': -1})

    # 6. Write to Zarr
    if not os.path.exists(zarr_path):
        print(f"Initializing Zarr store at {zarr_path}")
        ds_to_save.to_zarr(zarr_path, mode='w', zarr_format=2)
    else:
        print(f"Appending to Zarr store at {zarr_path}")
        # append_dim must match the dimension you are extending
        ds_to_save.to_zarr(zarr_path, mode='a', append_dim='valid_time', zarr_format=2)


# ==========================================
# 2. PROCESSING FUNCTIONS
# ==========================================

def crop_cerra(ds, lat_min, lat_max, lon_min, lon_max, grid_size=128):
    mask = (
        (ds.latitude >= lat_min) & (ds.latitude <= lat_max) &
        (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
    )
    arg_mask = np.argwhere(mask.values)

    if arg_mask.size == 0:
        raise ValueError(f"No data found for bounds: Lat({lat_min},{lat_max}), Lon({lon_min},{lon_max})")

    y_idx_min, x_idx_min = arg_mask.min(axis=0)
    max_y, max_x = ds.sizes['y'], ds.sizes['x']

    y_idx_end = y_idx_min + grid_size
    x_idx_end = x_idx_min + grid_size

    if x_idx_end > max_x or y_idx_end > max_y:
        raise IndexError(f"OUT OF BOUNDS: Crop ends at (y:{y_idx_end}, x:{x_idx_end})")

    ds_cerra_sub = ds.isel(
        x=slice(x_idx_min, x_idx_end),
        y=slice(y_idx_min, y_idx_end)
    )

    print(f"Crop Successful: Start(y:{y_idx_min}, x:{x_idx_min}) -> End(y:{y_idx_end}, x:{x_idx_end})")
    return ds_cerra_sub

def era5_to_cerra(ds_era5, ds_cerra_sub):
    """
    Interpolates ERA5 to CERRA grid and enforces EXACT coordinate matching.
    """
    # Normalize longitude
    ds_era5['longitude'] = ((ds_era5.longitude + 180) % 360) - 180
    ds_era5 = ds_era5.sortby(['latitude', 'longitude'])

    # Buffer for pre-cropping (improves interp speed)
    buffer = 2.0
    era5_lat_min = ds_cerra_sub.latitude.min().values - buffer
    era5_lat_max = ds_cerra_sub.latitude.max().values + buffer
    era5_lon_min = ds_cerra_sub.longitude.min().values - buffer
    era5_lon_max = ds_cerra_sub.longitude.max().values + buffer

    ds_era5_cropped = ds_era5.sel(
        latitude=slice(era5_lat_min, era5_lat_max),
        longitude=slice(era5_lon_min, era5_lon_max)
    )

    # 1. Interpolate
    ds_era5_on_cerra = ds_era5_cropped.interp(
        latitude=ds_cerra_sub.latitude, 
        longitude=ds_cerra_sub.longitude, 
        method='linear'
    )

    # 2. FORCE EXACT COORDINATES
    # Drop the interpolated coords to avoid floating point mismatches
    ds_era5_on_cerra = ds_era5_on_cerra.drop_vars(['latitude', 'longitude', 'x', 'y'], errors='ignore')
    
    # Assign the exact objects from CERRA
    ds_era5_on_cerra = ds_era5_on_cerra.assign_coords({
        'x': ds_cerra_sub.x,
        'y': ds_cerra_sub.y,
        'latitude': ds_cerra_sub.latitude,
        'longitude': ds_cerra_sub.longitude
    })

    return ds_era5_on_cerra, ds_era5_cropped

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================
def plot_all(ds_era5_on_cerra, ds_cerra_sub, idx, title_suffix="", save_path=None):
    cerra_vals = ds_cerra_sub['t2m'].isel(**idx)
    era5_vals = ds_era5_on_cerra['t2m'].isel(**idx)
    bias = cerra_vals - era5_vals

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    titles = [f"CERRA {title_suffix}", f"ERA5 (Interp) {title_suffix}", "Bias (CERRA - ERA5)"]
    datas = [cerra_vals, era5_vals, bias]
    cmaps = ['RdYlBu_r', 'RdYlBu_r', 'bwr']

    for ax, data, title, cmap in zip(axes, datas, titles, cmaps):
        data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                  cmap=cmap, cbar_kwargs={'shrink': 0.8})
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.close(fig)

def plot_statistics(c_mean, c_std, e_mean, e_std, title_suffix="", save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap_mean, cmap_std = 'coolwarm', 'viridis'

    var_name = None
    for v in c_mean.data_vars:
        if len(c_mean[v].dims) >= 2: 
            var_name = v
            break
            
    if var_name is None:
        return

    def get_2d_slice(ds, var):
        data = ds[var]
        if 'valid_time' in data.dims:
            data = data.isel(valid_time=0)
        return data

    c_m_plot = get_2d_slice(c_mean, var_name)
    e_m_plot = get_2d_slice(e_mean, var_name)
    c_s_plot = get_2d_slice(c_std, var_name)
    e_s_plot = get_2d_slice(e_std, var_name)
    
    # --- Plotting with Titles ---

    # 1. Top-Left: CERRA Mean
    c_m_plot.plot(ax=axes[0,0], transform=ccrs.PlateCarree(), 
                  x='longitude', y='latitude', cmap=cmap_mean)
    axes[0,0].set_title(f"CERRA Mean {title_suffix}")

    # 2. Top-Right: ERA5 Mean
    e_m_plot.plot(ax=axes[0,1], transform=ccrs.PlateCarree(), 
                  x='longitude', y='latitude', cmap=cmap_mean)
    axes[0,1].set_title(f"ERA5 Mean {title_suffix}")

    # 3. Bottom-Left: CERRA Std
    c_s_plot.plot(ax=axes[1,0], transform=ccrs.PlateCarree(), 
                  x='longitude', y='latitude', cmap=cmap_std)
    axes[1,0].set_title(f"CERRA Std {title_suffix}")

    # 4. Bottom-Right: ERA5 Std
    e_s_plot.plot(ax=axes[1,1], transform=ccrs.PlateCarree(), 
                  x='longitude', y='latitude', cmap=cmap_std)
    axes[1,1].set_title(f"ERA5 Std {title_suffix}")

    for ax in axes.flat:
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Stats plot saved: {save_path}")
    plt.close(fig)

import os
import glob
import random
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 1. FILE FINDER (Kept as requested)
# ==========================================
def find_matching_files(cerra_dir, era5_dir, years, months, cycles):
    """
    Finds pairs of CERRA and ERA5 files matching the given date and cycles.
    """
    matched_pairs = []
    for year in years:
        for month in months:
            mm = f"{int(month):02d}"

            for cycle in cycles:
                yyyy = str(year)
                # Construct filenames
                cerra_filename = f"cerra_t2m_{yyyy}{mm}_{cycle}.nc"
                era5_filename = f"erra5eda_t2m_{yyyy}{mm}_{cycle}.nc" # Note: 'erra5eda' typo in file name handled

                cerra_path = os.path.join(cerra_dir, cerra_filename)
                era5_path = os.path.join(era5_dir, era5_filename)

                if os.path.exists(cerra_path) and os.path.exists(era5_path):
                    matched_pairs.append((cerra_path, era5_path))
                    print(f"[FOUND] {cycle}: {cerra_filename}")
                else:
                    print(f"[MISSING] Cycle {cycle} incomplete.")

            return matched_pairs

# ==========================================
# 2. PROCESSING FUNCTIONS (From your notebook)
# ==========================================
def open_ds(file_path):
    ds = xr.open_dataset(file_path)
    # Normalize longitude to -180 to 180
    ds['longitude'] = ((ds.longitude + 180) % 360) - 180
    return ds

def crop_cerra(ds, lat_min, lat_max, lon_min, lon_max, grid_size=128):
    """
    Crops a square grid of grid_size from the CERRA dataset.
    """
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
        raise IndexError(
            f"OUT OF BOUNDS: Crop ends at (y:{y_idx_end}, x:{x_idx_end}), "
            f"Exceeds limits (max_y:{max_y}, max_x:{max_x})."
        )

    ds_cerra_sub = ds.isel(
        x=slice(x_idx_min, x_idx_end),
        y=slice(y_idx_min, y_idx_end)
    )

    print(f"Crop Successful: Start(y:{y_idx_min}, x:{x_idx_min}) -> End(y:{y_idx_end}, x:{x_idx_end})")
    
    nan_count = ds_cerra_sub.t2m.isnull().sum().values
    if nan_count > 0:
        print(f"WARNING: Subdomain contains {nan_count} NaNs.")
    
    return ds_cerra_sub

def era5_to_cerra(ds_era5, ds_cerra_sub):
    # Ensure ERA5 is sorted and normalized (redundant if open_ds used, but safe)
    ds_era5['longitude'] = ((ds_era5.longitude + 180) % 360) - 180
    ds_era5 = ds_era5.sortby(['latitude', 'longitude'])

    # Buffer for pre-cropping
    buffer = 2.0
    era5_lat_min = ds_cerra_sub.latitude.min().values - buffer
    era5_lat_max = ds_cerra_sub.latitude.max().values + buffer
    era5_lon_min = ds_cerra_sub.longitude.min().values - buffer
    era5_lon_max = ds_cerra_sub.longitude.max().values + buffer

    # Pre-crop ERA5
    ds_era5_cropped = ds_era5.sel(
        latitude=slice(era5_lat_min, era5_lat_max),
        longitude=slice(era5_lon_min, era5_lon_max)
    )

    # Interpolate
    ds_era5_on_cerra = ds_era5_cropped.interp(
        latitude=ds_cerra_sub.latitude, 
        longitude=ds_cerra_sub.longitude, 
        method='linear'
    )

    # Assign coordinates for subtraction
    ds_era5_on_cerra = ds_era5_on_cerra.assign_coords(x=ds_cerra_sub.x, y=ds_cerra_sub.y)
    return ds_era5_on_cerra, ds_era5_cropped
def plot_all(ds_era5_on_cerra, ds_cerra_sub, idx, title_suffix="", save_path=None):
    """
    Plots CERRA, ERA5, and Bias side-by-side for a specific index.
    Optionally saves the figure to disk.
    """
    # Select data based on index (random member/time)
    cerra_vals = ds_cerra_sub['t2m'].isel(**idx)
    era5_vals = ds_era5_on_cerra['t2m'].isel(**idx)
    bias = cerra_vals - era5_vals

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    titles = [f"CERRA {title_suffix}", f"ERA5 (Interp) {title_suffix}", "Bias (CERRA - ERA5)"]
    datas = [cerra_vals, era5_vals, bias]
    cmaps = ['RdYlBu_r', 'RdYlBu_r', 'bwr']

    for ax, data, title, cmap in zip(axes, datas, titles, cmaps):
        im = data.plot(
            ax=ax, 
            x='longitude', y='latitude', 
            transform=ccrs.PlateCarree(),
            cmap=cmap, 
            add_colorbar=True, 
            cbar_kwargs={'shrink': 0.8}
        )
        
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_title(title)
        
        # Zoom to crop extent
        ax.set_extent([
            ds_cerra_sub.longitude.min(), ds_cerra_sub.longitude.max(), 
            ds_cerra_sub.latitude.min(), ds_cerra_sub.latitude.max()
        ], crs=ccrs.PlateCarree())

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # plt.show() # Optional: Comment out if running in batch mode on a server without display
    plt.close(fig) # Close to free memory
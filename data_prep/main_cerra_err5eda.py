# -*- coding: utf-8 -*-
"""
Author: Swapan Mallick, SMHI
Date: 15 December, 2025
#
Main driver program for CERRA vs ERA5-EDA comparison.

Responsibilities:
- Control overall workflow
- Define target domain
- Call NetCDF processing utilities
- Call plotting routines
- Generate comparisons_final

"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

def analyze_datasets(HH):
    ds_cerra = xr.open_dataset(f"cerra_{HH}.nc", decode_times=False)
    if float(ds_cerra["longitude"].max()) > 180:
        ds_cerra["longitude"] = ((ds_cerra["longitude"] + 180) % 360) - 180
    
    cerra_lat_min, cerra_lat_max = float(ds_cerra.latitude.min()), float(ds_cerra.latitude.max())
    cerra_lon_min, cerra_lon_max = float(ds_cerra.longitude.min()), float(ds_cerra.longitude.max())
    
    print(f"CERRA: Lat {cerra_lat_min:.2f} to {cerra_lat_max:.2f}, Lon {cerra_lon_min:.2f} to {cerra_lon_max:.2f}")
    print(f"CERRA grid: {ds_cerra.latitude.shape}")
    
    ds_era = xr.open_dataset(f"era5eda_{HH}.nc", decode_times=False)
    if float(ds_era["longitude"].max()) > 180:
        ds_era["longitude"] = ((ds_era["longitude"] + 180) % 360) - 180
    
    era_lat_min, era_lat_max = float(ds_era.latitude.min()), float(ds_era.latitude.max())
    era_lon_min, era_lon_max = float(ds_era.longitude.min()), float(ds_era.longitude.max())
    
    print(f"ERA5: Lat {era_lat_min:.2f} to {era_lat_max:.2f}, Lon {era_lon_min:.2f} to {era_lon_max:.2f}")
    print(f"ERA5 grid: {ds_era.latitude.shape}")
    
    ds_cerra.close()
    ds_era.close()
    
    return (cerra_lat_min, cerra_lat_max, cerra_lon_min, cerra_lon_max,
            era_lat_min, era_lat_max, era_lon_min, era_lon_max)

def extract_cerra_domain_points(cerra_file, desired_lat_min=35.0, desired_lat_max=72.0,
                             desired_lon_min=-20.0, desired_lon_max=40.0):
    ds = xr.open_dataset(cerra_file, decode_times=False)
    
    if float(ds["longitude"].max()) > 180:
        ds["longitude"] = ((ds["longitude"] + 180) % 360) - 180
    
    lat_2d = ds["latitude"].values
    lon_2d = ds["longitude"].values
    
    lat_mask = (lat_2d >= desired_lat_min) & (lat_2d <= desired_lat_max)
    lon_mask = (lon_2d >= desired_lon_min) & (lon_2d <= desired_lon_max)
    domain_mask = lat_mask & lon_mask
    
    if not np.any(domain_mask):
        domain_mask = np.ones_like(lat_2d, dtype=bool)
    
    lat_points = lat_2d[domain_mask]
    lon_points = lon_2d[domain_mask]
    
    actual_lat_min, actual_lat_max = float(lat_points.min()), float(lat_points.max())
    actual_lon_min, actual_lon_max = float(lon_points.min()), float(lon_points.max())
    
    print(f"CERRA points: {len(lat_points)}")
    print(f"CERRA domain: Lat {actual_lat_min:.2f} to {actual_lat_max:.2f}, Lon {actual_lon_min:.2f} to {actual_lon_max:.2f}")
    
    ds.close()
    
    return lat_points, lon_points, domain_mask, actual_lat_min, actual_lat_max, actual_lon_min, actual_lon_max

def create_extended_era5_grid(era_file, target_lat_min, target_lat_max, 
                             target_lon_min, target_lon_max, resolution=0.25):
    ds = xr.open_dataset(era_file, decode_times=False)
    
    if float(ds["longitude"].max()) > 180:
        ds["longitude"] = ((ds["longitude"] + 180) % 360) - 180
    
    extended_lats = np.arange(target_lat_min, target_lat_max + resolution, resolution)
    extended_lons = np.arange(target_lon_min, target_lon_max + resolution, resolution)
    
    ds.close()
    
    return extended_lats, extended_lons

def interpolate_era5_to_extended_grid(era_data, extended_lats, extended_lons):
    era_lats = era_data.latitude.values
    era_lons = era_data.longitude.values
    era_values = era_data.values
    
    ext_lon_grid, ext_lat_grid = np.meshgrid(extended_lons, extended_lats)
    ext_lats_flat = ext_lat_grid.flatten()
    ext_lons_flat = ext_lon_grid.flatten()
    
    src_lon_grid, src_lat_grid = np.meshgrid(era_lons, era_lats)
    src_lats_flat = src_lat_grid.flatten()
    src_lons_flat = src_lon_grid.flatten()
    src_vals_flat = era_values.flatten()
    
    valid_mask = ~np.isnan(src_vals_flat)
    if not np.any(valid_mask):
        raise ValueError("No valid ERA5 data")
    
    src_lats_valid = src_lats_flat[valid_mask]
    src_lons_valid = src_lons_flat[valid_mask]
    src_vals_valid = src_vals_flat[valid_mask]
    
    ext_vals_flat = griddata(
        (src_lons_valid, src_lats_valid),
        src_vals_valid,
        (ext_lons_flat, ext_lats_flat),
        method='linear',
        fill_value=np.nan
    )
    
    ext_vals = ext_vals_flat.reshape(len(extended_lats), len(extended_lons))
    
    ext_data = xr.DataArray(
        ext_vals,
        coords={"latitude": extended_lats, "longitude": extended_lons},
        dims=("latitude", "longitude")
    )
    
    return ext_data

def process_cerra_points(cerra_file, variable, lat_points, lon_points, domain_mask,
                      output_dir, actual_domain, HH):
    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    ds = xr.open_dataset(cerra_file, decode_times=False)
    
    if float(ds["longitude"].max()) > 180:
        ds["longitude"] = ((ds["longitude"] + 180) % 360) - 180
    
    data_var = ds[variable]
    time_vals = pd.to_datetime(ds["valid_time"].values, unit="s")
    
    all_means = []
    all_stds = []
    all_timestamps = []
    
    for i, dt in enumerate(time_vals):
        try:
            day_data = data_var.isel(valid_time=i)
            
            ens_mean = day_data.mean(dim="number")
            ens_std = day_data.std(dim="number")
            
            mean_vals = ens_mean.values[domain_mask]
            std_vals = ens_std.values[domain_mask]
            
            if np.all(np.isnan(mean_vals)) or np.all(np.isnan(std_vals)):
                continue
            
            mean_data = xr.DataArray(
                mean_vals,
                coords={
                    "latitude": (["points"], lat_points),
                    "longitude": (["points"], lon_points)
                },
                dims=["points"]
            )
            
            std_data = xr.DataArray(
                std_vals,
                coords={
                    "latitude": (["points"], lat_points),
                    "longitude": (["points"], lon_points)
                },
                dims=["points"]
            )
            
            timestamp = dt.strftime("%Y%m%d") + HH
            all_means.append(mean_data)
            all_stds.append(std_data)
            all_timestamps.append(timestamp)
            
            plot_point_data(
                f"cerra Mean - {timestamp}",
                mean_data,
                outdir / f"cerra_mean_{timestamp}.png",
                timestamp,
                actual_domain,
                is_std=False
            )
            
            plot_point_data(
                f"cerra SD - {timestamp}",
                std_data,
                outdir / f"cerra_SD_{timestamp}.png",
                timestamp,
                actual_domain,
                is_std=True
            )
            
            save_point_netcdf(
                mean_data, std_data, "cerra", variable, timestamp,
                output_dir=f"netcdf_cerra"
            )
            
            print(f"cerra {timestamp}")
            
        except Exception:
            continue
    
    ds.close()
    return all_means, all_stds, all_timestamps

def process_era5eda_extended(era_file, variable, lat_points, lon_points,
                         extended_lats, extended_lons, actual_domain,
                         output_dir, HH):
    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    ds = xr.open_dataset(era_file, decode_times=False)
    
    if float(ds["longitude"].max()) > 180:
        ds["longitude"] = ((ds["longitude"] + 180) % 360) - 180
    
    data_var = ds[variable]
    time_vals = pd.to_datetime(ds["valid_time"].values, unit="s")
    
    all_means = []
    all_stds = []
    all_timestamps = []
    
    for i, dt in enumerate(time_vals):
        try:
            day_data = data_var.isel(valid_time=i)
            
            ens_mean = day_data.mean(dim="number")
            ens_std = day_data.std(dim="number")
            
            mean_extended = interpolate_era5_to_extended_grid(ens_mean, extended_lats, extended_lons)
            std_extended = interpolate_era5_to_extended_grid(ens_std, extended_lats, extended_lons)
            
            mean_at_points = interpolate_to_points(mean_extended, lat_points, lon_points)
            std_at_points = interpolate_to_points(std_extended, lat_points, lon_points)
            
            timestamp = dt.strftime("%Y%m%d") + HH
            all_means.append(mean_at_points)
            all_stds.append(std_at_points)
            all_timestamps.append(timestamp)
            
            plot_point_data(
                f"era5eda Mean - {timestamp}",
                mean_at_points,
                outdir / f"era5eda_mean_{timestamp}.png",
                timestamp,
                actual_domain,
                is_std=False
            )
            
            plot_point_data(
                f"era5eda SD - {timestamp}",
                std_at_points,
                outdir / f"era5eda_sd_{timestamp}.png",
                timestamp,
                actual_domain,
                is_std=True
            )
            
            save_point_netcdf(
                mean_at_points, std_at_points, "era5eda", variable, timestamp,
                output_dir=f"netcdf_era5eda"
            )
            
            save_extended_grid_netcdf(
                mean_extended, std_extended, "era5_extended", variable, timestamp,
                output_dir=f"netcdf_final"
            )
            
            print(f"era5eda {timestamp}")
            
        except Exception:
            continue
    
    ds.close()
    return all_means, all_stds, all_timestamps

def interpolate_to_points(grid_data, target_lats, target_lons, method='linear'):
    grid_lats = grid_data.latitude.values
    grid_lons = grid_data.longitude.values
    grid_vals = grid_data.values
    
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lons, grid_lats)
    
    src_lats_flat = grid_lat_mesh.flatten()
    src_lons_flat = grid_lon_mesh.flatten()
    src_vals_flat = grid_vals.flatten()
    
    valid_mask = ~np.isnan(src_vals_flat)
    if not np.any(valid_mask):
        raise ValueError("No valid grid data")
    
    src_lats_valid = src_lats_flat[valid_mask]
    src_lons_valid = src_lons_flat[valid_mask]
    src_vals_valid = src_vals_flat[valid_mask]
    
    target_vals = griddata(
        (src_lons_valid, src_lats_valid),
        src_vals_valid,
        (target_lons, target_lats),
        method=method,
        fill_value=np.nan
    )
    
    point_data = xr.DataArray(
        target_vals,
        coords={
            "latitude": (["points"], target_lats),
            "longitude": (["points"], target_lons)
        },
        dims=["points"]
    )
    
    return point_data

def plot_point_data(title, data, filename, timestamp, domain, is_std=False):
    lat_min, lat_max, lon_min, lon_max = domain
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                     alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    lats = data.latitude.values
    lons = data.longitude.values
    values = data.values
    
    valid_mask = ~np.isnan(values)
    lats_valid = lats[valid_mask]
    lons_valid = lons[valid_mask]
    values_valid = values[valid_mask]
    
    if len(values_valid) == 0:
        plt.close(fig)
        return
    
    if is_std:
        cmap = 'jet'
        vmin, vmax = 0, 3
        label = 'SD (K)'
    else:
        cmap = 'jet'
        vmin, vmax = 200, 300
        label = 'Temperature (K)'
    
    scatter = ax.scatter(lons_valid, lats_valid, c=values_valid,
                        cmap=cmap, s=5, alpha=0.7,
                        transform=ccrs.PlateCarree(),
                        vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical',
                       pad=0.03, aspect=30, shrink=0.8)
    cbar.set_label(label, rotation=270, labelpad=20)
    
    plt.title(f"{title}\n{timestamp}", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)

def save_point_netcdf(mean_data, std_data, prefix, variable, timestamp,
                     output_dir="netcdf_output"):
    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    ds = xr.Dataset({
        f'{variable}_mean': mean_data,
        f'{variable}_std': std_data
    })
    
    output_file = outdir / f"{prefix}_{variable}_points_{timestamp}.nc"
    ds.to_netcdf(output_file)
    
    return output_file

def save_extended_grid_netcdf(mean_data, std_data, prefix, variable, timestamp,
                             output_dir="netcdf_output"):
    outdir = Path(output_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    
    ds = xr.Dataset({
        f'{variable}_mean': mean_data,
        f'{variable}_std': std_data
    })
    
    output_file = outdir / f"{prefix}_{variable}_extended_{timestamp}.nc"
    ds.to_netcdf(output_file)
    
    return output_file

def create_comparisons(means_cerra, stds_cerra, timestamps_cerra,
                      means_era, stds_era, timestamps_era,
                      domain, output_dir="comparisons_final", HH=""):
    common_timestamps = sorted(set(timestamps_cerra) & set(timestamps_era))
    
    if not common_timestamps:
        return
    
    comp_dir = Path(output_dir)
    comp_dir.mkdir(exist_ok=True, parents=True)
    
    lat_min, lat_max, lon_min, lon_max = domain
    
    for timestamp in common_timestamps:
        try:
            idx_cerra = timestamps_cerra.index(timestamp)
            idx_era = timestamps_era.index(timestamp)
            
            mean_cerra = means_cerra[idx_cerra]
            std_cerra = stds_cerra[idx_cerra]
            mean_era = means_era[idx_era]
            std_era = stds_era[idx_era]
            
            ds_comp = xr.Dataset({
                'cerra_mean': mean_cerra,
                'cerra_std': std_cerra,
                'era5_mean': mean_era,
                'era5_std': std_era,
                'mean_diff': mean_cerra - mean_era,
                'std_diff': std_cerra - std_era
            })
            
            output_file = comp_dir / f"comparison_{timestamp}.nc"
            ds_comp.to_netcdf(output_file)
            
            plot_comparison(
                mean_cerra, mean_era, timestamp, comp_dir, domain
            )
            
            print(f"comparison {timestamp}")
            
        except Exception:
            continue

def plot_comparison(mean_cerra, mean_era, timestamp, output_dir, domain):
    lat_min, lat_max, lon_min, lon_max = domain
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    titles = ['cerra', 'era5eda', 'difference']
    data_list = [mean_cerra, mean_era, mean_cerra - mean_era]
    
    for idx, (ax, title, data) in enumerate(zip(axes, titles, data_list)):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        lats = data.latitude.values
        lons = data.longitude.values
        values = data.values
        
        valid_mask = ~np.isnan(values)
        lats_valid = lats[valid_mask]
        lons_valid = lons[valid_mask]
        values_valid = values[valid_mask]
        
        if len(values_valid) == 0:
            continue
        
        if idx == 2:
            max_diff = max(abs(values_valid.min()), abs(values_valid.max()))
            scatter = ax.scatter(lons_valid, lats_valid, c=values_valid,
                               cmap='jet', s=5, alpha=0.7,
                               transform=ccrs.PlateCarree(),
                               vmin=-max_diff, vmax=max_diff)
        else:
            scatter = ax.scatter(lons_valid, lats_valid, c=values_valid,
                               cmap='jet', s=5, alpha=0.7,
                               transform=ccrs.PlateCarree(),
                               vmin=200, vmax=300)
        
        ax.set_title(title, fontsize=11)
        plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05)
    
    plt.suptitle(f'CERRA Comparison - {timestamp}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / f"comparison_{timestamp}.png"
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close(fig)

def main():
    if len(sys.argv) < 2:
        print("SWAPAN Usage: python main_cerra_err5eda.py HH; Hour")
        sys.exit(1)
    
    HH = sys.argv[1]
    cerra_file = f"cerra_{HH}.nc"
    era_file = f"era5eda_{HH}.nc"
    
    (cerra_lat_min, cerra_lat_max, cerra_lon_min, cerra_lon_max,
     era_lat_min, era_lat_max, era_lon_min, era_lon_max) = analyze_datasets(HH)
    
    desired_domain = (35.0, 72.0, -20.0, 40.0)
    lat_points, lon_points, domain_mask, actual_lat_min, actual_lat_max, actual_lon_min, actual_lon_max = extract_cerra_domain_points(
        cerra_file, *desired_domain
    )
    
    actual_domain = (actual_lat_min, actual_lat_max, actual_lon_min, actual_lon_max)
    
    extended_lats, extended_lons = create_extended_era5_grid(
        era_file,
        target_lat_min=min(actual_lat_min, 35.0),
        target_lat_max=max(actual_lat_max, 72.0),
        target_lon_min=min(actual_lon_min, -20.0),
        target_lon_max=max(actual_lon_max, 40.0),
        resolution=0.25
    )
    # Output folder name SM 
    means_cerra, stds_cerra, timestamps_cerra = process_cerra_points(
        cerra_file, "t2m", lat_points, lon_points, domain_mask,
        "plots_cerra", actual_domain, HH
    )
    
    means_era, stds_era, timestamps_era = process_era5eda_extended(
        era_file, "t2m", lat_points, lon_points,
        extended_lats, extended_lons, actual_domain,
        "plots_era5eda", HH
    )
    
    if means_cerra and means_era:
        create_comparisons(
            means_cerra, stds_cerra, timestamps_cerra,
            means_era, stds_era, timestamps_era,
            actual_domain, "comparisons_final", HH
        )

if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams['figure.max_open_warning'] = 0
    # 
    main()

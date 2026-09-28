#!/usr/bin/env python3

import os
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

# ==========================================================
# ARGUMENTS
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Plot variables from Zarr store")
    parser.add_argument("--zarr-path", required=True, help="Path to Zarr store")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--variables", nargs="+", required=True, help="Variables to plot")
    parser.add_argument("--lat-min", type=float, required=True, help="Minimum latitude")
    parser.add_argument("--lat-max", type=float, required=True, help="Maximum latitude")
    parser.add_argument("--lon-min", type=float, required=True, help="Minimum longitude")
    parser.add_argument("--lon-max", type=float, required=True, help="Maximum longitude")
    parser.add_argument("--plan-view", action="store_true", help="Generate plan view plots without axes")
    parser.add_argument("--full-bleed", action="store_true", help="Generate full-bleed images without any axes, colorbar, or labels")
    return parser.parse_args()


# ==========================================================
# SUBSET DOMAIN FOR 2D LAT/LON
# ==========================================================
def subset_domain_2d(da, lat2d, lon2d,
                     lat_min, lat_max, lon_min, lon_max):

    # Handle longitude if needed
    lon2d_adjusted = lon2d.copy()
    lon_min_adjusted = lon_min
    lon_max_adjusted = lon_max
    
    if float(lon2d.max()) > 180:
        lon_min_adjusted = lon_min % 360
        lon_max_adjusted = lon_max % 360
        # Also adjust longitude values if they're in 0-360 range
        lon2d_adjusted = lon2d.where(lon2d <= 180, lon2d - 360)

    mask = (
        (lat2d >= lat_min) & (lat2d <= lat_max) &
        (lon2d_adjusted >= lon_min_adjusted) & (lon2d_adjusted <= lon_max_adjusted)
    )

    da_masked = da.where(mask)

    return da_masked, lon2d_adjusted


# ==========================================================
# PLOT FUNCTION (Standard with axes)
# ==========================================================
def plot_variable(ds, var_name, time_idx, output_dir,
                  lat_min, lat_max, lon_min, lon_max):

    da = ds[var_name]

    if "valid_time" in da.dims:
        da = da.isel(valid_time=time_idx)
    if "number" in da.dims:
        da = da.isel(number=0)

    lat2d = ds["latitude"]
    lon2d = ds["longitude"]

    da, lon2d_adjusted = subset_domain_2d(da, lat2d, lon2d,
                                         lat_min, lat_max,
                                         lon_min, lon_max)

    if np.isnan(da.values).all():
        print(f"Skipping {var_name} — no data in selected domain")
        return

    # -------------------------------------------------
    # DEFINE FIXED COLORBAR SETTINGS
    # -------------------------------------------------
    if "std" in var_name.lower():
        levels = np.arange(0, 2.1, 0.2)
        cmap = plt.get_cmap("BuPu", len(levels)-1)
        norm = BoundaryNorm(levels, cmap.N)
    elif "mean" in var_name.lower():
        levels = np.arange(242, 300, 2)
        cmap = plt.get_cmap("jet", len(levels)-1)
        norm = BoundaryNorm(levels, cmap.N)
    else:
        levels = None
        cmap = "viridis"
        norm = None

    # -------------------------------------------------
    # TIME STRING
    # -------------------------------------------------
    if "valid_time" in ds.coords:
        time_val = str(ds.valid_time.isel(valid_time=time_idx).values)
        time_str = time_val[:16].replace(":", "").replace("-", "").replace("T", "_")
    else:
        time_val = f"time_{time_idx}"
        time_str = f"time_{time_idx:03d}"

    filename = f"{var_name}_{time_str}.png"
    save_path = os.path.join(output_dir, filename)

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon2d_adjusted,
        lat2d,
        da,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        norm=norm
    )

    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    gl = ax.gridlines(draw_labels=True,
                      crs=ccrs.PlateCarree(),
                      linestyle='--',
                      linewidth=0.5,
                      alpha=0.6)

    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f"{var_name} | {time_val}")

    cbar = plt.colorbar(mesh,
                        ax=ax,
                        shrink=0.7)
    if levels is not None:
        cbar.set_ticks(levels)

    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {save_path}")


# ==========================================================
# PLAN VIEW FUNCTION (No axes but with colorbar)
# ==========================================================
def plot_variable_plan(ds, var_name, time_idx, output_dir,
                       lat_min, lat_max, lon_min, lon_max):

    da = ds[var_name]

    if "valid_time" in da.dims:
        da = da.isel(valid_time=time_idx)
    if "number" in da.dims:
        da = da.isel(number=0)

    lat2d = ds["latitude"]
    lon2d = ds["longitude"]

    da, lon2d_adjusted = subset_domain_2d(da, lat2d, lon2d,
                                         lat_min, lat_max,
                                         lon_min, lon_max)

    if np.isnan(da.values).all():
        print(f"Skipping {var_name} plan view — no data in selected domain")
        return

    # -------------------------------------------------
    # DEFINE FIXED COLORBAR SETTINGS
    # -------------------------------------------------
    if "std" in var_name.lower():
        levels = np.arange(0, 2.1, 0.2)
        cmap = plt.get_cmap("BuPu", len(levels)-1)
        norm = BoundaryNorm(levels, cmap.N)
    elif "mean" in var_name.lower():
        levels = np.arange(242, 300, 2)
        cmap = plt.get_cmap("jet", len(levels)-1)
        norm = BoundaryNorm(levels, cmap.N)
    else:
        levels = None
        cmap = "viridis"
        norm = None

    # -------------------------------------------------
    # TIME STRING
    # -------------------------------------------------
    if "valid_time" in ds.coords:
        time_val = str(ds.valid_time.isel(valid_time=time_idx).values)
        time_str = time_val[:16].replace(":", "").replace("-", "").replace("T", "_")
    else:
        time_val = f"time_{time_idx}"
        time_str = f"time_{time_idx:03d}"

    filename = f"plan_{var_name}_{time_str}.png"
    save_path = os.path.join(output_dir, filename)

    # -------------------------------------------------
    # PLOT (without axes but with colorbar)
    # -------------------------------------------------
    #fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon2d_adjusted,
        lat2d,
        da,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        norm=norm
    )

    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # Remove all axes elements for plan view
    ax.set_axis_off()

    # Add colorbar
    cbar = plt.colorbar(mesh,
                        ax=ax,
                        shrink=0.7,
                        orientation='horizontal',
                        pad=0.05)
    if levels is not None:
        cbar.set_ticks(levels)
    cbar.set_label(var_name)

    plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved plan view: {save_path}")


# ==========================================================
# FULL BLEED FUNCTION (No axes, no colorbar, no labels - fills entire image)
# ==========================================================
def plot_variable_full_bleed(ds, var_name, time_idx, output_dir,
                             lat_min, lat_max, lon_min, lon_max):

    da = ds[var_name]

    if "valid_time" in da.dims:
        da = da.isel(valid_time=time_idx)
    if "number" in da.dims:
        da = da.isel(number=0)

    lat2d = ds["latitude"]
    lon2d = ds["longitude"]

    da, lon2d_adjusted = subset_domain_2d(da, lat2d, lon2d,
                                         lat_min, lat_max,
                                         lon_min, lon_max)

    if np.isnan(da.values).all():
        print(f"Skipping {var_name} full bleed — no data in selected domain")
        return

    # -------------------------------------------------
    # DEFINE FIXED COLORBAR SETTINGS (for colormap only)
    # -------------------------------------------------
    if "std" in var_name.lower():
        levels = np.arange(0, 2.1, 0.2)
        cmap = plt.get_cmap("BuPu", len(levels)-1)
        norm = BoundaryNorm(levels, cmap.N)
    elif "mean" in var_name.lower():
        levels = np.arange(242, 300, 2)
        cmap = plt.get_cmap("jet", len(levels)-1)
        norm = BoundaryNorm(levels, cmap.N)
    else:
        levels = None
        cmap = "viridis"
        norm = None

    # -------------------------------------------------
    # TIME STRING
    # -------------------------------------------------
    if "valid_time" in ds.coords:
        time_val = str(ds.valid_time.isel(valid_time=time_idx).values)
        time_str = time_val[:16].replace(":", "").replace("-", "").replace("T", "_")
    else:
        time_val = f"time_{time_idx}"
        time_str = f"time_{time_idx:03d}"

    filename = f"fullbleed_{var_name}_{time_str}.png"
    save_path = os.path.join(output_dir, filename)

    # -------------------------------------------------
    # FULL BLEED PLOT - SIMPLIFIED APPROACH
    # -------------------------------------------------
    
    # Create figure with specific size and no margins
    fig = plt.figure(figsize=(12, 10), frameon=False)
    
    # Add axes that fill the entire figure
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    
    # Plot the data
    mesh = ax.pcolormesh(
        lon2d_adjusted,
        lat2d,
        da,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        norm=norm
    )
    
    # Set extent to the domain
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())
    
    # Add coastlines (optional - remove if you want only data)
    ax.coastlines(linewidth=0.8, color='black', resolution='50m')
    
    # Add country borders (optional)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
    
    # Remove ALL axes elements
    ax.set_axis_off()
    
    # Ensure no extra whitespace
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with high DPI and no padding
    plt.savefig(save_path, 
                dpi=100, 
                bbox_inches='tight', 
                pad_inches=0,
                facecolor='white',
                edgecolor='none',
                transparent=False)
    
    plt.close(fig)
    
    # Verify file was created
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"Saved full bleed image: {save_path} ({file_size/1024:.1f} KB)")
    else:
        print(f"ERROR: Failed to create {save_path}")
        
    # Flush to ensure file is written
    plt.close('all')


# ==========================================================
# MAIN
# ==========================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening Zarr store: {args.zarr_path}")
    ds = xr.open_dataset(args.zarr_path, engine="zarr")

    print("Latitude range:", float(ds.latitude.min()), float(ds.latitude.max()))
    print("Longitude range:", float(ds.longitude.min()), float(ds.longitude.max()))

    n_times = ds.sizes["valid_time"] if "valid_time" in ds.dims else 1

    for var in args.variables:
        if var not in ds.data_vars:
            print(f"Skipping {var} (not found)")
            continue
        print(f"Processing variable: {var}")
        for t in range(n_times):
            # Always generate standard plot
            plot_variable(ds, var, t, args.output_dir,
                          args.lat_min, args.lat_max,
                          args.lon_min, args.lon_max)
            
            # Generate plan view if requested
            if args.plan_view:
                plot_variable_plan(ds, var, t, args.output_dir,
                                   args.lat_min, args.lat_max,
                                   args.lon_min, args.lon_max)
            
            # Generate full bleed image if requested
            if args.full_bleed:
                print(f"  Generating full bleed for {var} at time {t}")
                #plot_variable_full_bleed(ds, var, t, args.output_dir,
                #                         args.lat_min, args.lat_max,
                #                         args.lon_min, args.lon_max)
    
    ds.close()
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import re
import glob
import argparse
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.ndimage import zoom
from matplotlib.colors import BoundaryNorm, ListedColormap


# ==========================================================
# ARGUMENTS
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CERRA/Zarr with DIFFUSION files at 12 UTC after regridding DIFFUSION to CERRA shape"
    )

    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--npy-pattern",
        default=(
            "/cfs/klemming/scratch/y/yarongc/SDM_seasonalStrongCond/"
            "cfg_1/January/*/best_overall/best_t2m_cerra_mean*.npy"
        ),
    )

    parser.add_argument("--variable", default="t2m_cerra_mean")

    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)

    return parser.parse_args()


# ==========================================================
# DIFFUSION DATE EXTRACTOR
# Each DIFFUSION file is assumed valid at 12 UTC
# ==========================================================
def extract_time_from_npy_path(filepath):
    match = re.search(r"/(\d{8})/best_overall/", filepath)

    if match is None:
        raise ValueError(f"Could not extract date from path:\n{filepath}")

    date_str = match.group(1)
    return pd.to_datetime(date_str, format="%Y%m%d") + pd.Timedelta(hours=12)


# ==========================================================
# DOMAIN SUBSET
# ==========================================================
def subset_domain_2d(da, lat2d, lon2d, lat_min, lat_max, lon_min, lon_max):
    lon2d_adj = lon2d.copy()

    if float(lon2d.max()) > 180:
        lon2d_adj = lon2d.where(lon2d <= 180, lon2d - 360)

    mask = (
        (lat2d >= lat_min)
        & (lat2d <= lat_max)
        & (lon2d_adj >= lon_min)
        & (lon2d_adj <= lon_max)
    )

    return da.where(mask), lon2d_adj


# ==========================================================
# REGRID DIFFUSION TO CERRA SHAPE
# ==========================================================
def regrid_npy_to_cerra_shape(npy_data, target_shape):
    npy_data = np.asarray(npy_data)

    if npy_data.shape == target_shape:
        return npy_data.astype(np.float32)

    zoom_y = target_shape[0] / npy_data.shape[0]
    zoom_x = target_shape[1] / npy_data.shape[1]

    regridded = zoom(
        npy_data,
        zoom=(zoom_y, zoom_x),
        order=1,
        mode="nearest",
    )

    if regridded.shape != target_shape:
        regridded = regridded[: target_shape[0], : target_shape[1]]

    return regridded.astype(np.float32)


# ==========================================================
# CREATE NETCDF FROM ORIGINAL DIFFUSION FILES
# ==========================================================
def create_npy_netcdf(args):
    files = sorted(glob.glob(args.npy_pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No DIFFUSION files found:\n{args.npy_pattern}")

    print(f"Found {len(files)} DIFFUSION files")

    npy_output_dir = os.path.join(args.output_dir, "npy_to_netcdf")
    copy_dir = os.path.join(npy_output_dir, "copied_npy_files")
    output_nc = os.path.join(npy_output_dir, "best_t2m_cerra_mean_12UTC_original_grid.nc")

    os.makedirs(npy_output_dir, exist_ok=True)
    os.makedirs(copy_dir, exist_ok=True)

    data_list = []
    time_list = []

    for f in files:
        print(f"Reading DIFFUSION: {f}")

        data = np.load(f)
        data = np.squeeze(data)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array but got {data.shape} for:\n{f}")

        data_list.append(data.astype(np.float32))
        time_list.append(extract_time_from_npy_path(f))

        shutil.copy2(f, os.path.join(copy_dir, os.path.basename(f)))

    data_stack = np.stack(data_list, axis=0)

    ntime, ny, nx = data_stack.shape

    ds_npy = xr.Dataset(
        {
            args.variable: (
                ("time", "y", "x"),
                data_stack,
            )
        },
        coords={
            "time": time_list,
            "y": np.arange(ny),
            "x": np.arange(nx),
        },
    )

    ds_npy[args.variable].attrs["long_name"] = "Best t2m CERRA mean from DIFFUSION files"
    ds_npy[args.variable].attrs["units"] = "K"
    ds_npy.attrs["description"] = "Original DIFFUSION files combined into one NetCDF; valid at 12 UTC"

    encoding = {
        args.variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }

    ds_npy.to_netcdf(output_nc, encoding=encoding)

    print(f"Saved original-grid DIFFUSION NetCDF:\n{output_nc}")

    return ds_npy


# ==========================================================
# DIFFERENCE COLORMAP
# White/no color from -2 to +2 K
# Range: -10 to +10 K
# ==========================================================
def get_difference_cmap_norm():
    levels = [-10, -8, -6, -4, -2, 2, 4, 6, 8, 10]

    colors = [
        "#053061",
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#ffffff",
        "#f4a582",
        "#d6604d",
        "#b2182b",
        "#67001f",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)

    return cmap, norm, levels


# ==========================================================
# DIFFERENCE PLOT: CERRA - REGRIDDED DIFFUSION
# ==========================================================
def plot_cerra_npy_difference(
    cerra_da,
    npy_regridded,
    lat2d,
    lon2d_adj,
    args,
    valid_time,
):
    diff = cerra_da.values - npy_regridded

    cmap, norm, levels = get_difference_cmap_norm()

    time_str = valid_time.strftime("%Y%m%d_%H%M")

    out_file = os.path.join(
        args.output_dir,
        f"DIFF_CERRA_minus_REGRIDDED_DIFFUSION_{args.variable}_{time_str}.png",
    )

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon2d_adj,
        lat2d,
        diff,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    ax.set_extent(
        [args.lon_min, args.lon_max, args.lat_min, args.lat_max],
        crs=ccrs.PlateCarree(),
    )

    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        shrink=0.75,
        extend="both",
    )
    cbar.set_label("CERRA - DIFFUSION (K)")
    cbar.set_ticks(levels)

    plt.title(f"CERRA -  DIFFUSION | {valid_time.strftime('%Y-%m-%d %H:%M UTC')}")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved difference plot: {out_file}")


# ==========================================================
# TWO-PANEL TEMPERATURE PLOT: CERRA AND REGRIDDED DIFFUSION
# ==========================================================
def plot_cerra_npy_two_panel(
    cerra_da,
    npy_regridded,
    lat2d,
    lon2d_adj,
    args,
    valid_time,
):
    levels = np.arange(242, 300, 2)
    cmap = plt.get_cmap("jet", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)

    time_str = valid_time.strftime("%Y%m%d_%H%M")

    out_file = os.path.join(
        args.output_dir,
        f"TWO_PANEL_CERRA_REGRIDDED_DIFFUSION_{args.variable}_{time_str}.png",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    titles = [
        f"CERRA | {valid_time.strftime('%Y-%m-%d %H:%M UTC')}",
        f" DIFFUSION | {valid_time.strftime('%Y-%m-%d %H:%M UTC')}",
    ]

    data_list = [cerra_da.values, npy_regridded]

    for ax, data, title in zip(axes, data_list, titles):
        mesh = ax.pcolormesh(
            lon2d_adj,
            lat2d,
            data,
            cmap=cmap,
            norm=norm,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )

        ax.set_extent(
            [args.lon_min, args.lon_max, args.lat_min, args.lat_max],
            crs=ccrs.PlateCarree(),
        )

        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)

        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(title, fontsize=14)

    cbar = fig.colorbar(
        mesh,
        ax=axes.tolist(),
        orientation="vertical",
        shrink=0.75,
        pad=0.03,
    )

    cbar.set_label("T2M (K)")
    cbar.set_ticks(levels)

    plt.suptitle(
        f"CERRA and  DIFFUSION temperature comparison | {valid_time.strftime('%Y-%m-%d 12 UTC')}",
        fontsize=16,
    )

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved two-panel plot: {out_file}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ds_npy = create_npy_netcdf(args)

    print(f"Opening Zarr: {args.zarr_path}")
    ds_cerra = xr.open_dataset(args.zarr_path, engine="zarr")

    if args.variable not in ds_cerra:
        raise ValueError(f"Variable '{args.variable}' not found in Zarr file")

    time_dim = "valid_time" if "valid_time" in ds_cerra[args.variable].dims else "time"

    lat2d = ds_cerra["latitude"]
    lon2d = ds_cerra["longitude"]

    npy_times = pd.to_datetime(ds_npy["time"].values)

    for t in range(ds_cerra.sizes[time_dim]):

        time_val = ds_cerra[time_dim].isel({time_dim: t}).values
        time_pd = pd.to_datetime(time_val)

        if time_pd.hour != 12:
            continue

        print(f"\nProcessing 12 UTC time: {time_pd}")

        if time_pd not in npy_times:
            print(f"No matching DIFFUSION file for {time_pd}, skipping")
            continue

        cerra_da = ds_cerra[args.variable].isel({time_dim: t})

        if "number" in cerra_da.dims:
            cerra_da = cerra_da.isel(number=0)

        cerra_da, lon2d_adj = subset_domain_2d(
            cerra_da,
            lat2d,
            lon2d,
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
        )

        npy_da = ds_npy[args.variable].sel(time=time_pd)
        npy_data = npy_da.values

        print(f"CERRA shape: {cerra_da.shape}")
        print(f"Original DIFFUSION shape: {npy_data.shape}")

        npy_regridded = regrid_npy_to_cerra_shape(
            npy_data,
            target_shape=cerra_da.shape,
        )

        print(f" DIFFUSION shape: {npy_regridded.shape}")

        plot_cerra_npy_difference(
            cerra_da,
            npy_regridded,
            lat2d,
            lon2d_adj,
            args,
            time_pd,
        )

        plot_cerra_npy_two_panel(
            cerra_da,
            npy_regridded,
            lat2d,
            lon2d_adj,
            args,
            time_pd,
        )

    ds_cerra.close()
    ds_npy.close()

    print("\nDONE")


if __name__ == "__main__":
    main()

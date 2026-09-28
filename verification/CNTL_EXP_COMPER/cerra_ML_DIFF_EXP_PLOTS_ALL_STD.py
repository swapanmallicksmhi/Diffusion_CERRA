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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CERRA/Zarr with EXP-1 to EXP-5 DIFFUSION files at 12 UTC"
    )

    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--npy-pattern",
        required=True,
        help="Use {exp} in the path, e.g. /path/{exp}/January/*/best_overall/best_t2m_cerra_mean*.npy",
    )

    parser.add_argument(
        "--experiments",
        default="EXP-1,EXP-2,EXP-3,EXP-4,EXP-5",
        help="Comma-separated experiment names",
    )

    parser.add_argument("--variable", default="t2m_cerra_mean")

    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)

    return parser.parse_args()


def extract_time_from_npy_path(filepath):
    match = re.search(r"/(\d{8})/best_overall/", filepath)

    if match is None:
        raise ValueError(f"Could not extract date from path:\n{filepath}")

    date_str = match.group(1)
    return pd.to_datetime(date_str, format="%Y%m%d") + pd.Timedelta(hours=12)


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


def create_npy_netcdf(args, exp_name, exp_pattern, exp_output_dir):
    files = sorted(glob.glob(exp_pattern))

    if len(files) == 0:
        print(f"No DIFFUSION files found for {exp_name}:\n{exp_pattern}")
        return None

    print(f"\n{exp_name}: Found {len(files)} DIFFUSION files")

    npy_output_dir = os.path.join(exp_output_dir, "npy_to_netcdf")
    copy_dir = os.path.join(npy_output_dir, "copied_npy_files")
    output_nc = os.path.join(
        npy_output_dir,
        f"{exp_name}_best_t2m_cerra_mean_12UTC_original_grid.nc",
    )

    os.makedirs(npy_output_dir, exist_ok=True)
    os.makedirs(copy_dir, exist_ok=True)

    data_list = []
    time_list = []

    for f in files:
        print(f"Reading {exp_name}: {f}")

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

    ds_npy[args.variable].attrs["long_name"] = f"Best t2m CERRA mean from {exp_name} DIFFUSION files"
    ds_npy[args.variable].attrs["units"] = "K"
    ds_npy.attrs["description"] = f"Original {exp_name} DIFFUSION files combined into one NetCDF; valid at 12 UTC"

    encoding = {
        args.variable: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }

    ds_npy.to_netcdf(output_nc, encoding=encoding)

    print(f"Saved original-grid {exp_name} NetCDF:\n{output_nc}")

    return ds_npy


def get_difference_cmap_norm():
    levels = [-1, -.8, -.6, -.4, -.1, .1, .4, .6, .8, 1]
    #levels = [-10, -8, -6, -4, -1, 1, 4, 6, 8, 10]

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


def plot_cerra_npy_difference(
    cerra_da,
    npy_regridded,
    lat2d,
    lon2d_adj,
    args,
    valid_time,
    exp_name,
    exp_output_dir,
):
    diff = cerra_da.values - npy_regridded

    cmap, norm, levels = get_difference_cmap_norm()

    time_str = valid_time.strftime("%Y%m%d_%H%M")

    out_file = os.path.join(
        exp_output_dir,
        f"DIFF_STD_CERRA_minus_REGRIDDED_{exp_name}_{args.variable}_{time_str}.png",
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

    plt.title(f"CERRA - {exp_name} | {valid_time.strftime('%Y-%m-%d %H:%M UTC')}")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved difference plot: {out_file}")


def plot_cerra_npy_two_panel(
    cerra_da,
    npy_regridded,
    lat2d,
    lon2d_adj,
    args,
    valid_time,
    exp_name,
    exp_output_dir,
):
    levels = np.arange(242, 300, 2)
    cmap = plt.get_cmap("jet", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)

    time_str = valid_time.strftime("%Y%m%d_%H%M")

    out_file = os.path.join(
        exp_output_dir,
        f"TWO_PANEL_STD_CERRA_REGRIDDED_{exp_name}_{args.variable}_{time_str}.png",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    titles = [
        f"CERRA | {valid_time.strftime('%Y-%m-%d %H:%M UTC')}",
        f"{exp_name} | {valid_time.strftime('%Y-%m-%d %H:%M UTC')}",
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
        f"CERRA and {exp_name} temperature comparison | {valid_time.strftime('%Y-%m-%d 12 UTC')}",
        fontsize=16,
    )

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved two-panel plot: {out_file}")


def process_experiment(args, exp_name, ds_cerra, time_dim, lat2d, lon2d):
    exp_pattern = args.npy_pattern.format(exp=exp_name)
    exp_output_dir = os.path.join(args.output_dir, exp_name)

    os.makedirs(exp_output_dir, exist_ok=True)

    ds_npy = create_npy_netcdf(args, exp_name, exp_pattern, exp_output_dir)

    if ds_npy is None:
        return

    npy_times = pd.to_datetime(ds_npy["time"].values)

    for t in range(ds_cerra.sizes[time_dim]):

        time_val = ds_cerra[time_dim].isel({time_dim: t}).values
        time_pd = pd.to_datetime(time_val)

        if time_pd.hour != 12:
            continue

        print(f"\n{exp_name}: Processing 12 UTC time: {time_pd}")

        if time_pd not in npy_times:
            print(f"{exp_name}: No matching DIFFUSION file for {time_pd}, skipping")
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

        print(f"{exp_name}: CERRA shape: {cerra_da.shape}")
        print(f"{exp_name}: Original DIFFUSION shape: {npy_data.shape}")

        npy_regridded = regrid_npy_to_cerra_shape(
            npy_data,
            target_shape=cerra_da.shape,
        )

        print(f"{exp_name}: Regridded DIFFUSION shape: {npy_regridded.shape}")

        plot_cerra_npy_difference(
            cerra_da,
            npy_regridded,
            lat2d,
            lon2d_adj,
            args,
            time_pd,
            exp_name,
            exp_output_dir,
        )

        plot_cerra_npy_two_panel(
            cerra_da,
            npy_regridded,
            lat2d,
            lon2d_adj,
            args,
            time_pd,
            exp_name,
            exp_output_dir,
        )

    ds_npy.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening Zarr: {args.zarr_path}")
    ds_cerra = xr.open_dataset(args.zarr_path, engine="zarr")

    if args.variable not in ds_cerra:
        raise ValueError(f"Variable '{args.variable}' not found in Zarr file")

    time_dim = "valid_time" if "valid_time" in ds_cerra[args.variable].dims else "time"

    lat2d = ds_cerra["latitude"]
    lon2d = ds_cerra["longitude"]

    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]

    for exp_name in experiments:
        print(f"\n==================================================")
        print(f"Processing {exp_name}")
        print(f"==================================================")

        process_experiment(
            args,
            exp_name,
            ds_cerra,
            time_dim,
            lat2d,
            lon2d,
        )

    ds_cerra.close()

    print("\nDONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
import re
import glob
import argparse
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
        description=(
            "Calculate all-days mean for CERRA and DIFFUSION datasets at 12 UTC, "
            "then plot mean CERRA, mean DIFFUTION, and their difference."
        )
    )
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--npy-pattern",
        default=(
            "/cfs/klemming/scratch/y/yarongc/SDM_seasonalStrongCond/"
            "cfg_1/June/*/best_overall/best_t2m_cerra_mean*.npy"
            #"cfg_1/January/*/best_overall/best_t2m_cerra_mean*.npy"
        ),
    )

    parser.add_argument("--variable", default="t2m_cerra_mean")

    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)

    return parser.parse_args()


# ==========================================================
# ==========================================================
def extract_time_from_npy_path(filepath):
    match = re.search(r"/(\d{8})/best_overall/", filepath)

    if match is None:
        raise ValueError(f"Could not extract date from path:\n{filepath}")

    date_str = match.group(1)

    return pd.to_datetime(date_str, format="%Y%m%d") + pd.Timedelta(hours=12)


# ==========================================================
# SUBSET DOMAIN
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
# DIFFERENCE COLORMAP
# White between -2 and +2 K
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
# TEMPERATURE COLORMAP
# Same as previous structure
# ==========================================================
def get_temperature_cmap_norm():
    levels = np.arange(242, 300, 2)
    cmap = plt.get_cmap("jet", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)

    return cmap, norm, levels


# ==========================================================
# LOAD DIFFUSIO FILES INTO DICTIONARY
# ==========================================================
def load_npy_files(args):
    files = sorted(glob.glob(args.npy_pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No DIFFUSION files found:\n{args.npy_pattern}")

    npy_dict = {}

    for f in files:
        valid_time = extract_time_from_npy_path(f)

        data = np.load(f)
        data = np.squeeze(data)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array but got {data.shape} for:\n{f}")

        npy_dict[valid_time] = data.astype(np.float32)

    print(f"Loaded {len(npy_dict)} DIFFUSION files")

    return npy_dict


# ==========================================================
# TWO-PANEL MEAN TEMPERATURE PLOT
# ==========================================================
def plot_mean_two_panel(
    mean_cerra,
    mean_npy,
    lat2d,
    lon2d_adj,
    args,
):
    cmap, norm, levels = get_temperature_cmap_norm()

    out_file = os.path.join(
        args.output_dir,
        f"MEAN_TWO_PANEL_CERRA_DIFFUSION_{args.variable}.png",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    titles = [
        "Mean CERRA temperature",
        "Mean DIFFUSION temperature",
    ]

    data_list = [mean_cerra, mean_npy]

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

    plt.suptitle("All-days mean temperature comparison", fontsize=16)

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved mean two-panel plot: {out_file}")


# ==========================================================
# MEAN DIFFERENCE PLOT
# ==========================================================
def plot_mean_difference(
    mean_diff,
    lat2d,
    lon2d_adj,
    args,
):
    cmap, norm, levels = get_difference_cmap_norm()

    out_file = os.path.join(
        args.output_dir,
        f"MEAN_DIFF_CERRA_minus_REGRIDDED_DIFFUSION_{args.variable}.png",
    )

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon2d_adj,
        lat2d,
        mean_diff,
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

    cbar.set_label("Mean CERRA - Mean DIFFUSION (K)")
    cbar.set_ticks(levels)

    plt.title("All-days mean difference: CERRA - DIFFUSION")

    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved mean difference plot: {out_file}")


# ==========================================================
# SAVE MEAN FIELDS TO NETCDF
# ==========================================================
def save_mean_to_netcdf(
    mean_cerra,
    mean_npy,
    mean_diff,
    lat2d,
    lon2d_adj,
    args,
):
    out_nc = os.path.join(
        args.output_dir,
        f"MEAN_CERRA_DIFFUSION_DIFF_{args.variable}.nc",
    )

    ds_out = xr.Dataset(
        {
            "mean_cerra": (("y", "x"), mean_cerra.astype(np.float32)),
            "mean_npy_regridded": (("y", "x"), mean_npy.astype(np.float32)),
            "mean_difference_cerra_minus_npy": (
                ("y", "x"),
                mean_diff.astype(np.float32),
            ),
        },
        coords={
            "latitude": (("y", "x"), lat2d.values.astype(np.float32)),
            "longitude": (("y", "x"), lon2d_adj.values.astype(np.float32)),
        },
    )

    ds_out["mean_cerra"].attrs["units"] = "K"
    ds_out["mean_npy_regridded"].attrs["units"] = "K"
    ds_out["mean_difference_cerra_minus_npy"].attrs["units"] = "K"

    ds_out.attrs["description"] = (
        "All-days 12 UTC mean fields for CERRA, DIFFUSION, "
        "and their difference."
    )

    encoding = {
        "mean_cerra": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "mean_npy_regridded": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "mean_difference_cerra_minus_npy": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        },
    }

    ds_out.to_netcdf(out_nc, encoding=encoding)

    print(f"Saved mean NetCDF: {out_nc}")


# ==========================================================
# MAIN
# ==========================================================
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    npy_dict = load_npy_files(args)

    print(f"Opening Zarr: {args.zarr_path}")
    ds_cerra = xr.open_dataset(args.zarr_path, engine="zarr")

    if args.variable not in ds_cerra:
        raise ValueError(f"Variable '{args.variable}' not found in Zarr file")

    time_dim = "valid_time" if "valid_time" in ds_cerra[args.variable].dims else "time"

    lat2d_full = ds_cerra["latitude"]
    lon2d_full = ds_cerra["longitude"]

    cerra_sum = None
    npy_sum = None
    count = 0

    final_lat2d = None
    final_lon2d_adj = None

    for t in range(ds_cerra.sizes[time_dim]):

        time_val = ds_cerra[time_dim].isel({time_dim: t}).values
        time_pd = pd.to_datetime(time_val)

        if time_pd.hour != 12:
            continue

        if time_pd not in npy_dict:
            print(f"No matching DIFFUSION file for {time_pd}, skipping")
            continue

        print(f"Using matched time: {time_pd}")

        cerra_da = ds_cerra[args.variable].isel({time_dim: t})

        if "number" in cerra_da.dims:
            cerra_da = cerra_da.isel(number=0)

        cerra_da, lon2d_adj = subset_domain_2d(
            cerra_da,
            lat2d_full,
            lon2d_full,
            args.lat_min,
            args.lat_max,
            args.lon_min,
            args.lon_max,
        )

        npy_data = npy_dict[time_pd]

        npy_regridded = regrid_npy_to_cerra_shape(
            npy_data,
            target_shape=cerra_da.shape,
        )

        cerra_values = cerra_da.values.astype(np.float64)
        npy_values = npy_regridded.astype(np.float64)

        valid_mask = np.isfinite(cerra_values) & np.isfinite(npy_values)

        if cerra_sum is None:
            cerra_sum = np.zeros_like(cerra_values, dtype=np.float64)
            npy_sum = np.zeros_like(npy_values, dtype=np.float64)
            count_arr = np.zeros_like(cerra_values, dtype=np.float64)

            final_lat2d = lat2d_full
            final_lon2d_adj = lon2d_adj

        cerra_sum[valid_mask] += cerra_values[valid_mask]
        npy_sum[valid_mask] += npy_values[valid_mask]
        count_arr[valid_mask] += 1

        count += 1

    if count == 0:
        raise RuntimeError("No matched 12 UTC CERRA and DIFFUSION times found.")

    mean_cerra = np.full_like(cerra_sum, np.nan, dtype=np.float64)
    mean_npy = np.full_like(npy_sum, np.nan, dtype=np.float64)

    valid_count = count_arr > 0

    mean_cerra[valid_count] = cerra_sum[valid_count] / count_arr[valid_count]
    mean_npy[valid_count] = npy_sum[valid_count] / count_arr[valid_count]

    mean_diff = mean_cerra - mean_npy

    print(f"Total matched days used for mean: {count}")

    save_mean_to_netcdf(
        mean_cerra,
        mean_npy,
        mean_diff,
        final_lat2d,
        final_lon2d_adj,
        args,
    )

    plot_mean_two_panel(
        mean_cerra,
        mean_npy,
        final_lat2d,
        final_lon2d_adj,
        args,
    )

    plot_mean_difference(
        mean_diff,
        final_lat2d,
        final_lon2d_adj,
        args,
    )

    ds_cerra.close()

    print("\nDONE")


if __name__ == "__main__":
    main()

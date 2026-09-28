#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

# ==========================================================
# ARGUMENTS
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Zarr + OBS + Difference plotting")
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    return parser.parse_args()


# ==========================================================
# SUBSET DOMAIN
# ==========================================================
def subset_domain_2d(da, lat2d, lon2d, lat_min, lat_max, lon_min, lon_max):
    lon2d_adj = lon2d.copy()
    if float(lon2d.max()) > 180:
        lon2d_adj = lon2d.where(lon2d <= 180, lon2d - 360)

    mask = ((lat2d >= lat_min) & (lat2d <= lat_max) &
            (lon2d_adj >= lon_min) & (lon2d_adj <= lon_max))

    return da.where(mask), lon2d_adj


# ==========================================================
# OBSERVATION SUBROUTINE
# ==========================================================
def overlay_observations(ax, csv_file, valid_time_str, cmap, norm,
                         lat_min, lat_max, lon_min, lon_max):

    df = pd.read_csv(csv_file)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["t2m_k"] = pd.to_numeric(df["t2m_k"], errors="coerce")
    df["wmo_id"] = pd.to_numeric(df["wmo_id"], errors="coerce")
    df["valid_time"] = df["valid_time"].astype(str)
    df = df.dropna(subset=["latitude", "longitude", "t2m_k", "valid_time", "wmo_id"])

    df_obs = df[df["valid_time"] == valid_time_str].copy()
    if df_obs.empty:
        return None

    valid_prefixes = ("11", "13", "14", "20", "21", "22", "23", "24", "25")
    df_obs["wmo_str"] = df_obs["wmo_id"].astype("Int64").astype(str)
    df_obs = df_obs[df_obs["wmo_str"].str.startswith(valid_prefixes)]
    df_obs = df_obs[(df_obs["latitude"].between(lat_min, lat_max)) &
                    (df_obs["longitude"].between(lon_min, lon_max))]

    if df_obs.empty:
        return None

    ax.scatter(
        df_obs["longitude"], df_obs["latitude"],
        c=df_obs["t2m_k"], cmap=cmap, norm=norm,
        s=60, edgecolors="black", linewidths=0.5,
        transform=ccrs.PlateCarree(), zorder=6)
    #print(f"Overlayed {len(df_obs)} filtered observations")
    return df_obs

# ==========================================================
# INTERPOLATE MODEL TO OBS
# ==========================================================
def interpolate_to_stations(da_model, df_obs, lat2d, lon2d):
    """
    Interpolates a 2D or 1D model field to station points using griddata.
    """

    lat_vals = lat2d.values
    lon_vals = lon2d.values
    model_vals = da_model.values

    # Flatten 2D coordinates if needed
    points = np.column_stack((lon_vals.flatten(), lat_vals.flatten()))
    values = model_vals.flatten()

    station_points = np.column_stack((df_obs["longitude"].values, df_obs["latitude"].values))
    interp_vals = griddata(points, values, station_points, method="linear")

    # Fill NaNs with nearest neighbor
    nan_mask = np.isnan(interp_vals)
    if nan_mask.any():
        interp_vals[nan_mask] = griddata(points, values, station_points[nan_mask], method="nearest")

    return interp_vals


# ==========================================================
# DIFFERENCE PLOT
# ==========================================================
def plot_difference(var, da_model, df_obs, args, valid_time_str, time_str):

    if df_obs is None or df_obs.empty:
        return

    model_vals = interpolate_to_stations(da_model, df_obs,
                                         da_model.latitude if "latitude" in da_model.coords else da_model.y,
                                         da_model.longitude if "longitude" in da_model.coords else da_model.x)

    diff = df_obs["t2m_k"].values - model_vals

    levels = np.arange(-10, 11, 1)
    cmap = plt.get_cmap("RdBu_r", len(levels)-1)
    norm = BoundaryNorm(levels, cmap.N)

    out_file = os.path.join(args.output_dir, f"DIFF_{var}_{time_str}.png")

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([args.lon_min, args.lon_max, args.lat_min, args.lat_max])
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    sc = ax.scatter(df_obs["longitude"], df_obs["latitude"],
                    c=diff, cmap=cmap, norm=norm,
                    s=70, edgecolors="black",
                    transform=ccrs.PlateCarree(), zorder=6)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("OBS - MODEL (K)")
    cbar.set_ticks(levels)

    plt.title(f"Difference (OBS - MODEL) | {valid_time_str}")
    plt.savefig(out_file, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved difference: {out_file}")


# ==========================================================
# MAIN VARIABLE PLOT
# ==========================================================
def plot_variable(ds, var, t, args):
    da = ds[var]

    time_dim = "valid_time" if "valid_time" in da.dims else "time"
    da = da.isel({time_dim: t})

    if "number" in da.dims:
        da = da.isel(number=0)

    lat2d = ds["latitude"]
    lon2d = ds["longitude"]

    da, lon2d_adj = subset_domain_2d(da, lat2d, lon2d,
                                     args.lat_min, args.lat_max,
                                     args.lon_min, args.lon_max)

    if np.isnan(da.values).all():
        return

    levels = np.arange(242, 300, 2)
    cmap = plt.get_cmap("jet", len(levels)-1)
    norm = BoundaryNorm(levels, cmap.N)

    time_val = ds[time_dim].isel({time_dim: t}).values
    time_pd = pd.to_datetime(time_val)
    valid_time_str = time_pd.strftime("%Y-%m-%d %H:%M")
    time_str = time_pd.strftime("%Y%m%d_%H%M")

    print(f"\nProcessing {var} at {valid_time_str}")

    out_file = os.path.join(args.output_dir, f"{var}_{time_str}.png")
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(lon2d_adj, lat2d, da, cmap=cmap, norm=norm,
                         shading="auto", transform=ccrs.PlateCarree())

    ax.set_extent([args.lon_min, args.lon_max, args.lat_min, args.lat_max])
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    df_obs = overlay_observations(ax, args.csv_file, valid_time_str,
                                  cmap, norm, args.lat_min, args.lat_max,
                                  args.lon_min, args.lon_max)

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.7)
    cbar.set_label("T2M (K)")
    cbar.set_ticks(levels)

    plt.title(f"{var} + OBS | {valid_time_str}")
    plt.savefig(out_file, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")

    plot_difference(var, da, df_obs, args, valid_time_str, time_str)


# ==========================================================
# MAIN
# ==========================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Opening Zarr: {args.zarr_path}")
    ds = xr.open_dataset(args.zarr_path, engine="zarr")

    n_times = ds.sizes["valid_time"]

    for var in args.variables:
        if var not in ds:
            print(f"Skipping {var}")
            continue

        print(f"Processing variable: {var}")
        for t in range(n_times):
            plot_variable(ds, var, t, args)

    ds.close()
    print("DONE")


if __name__ == "__main__":
    main()

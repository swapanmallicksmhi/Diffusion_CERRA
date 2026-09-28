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
from scipy.stats import pearsonr

# ==========================================================
# ARGUMENTS
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Full verification pipeline")
    parser.add_argument("--zarr-path", required=True, help="Path to Zarr dataset")
    parser.add_argument("--csv-file", required=True, help="CSV file with observations")
    parser.add_argument("--output-dir", required=True, help="Directory to save output plots")
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    return parser.parse_args()


# ==========================================================
# LOAD OBSERVATIONS
# ==========================================================
def load_obs(csv_file):
    df = pd.read_csv(csv_file)
    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["t2m_k"] = pd.to_numeric(df["t2m_k"], errors="coerce")
    df["wmo_id"] = pd.to_numeric(df["wmo_id"], errors="coerce")
    df = df.dropna(subset=["valid_time", "latitude", "longitude", "t2m_k"])
    df["wmo_str"] = df["wmo_id"].astype("Int64").astype(str)
    valid_prefixes = ("11", "13", "14", "20", "21", "22", "23", "24", "25")
    df = df[df["wmo_str"].str.startswith(valid_prefixes)]
    return df


# ==========================================================
# FILTER DOMAIN
# ==========================================================
def filter_domain(df, lat_min, lat_max, lon_min, lon_max):
    return df[
        (df["latitude"].between(lat_min, lat_max)) &
        (df["longitude"].between(lon_min, lon_max))
    ]


# ==========================================================
# INTERPOLATE MODEL TO STATIONS (2D SAFE)
# ==========================================================
def interpolate_model(ds, var, lats, lons, t_idx, time_dim):
    da = ds[var].isel({time_dim: t_idx})

    # Determine coordinates
    if ("latitude" in da.dims or "longitude" in da.dims) or ("latitude" in da.coords and "longitude" in da.coords):
        lat_name = "latitude" if "latitude" in da.coords else "y"
        lon_name = "longitude" if "longitude" in da.coords else "x"
        lat_vals = da[lat_name].values
        lon_vals = da[lon_name].values
        # Handle 2D coordinates
        if lat_vals.ndim == 2 and lon_vals.ndim == 2:
            points = np.column_stack((lon_vals.flatten(), lat_vals.flatten()))
            values = da.values.flatten()
            interp_vals = griddata(points, values, (lons, lats), method="linear")
            nan_mask = np.isnan(interp_vals)
            if nan_mask.any():
                interp_vals[nan_mask] = griddata(points, values, (lons[nan_mask], lats[nan_mask]), method="nearest")
            return interp_vals
        else:  # 1D lat/lon
            from scipy.interpolate import RegularGridInterpolator
            interpolator = RegularGridInterpolator((lat_vals, lon_vals), da.values, bounds_error=False, fill_value=np.nan)
            pts = np.column_stack((lats, lons))
            interp_vals = interpolator(pts)
            return interp_vals
    else:
        raise ValueError(f"Cannot find latitude/longitude coordinates in {var}")


# ==========================================================
# STATION TIME SERIES
# ==========================================================
def plot_station_timeseries(df, output_dir):
    print("Plotting station time series...")
    for wmo, group in df.groupby("wmo_id"):
        group = group.sort_values("time")
        plt.figure(figsize=(10, 5))
        plt.scatter(group["time"], group["obs"], label="OBS", color="black", s=40, marker='o')
        plt.plot(group["time"], group["cerra"], label="CERRA")
        #plt.plot(group["time"], group["era5"], label="ERA5")
        plt.title(f"Station {int(wmo)}")
        plt.ylabel("T2M (K)")
        plt.legend()
        plt.xticks(rotation=45)
        fname = os.path.join(output_dir, f"timeseries_{int(wmo)}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=100)
        plt.close()


# ==========================================================
# SPATIAL MAPS WITH FIXED SCALE
# ==========================================================
def plot_map(lon, lat, data, title, out_file, vmin=-6, vmax=6, cmap="RdBu_r"):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    sc = ax.scatter(lon, lat, c=data, cmap=cmap, s=40, edgecolors="k", vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    plt.colorbar(sc, ax=ax, shrink=0.7, label="K")
    plt.title(title)
    plt.savefig(out_file, dpi=100)
    plt.close()


# ==========================================================
# STATISTICS FOR PAPER
# ==========================================================
def compute_statistics(df):
    stats = {}
    #for model in ["cerra", "era5"]:
    for model in ["cerra"]:
        bias = df[model] - df["obs"]
        rmse = np.sqrt(np.mean(bias**2))
        mean_bias = np.mean(bias)
        std_obs = np.std(df["obs"])
        std_model = np.std(df[model])
        # Pearson correlation
        corr, _ = pearsonr(df["obs"], df[model])
        stats[model] = {
            "mean_bias": mean_bias,
            "rmse": rmse,
            "std_obs": std_obs,
            "std_model": std_model,
            "correlation": corr
        }
    return stats


# ==========================================================
# MAIN
# ==========================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    ds = xr.open_dataset(args.zarr_path, engine="zarr")
    df_obs = load_obs(args.csv_file)
    df_obs = filter_domain(df_obs, args.lat_min, args.lat_max, args.lon_min, args.lon_max)

    # Determine time dimension
    if "valid_time" in ds.coords:
        time_dim = "valid_time"
    elif "time" in ds.coords:
        time_dim = "time"
    else:
        raise ValueError("No time coordinate found in the dataset!")

    times = pd.to_datetime(ds[time_dim].values)
    all_records = []

    for t_idx, t_val in enumerate(times):
        print(f"Processing {t_val}")
        df_t = df_obs[df_obs["valid_time"] == t_val]
        if df_t.empty:
            continue
        lats = df_t["latitude"].values
        lons = df_t["longitude"].values
        cerra_vals = interpolate_model(ds, "t2m", lats, lons, t_idx, time_dim)
        #cerra_vals = interpolate_model(ds, "t2m_cerra_mean", lats, lons, t_idx, time_dim)
        #era5_vals  = interpolate_model(ds, "t2m_era5_mean", lats, lons, t_idx, time_dim)
        for i in range(len(df_t)):
            all_records.append({
                "wmo_id": df_t.iloc[i]["wmo_id"],
                "time": t_val,
                "lat": lats[i],
                "lon": lons[i],
                "obs": df_t.iloc[i]["t2m_k"],
                "cerra": cerra_vals[i],
                #"era5": era5_vals[i]
            })

    df_all = pd.DataFrame(all_records)

    # Compute bias/RMSE
    df_all["bias_cerra"] = df_all["cerra"] - df_all["obs"]
    #df_all["bias_era5"]  = df_all["era5"] - df_all["obs"]
    df_all["rmse_cerra"] = np.sqrt(df_all["bias_cerra"]**2)
    #df_all["rmse_era5"]  = np.sqrt(df_all["bias_era5"]**2)

    # Aggregate spatially
    df_mean = df_all.groupby(["lat", "lon"]).mean(numeric_only=True).reset_index()
    df_mean["rmse_cerra"] = np.sqrt(df_mean["rmse_cerra"])
    #df_mean["rmse_era5"]  = np.sqrt(df_mean["rmse_era5"])

    # Plot maps
    plot_map(df_mean["lon"], df_mean["lat"], df_mean["bias_cerra"], "CERRA Bias",
             os.path.join(args.output_dir, "bias_cerra.png"))
    #plot_map(df_mean["lon"], df_mean["lat"], df_mean["bias_era5"], "ERA5 Bias",
    #         os.path.join(args.output_dir, "bias_era5.png"))
    plot_map(df_mean["lon"], df_mean["lat"], df_mean["rmse_cerra"], "CERRA RMSE",
             os.path.join(args.output_dir, "rmse_cerra.png"))
    #plot_map(df_mean["lon"], df_mean["lat"], df_mean["rmse_era5"], "ERA5 RMSE",
    #         os.path.join(args.output_dir, "rmse_era5.png"))

    # Time series
    plot_station_timeseries(df_all, args.output_dir)

    # Compute statistics
    stats = compute_statistics(df_all)
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(args.output_dir, "verification_statistics.csv"))
    print("Statistics saved to verification_statistics.csv")


if __name__ == "__main__":
    main()

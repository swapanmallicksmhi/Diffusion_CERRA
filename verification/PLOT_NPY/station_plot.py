#!/usr/bin/env python3

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr


def parse_args():
    parser = argparse.ArgumentParser(description="Full verification pipeline")
    parser.add_argument("--zarr-path", required=True, help="Path to Zarr dataset")
    parser.add_argument("--csv-file", required=True, help="CSV file with observations")
    parser.add_argument("--output-dir", required=True, help="Directory to save output plots")
    parser.add_argument(
        "--npy-pattern",
        default="/cfs/klemming/scratch/y/yarongc/SDM_seasonalStrongCond/cfg_1/June/*/best_overall/best_t2m_cerra_mean*.npy",
        #default="/cfs/klemming/scratch/y/yarongc/SDM_seasonalStrongCond/cfg_1/January/*/best_overall/best_t2m_cerra_mean*.npy",
    )
    parser.add_argument("--variable", default="t2m")
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


def load_npy_files(npy_pattern):
    files = sorted(glob.glob(npy_pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No NPY files found:\n{npy_pattern}")

    npy_dict = {}

    for f in files:
        valid_time = extract_time_from_npy_path(f)
        data = np.load(f)
        data = np.squeeze(data)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array but got {data.shape} for:\n{f}")

        npy_dict[valid_time] = data.astype(np.float32)

    return npy_dict


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
    df = df[df["valid_time"].dt.hour == 12]
    return df


def filter_domain(df, lat_min, lat_max, lon_min, lon_max):
    return df[
        (df["latitude"].between(lat_min, lat_max)) &
        (df["longitude"].between(lon_min, lon_max))
    ]


def interpolate_model(ds, var, lats, lons, t_idx, time_dim):
    da = ds[var].isel({time_dim: t_idx})

    if "number" in da.dims:
        da = da.isel(number=0)

    if ("latitude" in da.dims or "longitude" in da.dims) or ("latitude" in da.coords and "longitude" in da.coords):
        lat_name = "latitude" if "latitude" in da.coords else "y"
        lon_name = "longitude" if "longitude" in da.coords else "x"

        lat_vals = da[lat_name].values
        lon_vals = da[lon_name].values

        if lon_vals.max() > 180:
            lon_vals = np.where(lon_vals <= 180, lon_vals, lon_vals - 360)

        if lat_vals.ndim == 2 and lon_vals.ndim == 2:
            points = np.column_stack((lon_vals.flatten(), lat_vals.flatten()))
            values = da.values.flatten()

            mask = np.isfinite(values)
            points = points[mask]
            values = values[mask]

            interp_vals = griddata(points, values, (lons, lats), method="linear")

            nan_mask = np.isnan(interp_vals)
            if nan_mask.any():
                interp_vals[nan_mask] = griddata(
                    points,
                    values,
                    (lons[nan_mask], lats[nan_mask]),
                    method="nearest",
                )

            return interp_vals

        else:
            from scipy.interpolate import RegularGridInterpolator

            interpolator = RegularGridInterpolator(
                (lat_vals, lon_vals),
                da.values,
                bounds_error=False,
                fill_value=np.nan,
            )

            pts = np.column_stack((lats, lons))
            interp_vals = interpolator(pts)

            return interp_vals

    else:
        raise ValueError(f"Cannot find latitude/longitude coordinates in {var}")


def interpolate_npy_to_stations(npy_regridded, lat2d, lon2d, lats, lons):
    lat_vals = lat2d.values
    lon_vals = lon2d.values

    if lon_vals.max() > 180:
        lon_vals = np.where(lon_vals <= 180, lon_vals, lon_vals - 360)

    points = np.column_stack((lon_vals.flatten(), lat_vals.flatten()))
    values = npy_regridded.flatten()

    mask = np.isfinite(values)
    points = points[mask]
    values = values[mask]

    interp_vals = griddata(points, values, (lons, lats), method="linear")

    nan_mask = np.isnan(interp_vals)
    if nan_mask.any():
        interp_vals[nan_mask] = griddata(
            points,
            values,
            (lons[nan_mask], lats[nan_mask]),
            method="nearest",
        )

    return interp_vals


def plot_station_timeseries(df, output_dir):
    print("Plotting station time series...")

    for wmo, group in df.groupby("wmo_id"):
        group = group.sort_values("time")

        plt.figure(figsize=(10, 5))

        plt.scatter(
            group["time"],
            group["obs"],
            label="OBS",
            color="black",
            s=40,
            marker="o",
        )

        plt.plot(group["time"], group["cerra"], label="CERRA5.5")
        plt.plot(group["time"], group["npy"], label="CNTL")

        plt.title(f"Station {int(wmo)}")
        plt.ylabel("T2M (K)")
        plt.legend()
        plt.xticks(rotation=45)

        fname = os.path.join(output_dir, f"timeseries_{int(wmo)}_12UTC.png")

        plt.tight_layout()
        plt.savefig(fname, dpi=100)
        plt.close()


def plot_map(lon, lat, data, title, out_file, vmin=-6, vmax=6, cmap="RdBu_r"):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    sc = ax.scatter(
        lon,
        lat,
        c=data,
        cmap=cmap,
        s=40,
        edgecolors="k",
        vmin=vmin,
        vmax=vmax,
    )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)

    plt.colorbar(sc, ax=ax, shrink=0.7, label="K")
    plt.title(title)
    plt.savefig(out_file, dpi=100)
    plt.close()


def compute_statistics(df):
    stats = {}

    for model in ["cerra", "npy"]:
        valid = df[["obs", model]].dropna()

        bias = valid[model] - valid["obs"]
        rmse = np.sqrt(np.mean(bias**2))
        mean_bias = np.mean(bias)
        std_obs = np.std(valid["obs"])
        std_model = np.std(valid[model])

        if len(valid) > 1:
            corr, _ = pearsonr(valid["obs"], valid[model])
        else:
            corr = np.nan

        stats[model] = {
            "mean_bias": mean_bias,
            "rmse": rmse,
            "std_obs": std_obs,
            "std_model": std_model,
            "correlation": corr,
        }

    return stats


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")

    ds = xr.open_dataset(args.zarr_path, engine="zarr")
    df_obs = load_obs(args.csv_file)
    df_obs = filter_domain(
        df_obs,
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max,
    )

    npy_dict = load_npy_files(args.npy_pattern)

    if "valid_time" in ds.coords:
        time_dim = "valid_time"
    elif "time" in ds.coords:
        time_dim = "time"
    else:
        raise ValueError("No time coordinate found in the dataset!")

    times = pd.to_datetime(ds[time_dim].values)

    lat2d = ds["latitude"]
    lon2d = ds["longitude"]

    all_records = []

    for t_idx, t_val in enumerate(times):
        t_val = pd.to_datetime(t_val)

        if t_val.hour != 12:
            continue

        print(f"Processing {t_val}")

        if t_val not in npy_dict:
            print(f"No matching NPY file for {t_val}, skipping")
            continue

        df_t = df_obs[df_obs["valid_time"] == t_val]

        if df_t.empty:
            continue

        lats = df_t["latitude"].values
        lons = df_t["longitude"].values

        cerra_vals = interpolate_model(
            ds,
            args.variable,
            lats,
            lons,
            t_idx,
            time_dim,
        )

        cerra_da = ds[args.variable].isel({time_dim: t_idx})

        if "number" in cerra_da.dims:
            cerra_da = cerra_da.isel(number=0)

        npy_regridded = regrid_npy_to_cerra_shape(
            npy_dict[t_val],
            target_shape=cerra_da.shape,
        )

        npy_vals = interpolate_npy_to_stations(
            npy_regridded,
            lat2d,
            lon2d,
            lats,
            lons,
        )

        for i in range(len(df_t)):
            all_records.append(
                {
                    "wmo_id": df_t.iloc[i]["wmo_id"],
                    "time": t_val,
                    "lat": lats[i],
                    "lon": lons[i],
                    "obs": df_t.iloc[i]["t2m_k"],
                    "cerra": cerra_vals[i],
                    "npy": npy_vals[i],
                }
            )

    df_all = pd.DataFrame(all_records)

    if df_all.empty:
        raise RuntimeError("No matching 12 UTC observations, CERRA, and NPY data found.")

    df_all["bias_cerra"] = df_all["cerra"] - df_all["obs"]
    df_all["bias_npy"] = df_all["npy"] - df_all["obs"]

    df_all["rmse_cerra"] = df_all["bias_cerra"] ** 2
    df_all["rmse_npy"] = df_all["bias_npy"] ** 2

    df_mean = df_all.groupby(["lat", "lon"]).mean(numeric_only=True).reset_index()

    df_mean["rmse_cerra"] = np.sqrt(df_mean["rmse_cerra"])
    df_mean["rmse_npy"] = np.sqrt(df_mean["rmse_npy"])

    plot_map(
        df_mean["lon"],
        df_mean["lat"],
        df_mean["bias_cerra"],
        "CERRA5.5 Bias 12 UTC",
        os.path.join(args.output_dir, "bias_cerra_12UTC.png"),
    )

    plot_map(
        df_mean["lon"],
        df_mean["lat"],
        df_mean["bias_npy"],
        "CNTL Bias 12 UTC",
        os.path.join(args.output_dir, "bias_npy_12UTC.png"),
    )

    plot_map(
        df_mean["lon"],
        df_mean["lat"],
        df_mean["rmse_cerra"],
        "CERRA5.5 RMSE 12 UTC",
        os.path.join(args.output_dir, "rmse_cerra_12UTC.png"),
        vmin=0,
        vmax=6,
        cmap="viridis",
    )

    plot_map(
        df_mean["lon"],
        df_mean["lat"],
        df_mean["rmse_npy"],
        "CNTL RMSE 12 UTC",
        os.path.join(args.output_dir, "rmse_npy_12UTC.png"),
        vmin=0,
        vmax=6,
        cmap="viridis",
    )

    plot_station_timeseries(df_all, args.output_dir)

    stats = compute_statistics(df_all)
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(args.output_dir, "verification_statistics_12UTC.csv"))

    df_all.to_csv(os.path.join(args.output_dir, "station_timeseries_values_12UTC.csv"), index=False)

    print("Statistics saved to verification_statistics_12UTC.csv")
    print("Station values saved to station_timeseries_values_12UTC.csv")

    ds.close()


if __name__ == "__main__":
    main()

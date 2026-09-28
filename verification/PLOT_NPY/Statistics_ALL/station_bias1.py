#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils import load_data, COLORS


# Geographic domain
LATMIN = 54.0
LATMAX = 71.0
LONMIN = 4.0
LONMAX = 25.0


plt.rcParams["font.size"] = 18


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df, model_names = load_data(args.csv_file)

    df = df[
        (df["lat"] >= LATMIN)
        & (df["lat"] <= LATMAX)
        & (df["lon"] >= LONMIN)
        & (df["lon"] <= LONMAX)
    ].copy()

    total_stations = df["wmo_id"].nunique()
    total_records = len(df)

    print("======================================")
    print("Station statistics")
    print("======================================")
    print(f"Geographical domain: {LATMIN}-{LATMAX}°N, {LONMIN}-{LONMAX}°E")
    print(f"Total number of records: {total_records}")
    print(f"Total number of available stations: {total_stations}")
    print("======================================")

    for model_name in model_names:

        df_tmp = df.copy()

        df_tmp[f"bias_{model_name}"] = (
            df_tmp["obs"] - df_tmp[model_name]
        )

        df_station = (
            df_tmp.groupby("wmo_id")
            .agg(
                {
                    f"bias_{model_name}": "mean",
                    "lat": "mean",
                    "lon": "mean",
                }
            )
            .reset_index()
        )

        n_stations = df_station["wmo_id"].nunique()
        mean_bias = df_station[f"bias_{model_name}"].mean()
        std_bias = df_station[f"bias_{model_name}"].std()
        min_bias = df_station[f"bias_{model_name}"].min()
        max_bias = df_station[f"bias_{model_name}"].max()

        print(f"\nModel: {model_name.upper()}")
        print(f"Number of stations used: {n_stations}")
        print(f"Mean station bias: {mean_bias:.3f} K")
        print(f"Standard deviation of station bias: {std_bias:.3f} K")
        print(f"Minimum station bias: {min_bias:.3f} K")
        print(f"Maximum station bias: {max_bias:.3f} K")

        fig = plt.figure(figsize=(12, 12))

        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent(
            [LONMIN, LONMAX, LATMIN, LATMAX],
            crs=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linewidth=1.0)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="white")

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.6,
            linestyle="--",
            alpha=0.5,
        )

        gl.top_labels = False
        gl.right_labels = False

        gl.xlabel_style = {"size": 18}
        gl.ylabel_style = {"size": 18}

        cmap = plt.cm.RdBu_r.copy()

        bounds = [-6, -5, -4, -3, -2, -1,
                   1,  2,  3,  4,  5,  6]

        colors = [
            cmap(0.00),
            cmap(0.08),
            cmap(0.18),
            cmap(0.28),
            cmap(0.38),
            (1, 1, 1, 1),
            cmap(0.62),
            cmap(0.72),
            cmap(0.82),
            cmap(0.92),
            cmap(1.00),
        ]

        custom_cmap = mcolors.ListedColormap(colors)

        norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

        scatter = ax.scatter(
            df_station["lon"],
            df_station["lat"],
            c=df_station[f"bias_{model_name}"],
            cmap=custom_cmap,
            norm=norm,
            s=160,
            edgecolors="black",
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
        )

        cbar = plt.colorbar(
            scatter,
            ax=ax,
            orientation="vertical",
            shrink=0.82,
            pad=0.03,
            extend="both",
            ticks=[-6, -5, -4, -3, -2, -1,
                    1,  2,  3,  4,  5,  6],
        )

        cbar.set_label(
            "Mean bias (K)",
            fontsize=20,
            labelpad=15,
        )

        cbar.ax.tick_params(labelsize=18)

        ax.set_title(
            f"Station-wise Mean Bias ({model_name.upper()})",
            fontsize=22,
            pad=18,
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                args.output_dir,
                f"{model_name}_station_bias_map.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


if __name__ == "__main__":
    main()

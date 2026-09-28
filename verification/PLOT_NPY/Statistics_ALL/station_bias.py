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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df, model_names = load_data(args.csv_file)

    for model_name in model_names:

        df_tmp = df.copy()

        df_tmp[f"bias_{model_name}"] = (df_tmp["obs"]- df_tmp[model_name])
        #df_tmp[f"bias_{model_name}"] = ( df_tmp[model_name] - df_tmp["obs"])

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

        fig = plt.figure(figsize=(10, 10))

        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.set_extent(
            [LONMIN, LONMAX, LATMIN, LATMAX],
            crs=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="white")

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            linestyle="--",
            alpha=0.5,
        )

        gl.top_labels = False
        gl.right_labels = False

        vmin = -6.0
        vmax = 6.0

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
            s=120,
            edgecolors="black",
            linewidths=0.7,
            transform=ccrs.PlateCarree(),
        )

        cbar = plt.colorbar(
            scatter,
            ax=ax,
            orientation="vertical",
            shrink=0.82,
            pad=0.03,
            extend="both",
        )

        cbar.set_label("Mean bias (K)", fontsize=14)

        ax.set_title(
            f"Station-wise Mean Bias ({model_name.upper()})",
            fontsize=16,
            pad=14,
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

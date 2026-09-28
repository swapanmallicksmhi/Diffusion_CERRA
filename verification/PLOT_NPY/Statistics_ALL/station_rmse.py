#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, COLORS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, model_names = load_data(args.csv_file)

    station_stats = []

    for model_name in model_names:
        df_tmp = df.copy()
        df_tmp["error2"] = (df_tmp[model_name] - df_tmp["obs"]) ** 2

        df_station = (
            df_tmp.groupby("wmo_id")["error2"]
            .mean()
            .apply(np.sqrt)
            .reset_index()
            .rename(columns={"error2": f"rmse_{model_name}"})
        )

        station_stats.append(df_station)

    df_rmse = station_stats[0]

    for item in station_stats[1:]:
        df_rmse = df_rmse.merge(item, on="wmo_id", how="outer")

    df_rmse = df_rmse.sort_values(f"rmse_{model_names[0]}", ascending=False)

    x = np.arange(len(df_rmse))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model_name in enumerate(model_names):
        offset = (i - 0.5) * width if len(model_names) == 2 else 0

        ax.bar(
            x + offset,
            df_rmse[f"rmse_{model_name}"],
            width=width,
            label=model_name.upper(),
            color=COLORS.get(model_name),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_rmse["wmo_id"].astype(int).astype(str), rotation=90)
    ax.set_xlabel("WMO station ID")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Station-wise RMSE")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_station_rmse.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

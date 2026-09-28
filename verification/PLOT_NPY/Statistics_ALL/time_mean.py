#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, COLORS, MARKERS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df, model_names = load_data(args.csv_file)
    fig, ax = plt.subplots(figsize=(12, 5))
    df_tmp = df.copy()
    df_tmp["time"] = pd.to_datetime(df_tmp["time"])
    for model_name in model_names:
        df_tmp[f"error_{model_name}"] = df_tmp["obs"] - df_tmp[model_name]
        #df_tmp[f"error_{model_name}"] = df_tmp[model_name] - df_tmp["obs"]
        df_time = (
            df_tmp.groupby("time")[f"error_{model_name}"]
            .mean()
            .reset_index()
        )
        ax.plot(
            df_time["time"],
            df_time[f"error_{model_name}"],
            marker=MARKERS.get(model_name, "o"),
            linewidth=2,
            markersize=5,
            color=COLORS.get(model_name),
            label=model_name.upper(),
        )

    ax.axhline(0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xlabel("Time")
    ax.set_ylabel("(OBS - Model) in K")
    ax.set_ylim(-2, 2)
    ax.set_title("Daily Mean Error")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "time_mean_error.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

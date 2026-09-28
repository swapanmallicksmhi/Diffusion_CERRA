#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common_utils import load_data, COLORS, MARKERS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, model_names = load_data(args.csv_file)

    fig, ax = plt.subplots(figsize=(7, 7))

    min_val = np.nanmin(df[["obs"] + model_names].values)
    max_val = np.nanmax(df[["obs"] + model_names].values)

    for model_name in model_names:
        valid = df[["obs", model_name]].dropna()

        ax.scatter(
            valid["obs"],
            valid[model_name],
            s=28,
            alpha=0.6,
            marker=MARKERS.get(model_name, "o"),
            color=COLORS.get(model_name),
            edgecolors="none",
            label=model_name.upper(),
        )

    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5)
    ax.set_xlabel("OBS T2M (K)")
    ax.set_ylabel("Model T2M (K)")
    ax.set_title("Model temperature versus observations")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_scatter_model_vs_OBS.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

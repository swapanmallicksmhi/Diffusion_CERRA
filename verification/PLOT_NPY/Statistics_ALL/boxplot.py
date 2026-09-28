#!/usr/bin/env python3

import os
import argparse
import matplotlib.pyplot as plt
from common_utils import load_data, get_errors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, model_names = load_data(args.csv_file)
    errors_dict = get_errors(df, model_names)

    labels = [k.upper() for k in errors_dict.keys()]
    data = [errors_dict[k] for k in errors_dict.keys()]

    fig, ax = plt.subplots(figsize=(7, 6))

    box = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        patch_artist=True,
        widths=0.55,
    )

    colors = ["#1f77b4", "#d62728"]

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    ax.axhline(0, linestyle="--", linewidth=1.5, color="black")
    ax.set_ylabel("Model - OBS error (K)")
    ax.set_title("Error distribution comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_boxplot_error.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

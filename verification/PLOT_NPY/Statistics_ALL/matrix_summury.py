#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common_utils import load_data, compute_metrics, COLORS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, model_names = load_data(args.csv_file)

    all_stats = []

    for model_name in model_names:
        valid = df[["obs", model_name]].dropna()
        metrics, _, _, _ = compute_metrics(
            valid["obs"].values,
            valid[model_name].values,
        )
        metrics["Model"] = model_name
        all_stats.append(metrics)

    stats_df = pd.DataFrame(all_stats).set_index("Model")

    bar_metrics = ["Bias", "MAE", "RMSE", "Std_Error"]
    width = 0.38
    x_bar = np.arange(len(bar_metrics))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(stats_df.index):
        offset = (i - 0.5) * width if len(stats_df.index) == 2 else 0

        ax1.bar(
            x_bar + offset,
            stats_df.loc[model_name, bar_metrics],
            width=width,
            label=model_name.upper(),
            color=COLORS.get(model_name),
            alpha=0.85,
        )

    ax1.axhline(0, linestyle="--", linewidth=1.2, color="black")
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(bar_metrics)
    ax1.set_ylabel("Error statistics (K)")
    ax1.set_title("Summary verification statistics")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()

    for model_name in stats_df.index:
        ax2.scatter(
            len(bar_metrics) + 0.3,
            stats_df.loc[model_name, "Pearson_R"],
            s=90,
            color=COLORS.get(model_name),
            marker="D",
        )

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Pearson correlation")
    ax2.set_yticks(np.arange(0, 1.1, 0.2))

    ax1.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_metric_summary.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

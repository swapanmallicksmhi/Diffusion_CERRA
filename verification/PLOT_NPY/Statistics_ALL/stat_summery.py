#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from common_utils import load_data, compute_metrics


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
    stats_df.to_csv(os.path.join(args.output_dir, "combined_statistics_summary.csv"))

    print("Saved combined_statistics_summary.csv")


if __name__ == "__main__":
    main()

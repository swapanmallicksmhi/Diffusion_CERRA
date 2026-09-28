#!/usr/bin/env python3

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common_utils import load_data, get_errors, COLORS


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

    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, error in errors_dict.items():
        abs_error = np.sort(np.abs(error))
        cdf = np.arange(1, len(abs_error) + 1) / len(abs_error)

        ax.plot(
            abs_error,
            cdf,
            linewidth=2.5,
            color=COLORS.get(model_name),
            label=model_name.upper(),
        )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Absolute error |Model - OBS| (K)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Cumulative distribution of absolute error")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_absolute_error_cdf.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

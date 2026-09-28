#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils import load_data, get_errors, COLORS

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
    bins = np.arange(-10, 10.5, 0.1)
    for model_name, error in errors_dict.items():
        ax.hist(
            error,
            bins=bins,
            density=True,
            alpha=0.35,
            edgecolor="black",
            label=model_name.upper(),
            color=COLORS.get(model_name),
        )
        if len(error) > 2:
            kde = gaussian_kde(error)
            x = np.linspace(-10, 10, 200)
            ax.plot(x, kde(x), linewidth=2.5, color=COLORS.get(model_name))
        #ax.axvline(
        #    np.mean(error),
        #    linestyle="-",
        #    linewidth=2,
        #    color=COLORS.get(model_name),
        #)
    ax.axvline(0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xlim(-10, 10)
    ax.set_xlabel("(OBS- MODEL in K)")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "error_histogram.png"), dpi=300)
    plt.close()
#
if __name__ == "__main__":
    main()

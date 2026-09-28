#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, gaussian_kde

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create combined publication-quality statistics and plots for CERRA and NPY"
    )
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()

def compute_metrics(obs, model):
    mask = np.isfinite(obs) & np.isfinite(model)
    obs = obs[mask]
    model = model[mask]

    error = model - obs
    abs_error = np.abs(error)

    metrics = {
        "N": len(error),
        "Mean_OBS": np.mean(obs),
        "Mean_MODEL": np.mean(model),
        "Bias": np.mean(error),
        "MAE": np.mean(abs_error),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "Median_Error": np.median(error),
        "Std_Error": np.std(error),
        "Min_Error": np.min(error),
        "Max_Error": np.max(error),
        "Q05_Error": np.percentile(error, 5),
        "Q25_Error": np.percentile(error, 25),
        "Q75_Error": np.percentile(error, 75),
        "Q95_Error": np.percentile(error, 95),
        "Pearson_R": pearsonr(obs, model)[0] if len(error) > 1 else np.nan,
        "Spearman_R": spearmanr(obs, model)[0] if len(error) > 1 else np.nan,
    }

    return metrics, error, obs, model

def plot_combined_error_histogram(errors_dict, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    bins = np.arange(-10, 10.5, 0.5)
    colors = {"cerra": "#1f77b4", "ML": "#d62728"}
    for model_name, error in errors_dict.items():
        ax.hist(
            error,
            bins=bins,
            density=True,
            alpha=0.35,
            edgecolor="black",
            label=model_name.upper(),
            color=colors.get(model_name, None),
        )
        if len(error) > 2:
            kde = gaussian_kde(error)
            x = np.linspace(-10, 10, 400)
            ax.plot(
                x,
                kde(x),
                linewidth=2.5,
                color=colors.get(model_name, None),
            )
        ax.axvline(
            np.mean(error),
            linestyle="-",
            linewidth=2,
            color=colors.get(model_name, None),
        )
    ax.axvline(0, linestyle="--", linewidth=1.5, color="black")
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Model - OBS error (K)")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    out_file = os.path.join(output_dir, "combined_error_histogram.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def plot_combined_scatter(df, model_names, output_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"cerra": "#1f77b4", "ML": "#d62728"}
    markers = {"cerra": "o", "ML": "s"}
    min_val = np.nanmin(df[["obs"] + model_names].values)
    max_val = np.nanmax(df[["obs"] + model_names].values)

    for model_name in model_names:
        valid = df[["obs", model_name]].dropna()
        ax.scatter(
            valid["obs"],
            valid[model_name],
            s=28,
            alpha=0.6,
            marker=markers.get(model_name, "o"),
            color=colors.get(model_name, None),
            edgecolors="none",
            label=model_name.upper(),
        )
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5)
    ax.set_xlabel("OBS T2M (K)")
    ax.set_ylabel("Model T2M (K)")
    ax.set_title("Model temperature versus observations")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)
    out_file = os.path.join(output_dir, "combined_scatter_model_vs_OBS.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
def plot_combined_boxplot(errors_dict, output_dir):
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
    out_file = os.path.join(output_dir, "combined_boxplot_error.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def plot_combined_time_mean_error(df, model_names, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"cerra": "#1f77b4", "ML": "#d62728"}
    markers = {"cerra": "o", "ML": "s"}
    df_tmp = df.copy()
    df_tmp["time"] = pd.to_datetime(df_tmp["time"])
    for model_name in model_names:
        df_tmp[f"error_{model_name}"] = df_tmp[model_name] - df_tmp["obs"]
        df_time = (
            df_tmp.groupby("time")[f"error_{model_name}"]
            .mean()
            .reset_index()
        )
        ax.plot(
            df_time["time"],
            df_time[f"error_{model_name}"],
            marker=markers.get(model_name, "o"),
            linewidth=2,
            markersize=5,
            color=colors.get(model_name, None),
            label=model_name.upper(),
        )

    ax.axhline(0, linestyle="--", linewidth=1.5, color="black")

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean model - OBS error (K)")
    ax.set_title("Daily 12 UTC mean error")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.xticks(rotation=45)

    out_file = os.path.join(output_dir, "combined_time_mean_error.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_combined_station_rmse(df, model_names, output_dir):
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

    colors = {"cerra": "#1f77b4", "ML": "#d62728"}

    for i, model_name in enumerate(model_names):
        offset = (i - 0.5) * width if len(model_names) == 2 else 0

        ax.bar(
            x + offset,
            df_rmse[f"rmse_{model_name}"],
            width=width,
            label=model_name.upper(),
            color=colors.get(model_name, None),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_rmse["wmo_id"].astype(int).astype(str), rotation=90)

    ax.set_xlabel("WMO station ID")
    ax.set_ylabel("RMSE (K)")
    ax.set_title("Station-wise RMSE")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    out_file = os.path.join(output_dir, "combined_station_rmse.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_combined_station_bias(df, model_names, output_dir):
    station_stats = []

    for model_name in model_names:
        df_tmp = df.copy()
        df_tmp[f"bias_{model_name}"] = df_tmp[model_name] - df_tmp["obs"]

        df_station = (
            df_tmp.groupby("wmo_id")[f"bias_{model_name}"]
            .mean()
            .reset_index()
        )

        station_stats.append(df_station)

    df_bias = station_stats[0]

    for item in station_stats[1:]:
        df_bias = df_bias.merge(item, on="wmo_id", how="outer")

    df_bias = df_bias.sort_values(f"bias_{model_names[0]}")

    x = np.arange(len(df_bias))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {"cerra": "#1f77b4", "ML": "#d62728"}

    for i, model_name in enumerate(model_names):
        offset = (i - 0.5) * width if len(model_names) == 2 else 0

        ax.bar(
            x + offset,
            df_bias[f"bias_{model_name}"],
            width=width,
            label=model_name.upper(),
            color=colors.get(model_name, None),
            alpha=0.85,
        )

    ax.axhline(0, linestyle="--", linewidth=1.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(df_bias["wmo_id"].astype(int).astype(str), rotation=90)

    ax.set_xlabel("WMO station ID")
    ax.set_ylabel("Mean bias (K)")
    ax.set_title("Station-wise mean bias")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    out_file = os.path.join(output_dir, "combined_station_bias.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_combined_absolute_error_cdf(errors_dict, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"cerra": "#1f77b4", "ML": "#d62728"}

    for model_name, error in errors_dict.items():
        abs_error = np.sort(np.abs(error))
        cdf = np.arange(1, len(abs_error) + 1) / len(abs_error)

        ax.plot(
            abs_error,
            cdf,
            linewidth=2.5,
            color=colors.get(model_name, None),
            label=model_name.upper(),
        )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.01)

    ax.set_xlabel("Absolute error |Model - OBS| (K)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Cumulative distribution of absolute error")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.4)

    out_file = os.path.join(output_dir, "combined_absolute_error_cdf.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_combined_metric_summary(stats_df, output_dir):
    metrics = ["Bias", "MAE", "RMSE", "Std_Error", "Pearson_R"]

    stats_plot = stats_df[metrics].copy()

    x = np.arange(len(metrics))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = {"cerra": "#1f77b4", "ML": "#d62728"}

    bar_metrics = ["Bias", "MAE", "RMSE", "Std_Error"]
    x_bar = np.arange(len(bar_metrics))

    for i, model_name in enumerate(stats_plot.index):
        offset = (i - 0.5) * width if len(stats_plot.index) == 2 else 0

        ax1.bar(
            x_bar + offset,
            stats_plot.loc[model_name, bar_metrics],
            width=width,
            label=model_name.upper(),
            color=colors.get(model_name, None),
            alpha=0.85,
        )

    ax1.axhline(0, linestyle="--", linewidth=1.2, color="black")
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(bar_metrics)
    ax1.set_ylabel("Error statistics (K)")
    ax1.set_title("Summary verification statistics")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()

    for model_name in stats_plot.index:
        ax2.scatter(
            len(bar_metrics) + 0.3,
            stats_plot.loc[model_name, "Pearson_R"],
            s=90,
            color=colors.get(model_name, None),
            marker="D",
        )

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Pearson correlation")
    ax2.set_yticks(np.arange(0, 1.1, 0.2))

    ax1.legend(frameon=False, loc="upper left")

    out_file = os.path.join(output_dir, "combined_metric_summary.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_file)
    df["time"] = pd.to_datetime(df["time"])

    required_cols = ["wmo_id", "time", "lat", "lon", "obs"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    model_names = []

    for col in ["cerra", "ML"]:
        if col in df.columns:
            model_names.append(col)

    if len(model_names) == 0:
        raise ValueError("No model columns found. Expected columns: cerra and/or ML")

    all_stats = []
    errors_dict = {}

    for model_name in model_names:
        valid = df[["obs", model_name]].dropna()

        metrics, error, obs, model = compute_metrics(
            valid["obs"].values,
            valid[model_name].values,
        )

        metrics["Model"] = model_name
        all_stats.append(metrics)
        errors_dict[model_name] = error

    stats_df = pd.DataFrame(all_stats).set_index("Model")
    stats_df.to_csv(os.path.join(args.output_dir, "combined_statistics_summary.csv"))

    plot_combined_error_histogram(errors_dict, args.output_dir)
    plot_combined_scatter(df, model_names, args.output_dir)
    plot_combined_boxplot(errors_dict, args.output_dir)
    plot_combined_time_mean_error(df, model_names, args.output_dir)
    plot_combined_station_rmse(df, model_names, args.output_dir)
    plot_combined_station_bias(df, model_names, args.output_dir)
    plot_combined_absolute_error_cdf(errors_dict, args.output_dir)
    plot_combined_metric_summary(stats_df, args.output_dir)

    print("Saved combined statistics and figures in:")
    print(args.output_dir)


if __name__ == "__main__":
    main()

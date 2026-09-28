import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

COLORS = {"cerra": "#1f77b4", "ML": "#d62728"}
MARKERS = {"cerra": "o", "ML": "s"}

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"])

    required_cols = ["wmo_id", "time", "lat", "lon", "obs"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
#
    model_names = []
    for col in ["cerra", "ML"]:
        if col in df.columns:
            model_names.append(col)

    if len(model_names) == 0:
        raise ValueError("No model columns found")
    return df, model_names
#
def compute_metrics(obs, model):
    mask = np.isfinite(obs) & np.isfinite(model)
    obs = obs[mask]
    model = model[mask]
    error = obs -model
    #error = model - obs
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
def get_errors(df, model_names):
    errors_dict = {}
    for model_name in model_names:
        valid = df[["obs", model_name]].dropna()
        _, error, _, _ = compute_metrics(valid["obs"].values, valid[model_name].values,)
        errors_dict[model_name] = error
    return errors_dict

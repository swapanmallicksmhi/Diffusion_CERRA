import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def sample_normal(shape):
    return np.random.normal(0.0, 1.0, shape).astype(np.float32)


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)


def sigmoid_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    x = np.linspace(-6, 6, T)
    betas = 1.0 / (1.0 + np.exp(-x))
    betas = betas * (beta_end - beta_start) + beta_start
    return betas.astype(np.float32)


def quadratic_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(
        np.sqrt(beta_start),
        np.sqrt(beta_end),
        T,
        dtype=np.float32
    ) ** 2


def cosine_beta_schedule(T, s=0.008):
    x = np.linspace(0, T - 1, T)
    alpha_bar = np.cos(((x / (T - 1)) + s) / (1 + s) * np.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]

    betas = np.zeros(T, dtype=np.float32)
    betas[0] = 1e-4
    betas[1:] = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])

    return np.clip(betas, 1e-4, 0.999)


def prepare_terms(betas):
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    sqrt_alpha_bar = np.sqrt(alpha_bar).reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1, 1)

    return sqrt_alpha_bar, sqrt_one_minus_alpha_bar


def q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise):
    return sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise


def load_zarr_samples(zarr_path, var_name="t2m_cerra_mean", n_samples=100):
    ds = xr.open_zarr(zarr_path)

    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in the Zarr dataset.")

    if "valid_time" not in ds.sizes:
        raise KeyError("Dimension 'valid_time' not found in the Zarr dataset.")

    n_total = ds.sizes["valid_time"]

    if n_samples > n_total:
        raise ValueError(
            f"Requested {n_samples} samples, but dataset has only {n_total} time steps."
        )

    x0 = ds[var_name].isel(valid_time=slice(0, n_samples)).values.astype(np.float32)

    if x0.ndim != 3:
        raise ValueError(
            f"Expected data shape (sample, y, x), but got {x0.shape}"
        )

    return x0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zarr",
        default="/scratch/swe4281/CERRA_DATA2026/NEW_PAPER_JUNE/all_zarr-out_2015-2023.zar"
    )
    parser.add_argument("--var", default="t2m_cerra_mean")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--save", default="Diffusion_Line.png")
    args = parser.parse_args()

    np.random.seed(42)

    T = 4000
    step_stride = 20
    timesteps = np.arange(0, T + 1, step_stride)

    x0 = load_zarr_samples(
        zarr_path=args.zarr,
        var_name=args.var,
        n_samples=args.n_samples
    )

    noise = sample_normal(x0.shape)

    schedules = {
        "Linear": linear_beta_schedule(T + 1),
        "Sigmoid": sigmoid_beta_schedule(T + 1),
        "Cosine": cosine_beta_schedule(T + 1),
        #"Quadratic": quadratic_beta_schedule(T + 1),
    }

    stats = {}

    for name, betas in schedules.items():
        sqrt_ab, sqrt_1mab = prepare_terms(betas)

        means = []
        stds = []

        for t in timesteps:
            xt = q_sample(x0, t, sqrt_ab, sqrt_1mab, noise)
            values = xt.flatten()

            means.append(np.mean(values))
            stds.append(np.std(values))

        stats[name] = {
            "mean": np.array(means),
            "std": np.array(stds),
        }

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for name, values in stats.items():
        axes[0].plot(
            timesteps,
            values["mean"],
            linewidth=2.2,
            label=name
        )

    axes[0].set_title("Mean")
    axes[0].set_xlabel("Diffusion Step")
    axes[0].set_ylabel("Mean value")
    axes[0].set_xlim(0, T)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=False)

    for name, values in stats.items():
        axes[1].plot(
            timesteps,
            values["std"],
            linewidth=2.2,
            label=name
        )

    axes[1].set_title("Standard Deviation")
    axes[1].set_xlabel("Diffusion Step")
    axes[1].set_ylabel("Standard deviation")
    axes[1].set_xlim(0, T)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Evolution of Pixel Statistics Under Diffusion Noise Schedules",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(args.save, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

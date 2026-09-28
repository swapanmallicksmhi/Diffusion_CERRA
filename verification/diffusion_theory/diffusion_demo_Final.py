import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def load_zarr_data(zarr_path, var_name="t2m_cerra_mean", n_samples=100):
    ds = xr.open_zarr(zarr_path)

    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in Zarr file.")

    x0 = ds[var_name].isel(valid_time=slice(0, n_samples)).values.astype(np.float32)

    # Add channel dimension: (N, H, W) -> (N, H, W, 1)
    x0 = x0[..., None]

    # Normalize to [-1, 1] for diffusion visualization
    xmin = np.nanmin(x0)
    xmax = np.nanmax(x0)
    x0 = 2.0 * (x0 - xmin) / (xmax - xmin) - 1.0

    return x0


def to_display(x):
    x = (x + 1.0) / 2.0
    return np.clip(x.squeeze(), 0, 1)


def sample_normal(shape, mean=0.0, std=1.0):
    return np.random.normal(mean, std, shape).astype(np.float32)


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)


def q_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, noise):
    """
    Forward diffusion:
    """
    xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise
    return xt


def oracle_reverse_sample(xt, x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar):
    """
    """
    if t == 0:
        return x0

    predicted_noise = (xt - sqrt_alpha_bar[t] * x0) / sqrt_one_minus_alpha_bar[t]

    x_denoised = (
        xt - sqrt_one_minus_alpha_bar[t] * predicted_noise
    ) / sqrt_alpha_bar[t]

    return np.clip(x_denoised, -1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zarr",
        default="/scratch/swe4281/CERRA_DATA2026/NEW_PAPER_JUNE/all_zarr-out_2015-2023.zar"
    )
    parser.add_argument("--var", default="t2m_cerra_mean")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    T = 4000
    timesteps = [0, 10, 50, 100, 500, 1000, 4000]

    x0 = load_zarr_data(
        zarr_path=args.zarr,
        var_name=args.var,
        n_samples=args.n_samples
    )

    betas = linear_beta_schedule(T + 1)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    sqrt_alpha_bar = np.sqrt(alpha_bar).reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1, 1)

    noise = sample_normal(x0.shape)

    forward_images = []

    for t in timesteps:
        xt = q_sample(
            x0,
            t,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
            noise
        )
        forward_images.append(xt[0])

    xT = q_sample(
        x0,
        T,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        noise
    )

    reverse_images = []

    for t in timesteps[::-1]:
        xr = oracle_reverse_sample(
            xT,
            x0,
            t,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar
        )
        reverse_images.append(xr[0])

    reverse_images = reverse_images[::-1]

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

    fig, axes = plt.subplots(2, 7, figsize=(20, 6))

    for col, t in enumerate(timesteps):
        xt = forward_images[col]

        # Row 1: pixel-value distribution of x_t
        pixel_values = xt.flatten()

        axes[0, col].hist(
            pixel_values,
            bins=200,
            density=True,
            color="skyblue",
            edgecolor="navy",
            linewidth=0.4
        )
        axes[0, col].set_title(f"Diffusion Step={t}")
        axes[0, col].set_xlim(-3, 3)
        axes[0, col].set_ylim(0, 1)

        mean = np.mean(pixel_values)
        std = np.std(pixel_values)

        axes[0, col].text(
            0.05,
            0.95,
            f"Mean = {mean:.2f}\nStd = {std:.2f}",
            transform=axes[0, col].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.1)
        )

        # Row 2: forward noising process
        axes[1, col].imshow(to_display(forward_images[col]))
        #axes[1, col].set_title(f"Forward  Step={t}")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Data distribution")
    axes[1, 0].set_ylabel("Forward process")

    plt.tight_layout()
    plt.savefig('Diffusion_Forward.png', bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

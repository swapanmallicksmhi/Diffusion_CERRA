# -*- coding: utf-8 -*-
"""
ScoreDiffusion_process_DDPM.py
Diffusion Probabilistic Model / DDPM forward-noising version
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Load data
# =========================
data = np.load("/scratch/swe4281/CERRA_DATA2026/NEW_PAPER_JUNE/data_theory_SDM.npz", allow_pickle=True)

candidate_mean = data["candidate_mean"]
candidate_std = data["candidate_std"]
candidate_times = data["candidate_times"]

batch_mean = data["batch_mean"]
batch_std = data["batch_std"]
batch_times = data["batch_times"]

lat = data["latitude"]
lon = data["longitude"]

print("candidate_mean:", candidate_mean.shape)
print("candidate_std :", candidate_std.shape)
print("batch_mean    :", batch_mean.shape)
print("batch_std     :", batch_std.shape)
print("lat/lon       :", lat.shape, lon.shape)


# =========================
# Helper functions
# =========================
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Create linear beta schedule for DDPM."""
    return np.linspace(beta_start, beta_end, T, dtype=np.float64)


def ddpm_forward_sample(x0, t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, rng):
    """
    DDPM forward process:

        x_t = sqrt(alpha_bar_t) * x_0
              + sqrt(1 - alpha_bar_t) * epsilon

    where epsilon ~ N(0, I)
    """
    noise = rng.normal(loc=0.0, scale=1.0, size=x0.shape)

    xt = (
        sqrt_alpha_bar[t] * x0
        + sqrt_one_minus_alpha_bar[t] * noise
    )

    return xt


def hist_pdf(values, bins):
    """Compute normalized histogram/PDF."""
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return hist, centers


def smooth_curve(y, win=7):
    """Smooth a 1D curve using a moving-average window."""
    if win <= 1:
        return y

    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="same")


def build_dataset_density_ddpm(images, time_indices, bins, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, seed=42):
    """Build density evolution for dataset under DDPM forward diffusion."""
    rng = np.random.default_rng(seed)
    density = []

    for t in time_indices:
        noisy = ddpm_forward_sample(
            images,
            t,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
            rng,
        )

        vals = noisy.reshape(-1)

        hist, _ = np.histogram(vals, bins=bins, density=True)
        density.append(hist)

    return np.asarray(density)


# =========================
# Configuration
# =========================
sample_idx = 17

if sample_idx >= candidate_mean.shape[0]:
    raise IndexError(
        f"sample_idx={sample_idx} is outside candidate_mean size "
        f"{candidate_mean.shape[0]}"
    )

single_img = candidate_mean[sample_idx]

# DDPM time steps
T = 1000

betas = linear_beta_schedule(
    T=T,
    beta_start=1e-4,
    beta_end=0.02,
)

alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)

sqrt_alpha_bar = np.sqrt(alpha_bar)
sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar)

# Four DDPM steps for panel (a)
time_single = np.array(
    [
        0,
        int(0.25 * (T - 1)),
        int(0.50 * (T - 1)),
        T - 1,
    ],
    dtype=int,
)

# Full DDPM grid for panel (b)
time_grid = np.linspace(0, T - 1, 160, dtype=int)

# For x-axis, use DDPM effective noise level:
# sigma_t = sqrt(1 - alpha_bar_t)
sigma_single = sqrt_one_minus_alpha_bar[time_single]
sigma_grid = sqrt_one_minus_alpha_bar[time_grid]

sigma_max = float(sigma_grid[-1])

vmin_data = float(np.nanmin(batch_mean))
vmax_data = float(np.nanmax(batch_mean))

x_min = vmin_data - 2.5 * sigma_max
x_max = vmax_data + 2.5 * sigma_max

bins = np.linspace(x_min, x_max, 260)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

density_all = build_dataset_density_ddpm(
    images=batch_mean,
    time_indices=time_grid,
    bins=bins,
    sqrt_alpha_bar=sqrt_alpha_bar,
    sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar,
    seed=42,
)

log_density = np.log(density_all + 1e-12)
log_density_masked = np.ma.masked_where(density_all <= 1e-12, log_density)

valid = log_density_masked.compressed()

if valid.size == 0:
    raise ValueError("No valid density values found for plotting.")

vmin_plot = np.percentile(valid, 1)
vmax_plot = np.percentile(valid, 99.5)


# =========================
# Plot
# =========================
fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=220)
fig.subplots_adjust(wspace=0.12, top=0.88, bottom=0.14)


# =========================
# Panel (a): single image
# =========================
ax = axes[0]

if sigma_single[1] > 0:
    pdf_width = 0.38 * (sigma_single[1] - sigma_single[0])
else:
    pdf_width = 1.0

colors = ["black", "#1f77b4", "#2ca02c", "#ff7f0e"]

for i, (t, sigma, color) in enumerate(zip(time_single, sigma_single, colors)):
    rng = np.random.default_rng(100 + i)

    noisy_img = ddpm_forward_sample(
        single_img,
        t,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        rng,
    )

    pdf, centers = hist_pdf(noisy_img.ravel(), bins)
    pdf = smooth_curve(pdf, win=9)

    if np.nanmax(pdf) > 0:
        x_curve = sigma + pdf_width * (pdf / np.nanmax(pdf))
    else:
        x_curve = np.full_like(pdf, sigma)

    ax.plot(x_curve, centers, color=color, lw=1.8)
    ax.fill_betweenx(centers, sigma, x_curve, color=color, alpha=0.18)
    ax.axvline(sigma, color=color, lw=1.2, alpha=0.8, zorder=0)


# Insets
inset_w = 0.18
inset_h = 0.18
x_positions = [0.09, 0.293, 0.495, 0.80]

for i, (t, sigma, color, x0) in enumerate(zip(time_single, sigma_single, colors, x_positions)):
    rng = np.random.default_rng(100 + i)

    noisy_img = ddpm_forward_sample(
        single_img,
        t,
        sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar,
        rng,
    )

    axins = ax.inset_axes([x0, 0.80, inset_w, inset_h])
    axins.imshow(noisy_img, cmap="turbo", origin="lower")
    axins.set_xticks([])
    axins.set_yticks([])

    for spine in axins.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.0)


ax.set_xlim(
    sigma_single[0] - 0.12 * sigma_max,
    sigma_single[-1] + 0.12 * sigma_max,
)
ax.set_ylim(x_min, x_max)

ax.set_xlabel(r"Effective DDPM noise level $\sqrt{1-\bar{\alpha}_t}$", fontsize=13)
ax.set_ylabel("Pixel value / temperature (K)", fontsize=13)
ax.set_title("(a) A single image", fontsize=15)

ax.text(
    sigma_single[0],
    -0.10,
    "No noise",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=13,
)

ax.text(
    sigma_single[-1],
    -0.10,
    "All noise",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=13,
)


# =========================
# Panel (b): training dataset
# =========================
ax = axes[1]

cmap = plt.cm.RdYlGn_r.copy()
cmap.set_bad("white")

im = ax.imshow(
    log_density_masked.T,
    origin="lower",
    aspect="auto",
    extent=[
        sigma_grid.min(),
        sigma_grid.max(),
        bin_centers.min(),
        bin_centers.max(),
    ],
    cmap=cmap,
    vmin=vmin_plot,
    vmax=vmax_plot,
)

ax.set_facecolor("white")

ax.set_xlabel(r"Effective DDPM noise level $\sqrt{1-\bar{\alpha}_t}$", fontsize=13)
ax.set_ylabel("Pixel value / temperature (K)", fontsize=13)
ax.set_title("(b) Images in the training dataset", fontsize=15)

ax.text(
    sigma_grid[0],
    -0.10,
    "No noise",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=14,
)

ax.text(
    sigma_grid[-1],
    -0.10,
    "All noise",
    transform=ax.get_xaxis_transform(),
    ha="center",
    va="top",
    fontsize=13,
)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label(r"log $p(x_t)$", fontsize=13)


# =========================
# Save and show
# =========================
plt.savefig("ddpm_density_process.png", dpi=100, bbox_inches="tight")
# plt.show()

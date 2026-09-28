import xarray as xr
import numpy as np
from pathlib import Path

# =========================
# User settings
# =========================
ZARR_PATH = "/scratch/swe4281/CERRA_DATA2026/NEW_PAPER_JUNE/all_zarr-out_2015-2023.zar"
OUT_PATH = "/scratch/swe4281/CERRA_DATA2026/NEW_PAPER_JUNE/data_theory_SDM.npz"

# the left-panel examples
N_CANDIDATES = 50

# the right-panel distribution
N_BATCH = 200

# How to choose
# "uniform": evenly spaced across whole training period
# "random": random but reproducible
SAMPLING_MODE = "uniform"

# only used when SAMPLING_MODE = "random"
RANDOM_SEED = 42


# =========================
# Helper functions
# =========================
def choose_indices(n_total: int, n_pick: int, mode: str = "uniform", seed: int = 42) -> np.ndarray:
    if n_pick > n_total:
        raise ValueError(f"Requested {n_pick} samples, but dataset only has {n_total} time steps.")

    if mode == "uniform":
        idx = np.linspace(0, n_total - 1, n_pick, dtype=int)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=n_pick, replace=False))
    else:
        raise ValueError("mode must be 'uniform' or 'random'")

    return idx


def to_numpy_time(arr) -> np.ndarray:
    return np.asarray(arr.values)


# =========================
# Main
# =========================
def main():
    print(f"Opening zarr dataset:\n  {ZARR_PATH}")
    ds = xr.open_zarr(ZARR_PATH)

    print("\nDataset opened successfully.")
    print(ds)

    required_vars = [
        "t2m_cerra_mean",
        "t2m_cerra_std",
        "latitude",
        "longitude",
        "valid_time",
    ]
    for v in required_vars:
        if v not in ds and v not in ds.coords:
            raise KeyError(f"Required variable/coord '{v}' not found in dataset.")

    # Basic checks
    n_time = ds.sizes["valid_time"]
    print(f"\nTotal valid_time steps: {n_time}")

    # Choose indices
    candidate_idx = choose_indices(
        n_total=n_time,
        n_pick=N_CANDIDATES,
        mode=SAMPLING_MODE,
        seed=RANDOM_SEED,
    )
    batch_idx = choose_indices(
        n_total=n_time,
        n_pick=N_BATCH,
        mode=SAMPLING_MODE,
        seed=RANDOM_SEED + 1,
    )

    print(f"\nCandidate indices ({len(candidate_idx)}):")
    print(candidate_idx)

    print(f"\nBatch indices ({len(batch_idx)}):")
    print(batch_idx)

    # Read candidate fields
    print("\nLoading candidate mean/std fields...")
    candidate_mean = ds["t2m_cerra_mean"].isel(valid_time=candidate_idx).values
    candidate_std = ds["t2m_cerra_std"].isel(valid_time=candidate_idx).values
    candidate_times = to_numpy_time(ds["valid_time"].isel(valid_time=candidate_idx))

    # Read batch fields
    print("Loading batch mean/std fields...")
    batch_mean = ds["t2m_cerra_mean"].isel(valid_time=batch_idx).values
    batch_std = ds["t2m_cerra_std"].isel(valid_time=batch_idx).values
    batch_times = to_numpy_time(ds["valid_time"].isel(valid_time=batch_idx))

    # Read coordinates
    print("Loading latitude/longitude...")
    latitude = ds["latitude"].values
    longitude = ds["longitude"].values

    # Save
    out_path = Path(OUT_PATH)
    print(f"\nSaving extracted data to:\n  {out_path.resolve()}")

    np.savez_compressed(
        out_path,
        candidate_mean=candidate_mean,      # shape: (N_CANDIDATES, 256, 256)
        candidate_std=candidate_std,        # shape: (N_CANDIDATES, 256, 256)
        candidate_times=candidate_times,    # shape: (N_CANDIDATES,)
        candidate_idx=candidate_idx,        # shape: (N_CANDIDATES,)

        batch_mean=batch_mean,              # shape: (N_BATCH, 256, 256)
        batch_std=batch_std,                # shape: (N_BATCH, 256, 256)
        batch_times=batch_times,            # shape: (N_BATCH,)
        batch_idx=batch_idx,                # shape: (N_BATCH,)

        latitude=latitude,                  # shape: (256, 256)
        longitude=longitude,                # shape: (256, 256)
    )

    print("\nDone.")
    print(f"Saved file size should be manageable for local download.")

    print("\nSaved arrays summary:")
    print(f"  candidate_mean : {candidate_mean.shape}, dtype={candidate_mean.dtype}")
    print(f"  candidate_std  : {candidate_std.shape}, dtype={candidate_std.dtype}")
    print(f"  batch_mean     : {batch_mean.shape}, dtype={batch_mean.dtype}")
    print(f"  batch_std      : {batch_std.shape}, dtype={batch_std.dtype}")
    print(f"  latitude       : {latitude.shape}, dtype={latitude.dtype}")
    print(f"  longitude      : {longitude.shape}, dtype={longitude.dtype}")

    print("\nCandidate times:")
    for i, t in enumerate(candidate_times):
        print(f"  {i:02d}: idx={candidate_idx[i]:5d}, time={str(t)}")


if __name__ == "__main__":
    main()

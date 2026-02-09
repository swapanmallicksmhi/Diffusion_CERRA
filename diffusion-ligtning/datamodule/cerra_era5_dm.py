import torch
import xarray as xr
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class ZarrCerraEra5Dataset(Dataset):
    """
    PyTorch Dataset that wraps the Xarray/Zarr store.
    """
    def __init__(self, ds, include_members=False):
        """
        Args:
            ds (xr.Dataset): The Xarray dataset slice (e.g., train split).
            include_members (bool): Whether to load ensemble member data.
        """
        self.ds = ds
        self.include_members = include_members
        
        # Determine available variables to avoid KeyErrors
        self.has_cerra_members = 't2m_cerra_members' in ds
        self.has_era5_members = 't2m_era5_members' in ds

    def __len__(self):
        return len(self.ds.valid_time)

    def __getitem__(self, idx):
        # Select specific time step
        # We load .values here to convert Dask arrays to NumPy
        item = self.ds.isel(valid_time=idx)

        # Initialize dictionary
        data = {}

        # --- Load Mean & Std (Shape: [1, 128, 128]) ---
        # We perform .astype(np.float32) to ensure consistency for PyTorch
        
        # CERRA
        data['cerra_mean'] = torch.from_numpy(
            item.t2m_cerra_mean.values.astype(np.float32)
        ).unsqueeze(0) 
        
        data['cerra_std'] = torch.from_numpy(
            item.t2m_cerra_std.values.astype(np.float32)
        ).unsqueeze(0)

        # ERA5
        data['era5_mean'] = torch.from_numpy(
            item.t2m_era5_mean.values.astype(np.float32)
        ).unsqueeze(0)
        
        data['era5_std'] = torch.from_numpy(
            item.t2m_era5_std.values.astype(np.float32)
        ).unsqueeze(0)

        # --- Load Members (Optional) (Shape: [10, 128, 128]) ---
        if self.include_members:
            if self.has_cerra_members:
                data['cerra_members'] = torch.from_numpy(
                    item.t2m_cerra_members.values.astype(np.float32)
                )
            
            if self.has_era5_members:
                data['era5_members'] = torch.from_numpy(
                    item.t2m_era5_members.values.astype(np.float32)
                )

        # Add metadata (optional, usually skipped in training loop for speed)
        # data['time'] = str(item.valid_time.values)

        return data


class CerraEra5rDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        zarr_path, 
        batch_size=32, 
        num_workers=4, 
        include_members=False, 
        split_ratios=(0.7, 0.15, 0.15)
    ):
        """
        Args:
            zarr_path (str): Path to output_data.zarr
            batch_size (int): Batch size for Dataloaders.
            include_members (bool): If True, loads the heavy 10-member ensemble data.
            split_ratios (tuple): (Train, Val, Test) ratios. Must sum to 1.
        """
        super().__init__()
        self.zarr_path = zarr_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.include_members = include_members
        self.split_ratios = split_ratios
        self.ds = None

    def setup(self, stage=None):
        # 1. Open Zarr Store
        # consolidated=False because of the Zarr v3 warning you saw earlier
        self.ds = xr.open_zarr(self.zarr_path, consolidated=False)

        # 2. Calculate Split Indices (Chronological Split)
        n_samples = len(self.ds.valid_time)
        n_train = int(n_samples * self.split_ratios[0])
        n_val = int(n_samples * self.split_ratios[1])
        # Test gets the remainder

        # 3. Create Datasets by slicing the Xarray object
        # Note: We slice valid_time dimension directly
        train_data = self.ds.isel(valid_time=slice(0, n_train))
        val_data = self.ds.isel(valid_time=slice(n_train, n_train + n_val))
        test_data = self.ds.isel(valid_time=slice(n_train + n_val, None))

        print(f"Data Split: Train={len(train_data.valid_time)}, Val={len(val_data.valid_time)}, Test={len(test_data.valid_time)}")

        # 4. Wrap in PyTorch Dataset
        if stage == 'fit' or stage is None:
            self.train_dataset = ZarrCerraEra5Dataset(train_data, include_members=self.include_members)
            self.val_dataset = ZarrCerraEra5Dataset(val_data, include_members=self.include_members)

        if stage == 'test' or stage is None:
            self.test_dataset = ZarrCerraEra5Dataset(test_data, include_members=self.include_members)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers, 
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, 
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, 
                          pin_memory=True)
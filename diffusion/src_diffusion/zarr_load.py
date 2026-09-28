from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import zarr
import torch
import torch as th
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os


class DateRangeSplitter:
    def __init__(
        self,
        train_range: Tuple[str, str],
        val_range: Tuple[str, str],
        test_range: Tuple[str, str],
        zarr_path: Optional[str] = None,
        zarr_mode: str = "r",
        zarr_group: Optional[str] = None,
        time_key: str = "valid_time"
    ):
        self.ranges = {
            "train": train_range,
            "val": val_range,
            "test": test_range
        }
        self.time_key = time_key
        self._validate_ranges()

        self.root = None
        if zarr_path:
            self.root = zarr.open(zarr_path, mode=zarr_mode)
            if zarr_group:
                self.root = self.root[zarr_group]

    def _validate_ranges(self):
        for split, (start, end) in self.ranges.items():
            if pd.Timestamp(start) > pd.Timestamp(end):
                raise ValueError(f"Invalid {split} range: {start} > {end}")

    def _load_timestamps_from_zarr(self) -> pd.DatetimeIndex:
        if self.root is None:
            raise ValueError("Zarr root not initialized. Provide zarr_path.")

        if self.time_key not in self.root:
            raise KeyError(f"'{self.time_key}' not found in {list(self.root.keys())}")

        time_arr = self.root[self.time_key]
        raw_values = time_arr[:]
        units_str = time_arr.attrs.get('units', 'seconds since 1970-01-01')
        unit_map = {'seconds': 's', 'minutes': 'm', 'hours': 'h', 'days': 'D'}
        unit_key = units_str.split(' ')[0].lower()
        pd_unit = unit_map.get(unit_key, 's')
        return pd.to_datetime(raw_values, unit=pd_unit, origin='unix', utc=True)

    def get_splits(self, timestamps: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None) -> Dict[str, List[int]]:
        if timestamps is None:
            timestamps = self._load_timestamps_from_zarr()

        if not isinstance(timestamps, pd.DatetimeIndex):
            timestamps = pd.to_datetime(timestamps, unit='s', utc=True)
        elif timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')

        results = {}
        for split_name, (start_str, end_str) in self.ranges.items():
            start_ts = pd.Timestamp(start_str, tz='UTC')
            end_ts = pd.Timestamp(end_str, tz='UTC')
            mask = (timestamps >= start_ts) & (timestamps <= end_ts)
            indices = np.where(mask)[0]
            results[split_name] = indices.tolist()
            print(f" {split_name.capitalize()}: {len(indices)} samples found.")

        return results


class ZarrNormalizer:
    def __init__(self, min_val: torch.Tensor, max_val: torch.Tensor):
        self.min = min_val.view(-1, 1, 1)
        self.max = max_val.view(-1, 1, 1)
        self.range = self.max - self.min
        self.range[self.range == 0] = 1.0

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return 2.0 * (tensor - self.min) / self.range - 1.0

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((tensor + 1.0) / 2.0) * self.range + self.min

class ZarrWeatherDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        indices: list,
        input_variables: list,
        target_variables: list,
        input_normalizer: Optional = None,
        target_normalizer: Optional = None,
        #image_size: int = 256
        image_size: int =1024,
        add_seasonal_cond: bool = False,
        seasonal_channels: int = 0, #seasonal_add
    ):
        self.store = zarr.open(zarr_path, mode='r')
        self.indices = indices
        self.input_variables = input_variables
        self.target_variables = target_variables
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer
        self.image_size = image_size
        self.add_seasonal_cond = add_seasonal_cond #add_seasonal
        self.seasonal_channels = seasonal_channels
        
        if add_seasonal_cond and self.seasonal_channels > 0:
            time_arr = self.store['valid_time']
            raw_times = time_arr[:]
            # CHANGED: use confirmed Unix-second valid_time
            self.times = pd.to_datetime(raw_times, unit='s', origin='unix', utc=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]

        input_data_list = []
        for var in self.input_variables:
            data = self.store[var][global_idx].astype(np.float32)
            input_data_list.append(data)

        target_data_list = []
        for var in self.target_variables:
            data = self.store[var][global_idx].astype(np.float32)
            target_data_list.append(data)

        input_tensor = torch.from_numpy(np.stack(input_data_list, axis=0)).float()
        target_tensor = torch.from_numpy(np.stack(target_data_list, axis=0)).float()

        if self.input_normalizer:
            input_tensor = self.input_normalizer.normalize(input_tensor)
        if self.target_normalizer:
            target_tensor = self.target_normalizer.normalize(target_tensor)
        
        if self.add_seasonal_cond and self.seasonal_channels > 0: #seasonal_add
            h, w = input_tensor.shape[-2], input_tensor.shape[-1]
            timestamp = self.times[global_idx]
            day_of_year = timestamp.dayofyear
            
            seasonal_features = []
            frequencies = [1, 2, 3, 4, 6, 12]
            for freq in frequencies:
                angle = 2 * np.pi * freq * (day_of_year - 1) / 365.25
                seasonal_features.append(np.sin(angle))
                seasonal_features.append(np.cos(angle))
            
            seasonal_features = seasonal_features[:self.seasonal_channels]
            seasonal_tensor = torch.tensor(seasonal_features, dtype=torch.float32)
            seasonal_expanded = seasonal_tensor.view(-1, 1, 1).expand(-1, h, w)
            input_tensor = torch.cat([input_tensor, seasonal_expanded], dim=0)

        # Return (target, condition) format for diffusion model
        return target_tensor, input_tensor


def get_vars_indices(target_vars, z):
    vars_indices = []
    all_variables = np.array(z['variable'])
    for i in range(len(target_vars)):
        for j in range(len(all_variables)):
            if target_vars[i] == all_variables[j]:
                vars_indices.append(j)
                break
    return vars_indices


def load_data(
    zarr_path: str,
    batch_size: int,
    #image_size: int = 256,
    image_size: int = 1024,
    train_range: Tuple[str, str] = ("2015-01-01", "2023-12-31"),
    val_range: Tuple[str, str] = ("2024-01-01", "2024-08-15"),
    test_range: Tuple[str, str] = ("2024-08-16", "2024-12-31"),
    input_vars: List[str] = None,
    target_vars: List[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    normalize: bool = True,
    class_cond: bool = False,
    deterministic: bool = False,
    add_seasonal_cond: bool = False,
    seasonal_channels: int = 0,
):
    print("=" * 80)
    print("Loading Zarr Data for Diffusion Training")
    print("=" * 80)

    if input_vars is None:
        input_vars = ['t2m_era5_mean', 't2m_era5_std',
                      #'d2m_era5_mean', 
                      #'sp_era5_mean',  
                      ]
    if target_vars is None:
        target_vars = ['t2m_cerra_mean', 't2m_cerra_std']

    print("\n--- Step 1: Creating Date-Based Splits ---")
    splitter = DateRangeSplitter(
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        zarr_path=zarr_path
    )
    splits = splitter.get_splits()
    train_indices = splits['train']
    print(f"Training samples: {len(train_indices)}")
    print("\n--- Step 2: Loading Statistics for Normalization ---")
    z = zarr.open(zarr_path, mode='r')
    target_vars_indices = get_vars_indices(target_vars, z)
    input_vars_indices = get_vars_indices(input_vars, z)

    min_stats = torch.from_numpy(z['min'][:].astype(np.float32))
    max_stats = torch.from_numpy(z['max'][:].astype(np.float32))

    target_min_stats = min_stats[target_vars_indices]
    target_max_stats = max_stats[target_vars_indices]
    input_min_stats = min_stats[input_vars_indices]
    input_max_stats = max_stats[input_vars_indices]
    
    print(f"Input variables: {input_vars}")
    print(f"Target variables: {target_vars}")

    print("\n--- Step 3: Initializing Normalizers ---")
    target_normalizer = ZarrNormalizer(min_val=target_min_stats, max_val=target_max_stats) if normalize else None
    input_normalizer = ZarrNormalizer(min_val=input_min_stats, max_val=input_max_stats) if normalize else None 

    print("\n--- Step 4: Creating Dataset ---")
    dataset = ZarrWeatherDataset(
        zarr_path=zarr_path,
        indices=train_indices,
        input_variables=input_vars,
        target_variables=target_vars,
        input_normalizer=input_normalizer,
        target_normalizer=target_normalizer,
        image_size=image_size,
        add_seasonal_cond=add_seasonal_cond,
        seasonal_channels=seasonal_channels, #seasonal_add
    )
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Sample shape: {dataset[0][0].shape} (target), {dataset[0][1].shape} (condition)")

    print("\n--- Step 5: Creating DataLoader ---")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    print(f"DataLoader created with batch size {batch_size}")
    print("=" * 80)

    return loader

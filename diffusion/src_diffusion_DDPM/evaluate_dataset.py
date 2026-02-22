#!/usr/bin/env python3
#
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# -------- dataset --------
class PairedImageDataset(Dataset):
    def __init__(self, era5_files, cerra_files, image_size):
        assert len(era5_files) == len(cerra_files)
        self.era5_files = era5_files
        self.cerra_files = cerra_files
        self.image_size = image_size

    def __len__(self):
        return len(self.era5_files)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.array(img).astype(np.float32)
        arr = arr / 127.5 - 1.0
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()
        return tensor

    def __getitem__(self, idx):
        return (
            self._load_image(self.era5_files[idx]),
            self._load_image(self.cerra_files[idx]),
            os.path.basename(self.era5_files[idx]),
        )

def collate_with_fnames(batch):
    era5s, cerras, fnames = zip(*batch)
    era5_batch = torch.stack(era5s, dim=0)
    cerra_batch = torch.stack(cerras, dim=0)
    return era5_batch, cerra_batch, list(fnames)

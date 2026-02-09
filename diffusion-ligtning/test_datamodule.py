from .datamodule.cerra_era5_dm import CerraEra5rDataModule


# Initialize the DataModule
dm = CerraEra5rDataModule(
    zarr_path='../output_data.zarr',
    batch_size=16,
    include_members=True  # Set False if you only want Mean/Std
)

# Setup (loads file, performs split)
dm.setup()

# Get a batch
loader = dm.train_dataloader()
batch = next(iter(loader))

# Check shapes
print("Batch Keys:", batch.keys())
print("ERA5 Mean Shape:", batch['era5_mean'].shape)      # Expect: [16, 1, 128, 128]
if 'era5_members' in batch:
    print("ERA5 Members Shape:", batch['era5_members'].shape) # Expect: [16, 10, 128, 128]
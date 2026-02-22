import cdsapi
import calendar
from pathlib import Path

dataset = "reanalysis-era5-single-levels"
year = "2020"

output_dir = Path("/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/ERA5EDA_DATA")
output_dir.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

times = ["00:00", "06:00", "12:00", "18:00"]

for time in times:
    for month in range(1, 12):

        mm = f"{month:02d}"
        ndays = calendar.monthrange(int(year), month)[1]

        request = {
            "product_type": [ "ensemble_members" ],
            "variable": ["2m_temperature"],
            "year": [year],
            "month": [mm],
            "day": [f"{day:02d}" for day in range(1, ndays + 1)],
            "time": [time],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [75.48, -60.4, 17.61, 76.4] }

        outfile = output_dir / f"erra5eda_t2m_{year}{mm}_{time.replace(':','')}.nc"

        print(f"Downloading {outfile} ...", flush=True)

        client.retrieve(
            dataset,
            request,
            str(outfile)
        )

print("All months successfully downloaded.")

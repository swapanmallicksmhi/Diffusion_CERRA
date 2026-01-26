import cdsapi
import calendar
from pathlib import Path

dataset = "reanalysis-cerra-single-levels"
year = "2024"

output_dir = Path("/lus/h2resw01/scratch/swe4281/CERRA_DATA2026/CERRA_DATA_DYN")
output_dir.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()

times = ["00:00", "06:00", "12:00", "18:00"]
"""
import cdsapi

dataset = "reanalysis-cerra-single-levels"
request = {
    "variable": ["2m_temperature"],
    "level_type": "surface_or_atmosphere",
    "data_type": ["reanalysis"],
    "product_type": "analysis",
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
"""

for time in times:
    for month in range(1, 13):

        mm = f"{month:02d}"
        ndays = calendar.monthrange(int(year), month)[1]

        request = {
            "variable": ["2m_temperature"],
            "level_type": "surface_or_atmosphere",
            "data_type": ["reanalysis"],
            "product_type": "analysis",
            "year": [year],
            "month": [mm],
            "day": [f"{day:02d}" for day in range(1, ndays + 1)],
            "time": [time],
            "data_format": "netcdf"
        }

        outfile = output_dir / f"cerra_t2m_dyn_{year}{mm}_{time.replace(':','')}.nc"

        print(f"Downloading {outfile} ...", flush=True)

        client.retrieve(
            dataset,
            request,
            str(outfile)
        )

print("All months successfully downloaded.")

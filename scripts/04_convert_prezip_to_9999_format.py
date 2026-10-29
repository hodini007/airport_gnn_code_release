"""
Convert BTS PREZIP-format monthly CSVs (CamelCase columns, ISO FlightDate --
the flat bulk-download server format) into the exact column names, column
order, and date-string format already used by the files in 9999/ (which were
built through BTS's interactive field-selector tool and use
ALL_CAPS_WITH_UNDERSCORES names plus a single "M/D/YYYY 12:00:00 AM" FL_DATE
string).

This lets newly downloaded months (2019 backfill, and the 5 gap months found
missing in 2020-2023) be processed by the existing Stage 1-2 pipeline
(02_clean_and_merge_bts.py) exactly as if they had come from the same source
as everything already in 9999/ -- verified column-for-column and value-for-
value identical against the official source in scripts/03_verify_against_official_bts.py.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR = Path("./bts_supplemental_raw")
OUT_DIR = Path("./bts_supplemental_converted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 9999/-style column name -> PREZIP column name, in the exact order 9999/ uses.
COLUMN_ORDER = [
    ("FL_DATE", "FlightDate"),
    ("OP_UNIQUE_CARRIER", "Reporting_Airline"),
    ("OP_CARRIER_AIRLINE_ID", "DOT_ID_Reporting_Airline"),
    ("OP_CARRIER", "IATA_CODE_Reporting_Airline"),
    ("TAIL_NUM", "Tail_Number"),
    ("OP_CARRIER_FL_NUM", "Flight_Number_Reporting_Airline"),
    ("ORIGIN", "Origin"),
    ("ORIGIN_CITY_NAME", "OriginCityName"),
    ("DEST", "Dest"),
    ("DEST_CITY_NAME", "DestCityName"),
    ("CRS_DEP_TIME", "CRSDepTime"),
    ("DEP_TIME", "DepTime"),
    ("DEP_DELAY", "DepDelay"),
    ("DEP_DELAY_NEW", "DepDelayMinutes"),
    ("DEP_DEL15", "DepDel15"),
    ("DEP_DELAY_GROUP", "DepartureDelayGroups"),
    ("DEP_TIME_BLK", "DepTimeBlk"),
    ("TAXI_OUT", "TaxiOut"),
    ("WHEELS_OFF", "WheelsOff"),
    ("WHEELS_ON", "WheelsOn"),
    ("TAXI_IN", "TaxiIn"),
    ("CRS_ARR_TIME", "CRSArrTime"),
    ("ARR_TIME", "ArrTime"),
    ("ARR_DELAY", "ArrDelay"),
    ("ARR_DEL15", "ArrDel15"),
    ("CANCELLED", "Cancelled"),
    ("CANCELLATION_CODE", "CancellationCode"),
    ("DIVERTED", "Diverted"),
    ("CRS_ELAPSED_TIME", "CRSElapsedTime"),
    ("ACTUAL_ELAPSED_TIME", "ActualElapsedTime"),
    ("AIR_TIME", "AirTime"),
    ("DISTANCE", "Distance"),
    ("CARRIER_DELAY", "CarrierDelay"),
    ("WEATHER_DELAY", "WeatherDelay"),
    ("NAS_DELAY", "NASDelay"),
    ("SECURITY_DELAY", "SecurityDelay"),
    ("LATE_AIRCRAFT_DELAY", "LateAircraftDelay"),
]

TARGET_COLS = [t for t, _ in COLUMN_ORDER]
SOURCE_COLS = [s for _, s in COLUMN_ORDER]
RENAME_MAP = {s: t for t, s in COLUMN_ORDER}


def convert_one(zip_path: Path) -> Path:
    m = re.match(r"(\d{4})_(\d{1,2})\.zip$", zip_path.name)
    year, month = int(m.group(1)), int(m.group(2))
    out_path = OUT_DIR / f"{year:04d}_{month:02d}.csv"
    if out_path.exists():
        print(f"  {year}-{month:02d}: already converted, skipping")
        return out_path

    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        assert len(csv_names) == 1, f"expected exactly one CSV in {zip_path.name}, found {csv_names}"
        with z.open(csv_names[0]) as f:
            df = pd.read_csv(f, usecols=SOURCE_COLS, low_memory=False)

    df = df.rename(columns=RENAME_MAP)[TARGET_COLS]

    # PREZIP's FlightDate is ISO "YYYY-MM-DD"; 9999/ uses "M/D/YYYY 12:00:00 AM"
    # (no zero-padding on month/day, time always midnight since it's date-only).
    dt = pd.to_datetime(df["FL_DATE"], format="%Y-%m-%d")
    df["FL_DATE"] = dt.dt.strftime("%-m/%-d/%Y 12:00:00 AM")

    df.to_csv(out_path, index=False)
    print(f"  {year}-{month:02d}: converted {len(df):,} rows -> {out_path.name}")
    return out_path


def main():
    zips = sorted(RAW_DIR.glob("*.zip"))
    print(f"Converting {len(zips)} PREZIP files to 9999/-format CSVs...")
    for z in zips:
        convert_one(z)
    print(f"\nDone. Converted files are in {OUT_DIR}")


if __name__ == "__main__":
    main()

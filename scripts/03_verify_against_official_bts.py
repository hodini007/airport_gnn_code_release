"""
Cross-check the existing 9999/ dataset against an official BTS PREZIP download
for the same month (January 2020), to confirm 9999/ is genuine, unaltered BTS
data and to document the exact column mapping between the two export formats.
"""
import pandas as pd

OFFICIAL_PATH = "./bts_verify/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2020_1.csv"
EXISTING_PATH = "./9999/112020.csv"

# Column mapping: existing 9999/ name -> official PREZIP name
COLMAP = {
    "FL_DATE": "FlightDate",
    "OP_UNIQUE_CARRIER": "Reporting_Airline",
    "OP_CARRIER_AIRLINE_ID": "DOT_ID_Reporting_Airline",
    "OP_CARRIER": "IATA_CODE_Reporting_Airline",
    "TAIL_NUM": "Tail_Number",
    "OP_CARRIER_FL_NUM": "Flight_Number_Reporting_Airline",
    "ORIGIN": "Origin",
    "DEST": "Dest",
    "CRS_DEP_TIME": "CRSDepTime",
    "DEP_TIME": "DepTime",
    "DEP_DELAY": "DepDelay",
    "DEP_DEL15": "DepDel15",
    "TAXI_OUT": "TaxiOut",
    "TAXI_IN": "TaxiIn",
    "CRS_ARR_TIME": "CRSArrTime",
    "ARR_TIME": "ArrTime",
    "ARR_DELAY": "ArrDelay",
    "ARR_DEL15": "ArrDel15",
    "CANCELLED": "Cancelled",
    "DIVERTED": "Diverted",
    "CRS_ELAPSED_TIME": "CRSElapsedTime",
    "ACTUAL_ELAPSED_TIME": "ActualElapsedTime",
    "AIR_TIME": "AirTime",
    "DISTANCE": "Distance",
    "CARRIER_DELAY": "CarrierDelay",
    "WEATHER_DELAY": "WeatherDelay",
    "NAS_DELAY": "NASDelay",
    "SECURITY_DELAY": "SecurityDelay",
    "LATE_AIRCRAFT_DELAY": "LateAircraftDelay",
}

print("Loading existing 9999/112020.csv ...")
existing = pd.read_csv(EXISTING_PATH, low_memory=False)
print(f"  existing rows: {len(existing):,}, cols: {len(existing.columns)}")

print("Loading official PREZIP 2020_1 CSV ...")
official = pd.read_csv(OFFICIAL_PATH, low_memory=False)
print(f"  official rows: {len(official):,}, cols: {len(official.columns)}")

print("\n=== Row count comparison ===")
print(f"existing: {len(existing):,}   official: {len(official):,}   diff: {len(existing) - len(official):,}")

print("\n=== Column presence check (existing -> official) ===")
missing_in_official = [c for c in COLMAP.values() if c not in official.columns]
print("Expected official columns missing:", missing_in_official if missing_in_official else "none - all present")

# Build a comparable key on both sides: date + origin + dest + flight number + tail number
existing_key = (
    pd.to_datetime(existing["FL_DATE"]).dt.strftime("%Y-%m-%d")
    + "_" + existing["ORIGIN"].astype(str)
    + "_" + existing["DEST"].astype(str)
    + "_" + existing["OP_CARRIER_FL_NUM"].astype(str)
    + "_" + existing["TAIL_NUM"].astype(str)
)
official_key = (
    official["FlightDate"].astype(str)
    + "_" + official["Origin"].astype(str)
    + "_" + official["Dest"].astype(str)
    + "_" + official["Flight_Number_Reporting_Airline"].astype(str)
    + "_" + official["Tail_Number"].astype(str)
)

existing = existing.assign(_key=existing_key)
official = official.assign(_key=official_key)

print("\n=== Key uniqueness / match check ===")
print(f"existing unique keys: {existing['_key'].nunique():,} of {len(existing):,} rows")
print(f"official unique keys: {official['_key'].nunique():,} of {len(official):,} rows")

merged = existing.merge(official, on="_key", how="inner", suffixes=("_existing", "_official"))
print(f"matched rows on (date, origin, dest, flight#, tail#): {len(merged):,}")
print(f"existing rows with NO match in official: {len(existing) - len(merged):,}")
print(f"official rows with NO match in existing: {len(official) - len(merged):,}")

print("\n=== Value agreement on matched rows ===")
check_pairs = [
    ("DEP_DELAY", "DepDelay"),
    ("ARR_DELAY", "ArrDelay"),
    ("ARR_DEL15", "ArrDel15"),
    ("CARRIER_DELAY", "CarrierDelay"),
    ("WEATHER_DELAY", "WeatherDelay"),
    ("NAS_DELAY", "NASDelay"),
    ("SECURITY_DELAY", "SecurityDelay"),
    ("LATE_AIRCRAFT_DELAY", "LateAircraftDelay"),
    ("DISTANCE", "Distance"),
]
for ex_col, off_col in check_pairs:
    a = merged[f"{ex_col}_existing" if f"{ex_col}_existing" in merged.columns else ex_col]
    b = merged[f"{off_col}_official" if f"{off_col}_official" in merged.columns else off_col]
    a_num = pd.to_numeric(a, errors="coerce")
    b_num = pd.to_numeric(b, errors="coerce")
    both_null = a_num.isna() & b_num.isna()
    diff = (a_num - b_num).abs()
    mismatches = ((diff > 0.01) & ~both_null).sum()
    print(f"  {ex_col:22s} vs {off_col:20s}  mismatches: {mismatches:,} / {len(merged):,}  ({mismatches/len(merged)*100:.4f}%)")

print("\n=== Sample of 3 matched rows, side by side ===")
sample_cols_existing = ["FL_DATE", "ORIGIN", "DEST", "OP_CARRIER_FL_NUM", "TAIL_NUM", "DEP_DELAY", "ARR_DELAY", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY"]
sample_cols_official = ["FlightDate", "Origin", "Dest", "Flight_Number_Reporting_Airline", "Tail_Number", "DepDelay", "ArrDelay", "CarrierDelay", "WeatherDelay", "NASDelay"]
for i in range(min(3, len(merged))):
    row = merged.iloc[i]
    print(f"--- match {i+1} ---")
    for ec, oc in zip(sample_cols_existing, sample_cols_official):
        ev = row[f"{ec}_existing"] if f"{ec}_existing" in merged.columns else row[ec]
        ov = row[f"{oc}_official"] if f"{oc}_official" in merged.columns else row[oc]
        print(f"  {ec:22s} existing={ev!s:20s} official({oc})={ov!s}")

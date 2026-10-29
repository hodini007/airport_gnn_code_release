"""
Adds the "previous leg's dominant cause" feature specified in
Model_Architecture_Specification.docx Section 3.2 -- Head C needs this as
a direct input to detect cause-shift, rather than only inferring it from
delay size. This didn't exist anywhere in the pipeline yet.

Same two-pass, memory-safe approach as 05_build_rotation_feature.py (narrow
columns across all months, sort once by TAIL_NUM + chronological order,
shift, merge back), reusing the same lessons learned there: keep category
dtypes explicit after concat, avoid .dt.date, drop unneeded columns before
the sort.

Adds two columns to monthly_enriched/:
  - OWN_DOMINANT_CAUSE  : this flight's own dominant cause (one of the 5
                          BTS categories, or "NOT_DELAYED")
  - ROTATION_PREV_CAUSE : the same aircraft's immediately preceding leg's
                          OWN_DOMINANT_CAUSE (same 20-hour staleness rule
                          as the rotation delay feature; "NO_PREV_LEG" if
                          there isn't a usable previous leg)
"""
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

ENRICHED_DIR = Path("./data_processed/monthly_enriched")

CAUSE_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
NARROW_COLS = [
    "SOURCE_YEAR", "SOURCE_MONTH", "TAIL_NUM", "FL_DATE", "CRS_DEP_TIME", "DEP_TIME",
    "ARR_DEL15",
] + CAUSE_COLS


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    v = series.astype("float64")
    v = v.where(v != 2400, 0)
    return (v // 100) * 60 + (v % 100)


def mem_gb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e9


def compute_own_dominant_cause(df: pd.DataFrame) -> pd.Series:
    causes = df[CAUSE_COLS].copy()
    all_missing = causes.isna().all(axis=1)
    causes = causes.fillna(0.0)
    dominant = causes.idxmax(axis=1)
    dominant = dominant.where(~(all_missing | (df["ARR_DEL15"] != 1)), "NOT_DELAYED")
    return dominant.astype("category")


def load_and_compute() -> pd.DataFrame:
    files = sorted(ENRICHED_DIR.glob("*.parquet"))
    print(f"Loading narrow columns from {len(files)} months...", flush=True)
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=NARROW_COLS)
        df = df.reset_index(drop=True)
        df["_ROW_POS"] = np.arange(len(df), dtype="int32")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    combined["TAIL_NUM"] = combined["TAIL_NUM"].astype("category")
    print(f"Combined: {len(combined):,} rows, {mem_gb(combined):.2f} GB", flush=True)

    print("Computing each flight's own dominant cause...", flush=True)
    combined["OWN_DOMINANT_CAUSE"] = compute_own_dominant_cause(combined)
    combined = combined.drop(columns=CAUSE_COLS + ["ARR_DEL15"])
    gc.collect()

    print("Building departure datetimes for ordering...", flush=True)
    fl_date = pd.to_datetime(combined["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    dep_source = combined["DEP_TIME"].where(combined["DEP_TIME"].notna(), combined["CRS_DEP_TIME"])
    combined["_DEP_DT"] = fl_date + pd.to_timedelta(hhmm_to_minutes(dep_source), unit="m")
    combined = combined.drop(columns=["FL_DATE", "DEP_TIME", "CRS_DEP_TIME"])
    gc.collect()
    print(f"After trimming: {mem_gb(combined):.2f} GB", flush=True)

    print("Sorting by TAIL_NUM, then chronologically...", flush=True)
    combined = combined.sort_values(["TAIL_NUM", "_DEP_DT"], kind="mergesort").reset_index(drop=True)
    gc.collect()
    print(f"After sort: {mem_gb(combined):.2f} GB", flush=True)

    print("Computing previous-leg cause via groupby shift...", flush=True)
    g = combined.groupby("TAIL_NUM", observed=True, sort=False)
    combined["_PREV_DEP_DT"] = g["_DEP_DT"].shift(1)
    combined["ROTATION_PREV_CAUSE"] = g["OWN_DOMINANT_CAUSE"].shift(1)

    gap_hours = (combined["_DEP_DT"] - combined["_PREV_DEP_DT"]).dt.total_seconds() / 3600.0
    no_usable_prev = combined["_PREV_DEP_DT"].isna() | (gap_hours > 20)
    combined["ROTATION_PREV_CAUSE"] = combined["ROTATION_PREV_CAUSE"].astype("object")
    combined.loc[no_usable_prev, "ROTATION_PREV_CAUSE"] = "NO_PREV_LEG"
    combined["ROTATION_PREV_CAUSE"] = combined["ROTATION_PREV_CAUSE"].astype("category")
    gc.collect()

    return combined[["SOURCE_YEAR", "SOURCE_MONTH", "_ROW_POS", "OWN_DOMINANT_CAUSE", "ROTATION_PREV_CAUSE"]]


def merge_back_and_write(features: pd.DataFrame):
    files = sorted(ENRICHED_DIR.glob("*.parquet"))
    print(f"\nMerging back into {len(files)} monthly files...", flush=True)
    features = features.set_index(["SOURCE_YEAR", "SOURCE_MONTH", "_ROW_POS"])

    for f in files:
        year, month = int(f.stem[:4]), int(f.stem[5:7])
        df = pd.read_parquet(f)
        if "OWN_DOMINANT_CAUSE" in df.columns:
            print(f"  {f.stem}: already has this feature, skipping", flush=True)
            continue
        df = df.reset_index(drop=True)
        df["_ROW_POS"] = np.arange(len(df), dtype="int32")
        df["SOURCE_YEAR"] = year
        df["SOURCE_MONTH"] = month
        df = df.set_index(["SOURCE_YEAR", "SOURCE_MONTH", "_ROW_POS"])
        df = df.join(features, how="left")
        df = df.reset_index()

        # Write to a temp file and atomically replace -- if the process
        # crashes mid-write (this project has hit real WSL crashes before),
        # the original enriched file is never left partially overwritten.
        tmp_path = f.with_suffix(".tmp.parquet")
        df.to_parquet(tmp_path, index=False, compression="snappy")
        os.replace(tmp_path, f)

        shift_rate = (df["ROTATION_PREV_CAUSE"].astype(str) != df["OWN_DOMINANT_CAUSE"].astype(str)).mean() * 100
        print(f"  {f.stem}: {len(df):,} rows, cause differs from previous leg in {shift_rate:.1f}% of rows", flush=True)
        del df
        gc.collect()


def main():
    features = load_and_compute()
    merge_back_and_write(features)
    print("\nDone.")


if __name__ == "__main__":
    main()

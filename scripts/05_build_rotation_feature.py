"""
Stage 3a: Aircraft rotation carry-over feature.

For every flight, finds that same physical aircraft's (TAIL_NUM) immediately
preceding leg -- possibly in an earlier month's file, since rotations cross
month boundaries -- and attaches:

  - ROTATION_PREV_ARR_DELAY : the previous leg's actual arrival delay (the
    core propagation signal at the heart of this project)
  - ROTATION_TURNAROUND_MIN : minutes between the previous leg's actual
    arrival and this leg's actual departure (needed later for the
    counterfactual explainer, Stage 8)
  - ROTATION_SAME_DAY       : whether the previous leg was the same
    calendar day (rotations spanning >1 day are weaker signal)
  - ROTATION_ROUTE_CONTINUOUS : whether the previous leg's destination
    equals this leg's origin (a data-quality sanity flag -- should be True
    for the vast majority of genuine same-aircraft rotations)

Runs in two passes to stay within WSL's 9.7GB RAM cap on ~42M rows:
  1. Load only the ~10 columns needed for ordering (not all 37), across all
     78 months at once -- small enough to hold in memory, sort by
     (TAIL_NUM, FL_DATE, CRS_DEP_TIME), then compute the previous-leg
     features via a single shift() per aircraft.
  2. Merge those new columns back into each month's full file individually
     (one month in memory at a time) and write to monthly_enriched/.

Row identity across the two passes is (SOURCE_YEAR, SOURCE_MONTH,
row position within that month's parquet file), which is stable because
parquet preserves row order deterministically within a file.
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

MONTHLY_DIR = Path("./data_processed/monthly")
OUT_DIR = Path("./data_processed/monthly_enriched")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NARROW_COLS = [
    "SOURCE_YEAR", "SOURCE_MONTH", "TAIL_NUM", "ORIGIN", "DEST",
    "FL_DATE", "CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", "ARR_TIME",
    "ARR_DELAY", "CANCELLED",
]


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    v = series.astype("float64")
    v = v.where(v != 2400, 0)
    return (v // 100) * 60 + (v % 100)


def mem_gb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e9


def load_narrow_all_months() -> pd.DataFrame:
    files = sorted(MONTHLY_DIR.glob("*.parquet"))
    print(f"Pass 1: loading narrow columns from {len(files)} months for global rotation ordering...", flush=True)
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=NARROW_COLS)
        df = df.reset_index(drop=True)
        df["_ROW_POS"] = np.arange(len(df), dtype="int32")
        frames.append(df)
        print(f"  loaded {f.stem}: {len(df):,} rows", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # pd.concat across files whose per-file categorical value sets differ can
    # silently upcast category -> object (full Python strings instead of
    # compact integer codes), which is dramatically more memory-hungry at
    # 42M rows. Force back to category explicitly rather than trusting concat.
    # ORIGIN and DEST must share one unified category set (not just "be a
    # category") since ROTATION_ROUTE_CONTINUOUS compares them directly --
    # pandas refuses to compare two Categoricals with different categories.
    airport_categories = pd.api.types.union_categoricals(
        [combined["ORIGIN"].astype("category"), combined["DEST"].astype("category")]
    ).categories
    airport_dtype = pd.CategoricalDtype(categories=airport_categories)
    combined["ORIGIN"] = combined["ORIGIN"].astype(airport_dtype)
    combined["DEST"] = combined["DEST"].astype(airport_dtype)
    combined["TAIL_NUM"] = combined["TAIL_NUM"].astype("category")

    print(f"Combined narrow table: {len(combined):,} rows, {mem_gb(combined):.2f} GB", flush=True)
    return combined


def compute_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Building actual departure/arrival datetimes...", flush=True)
    fl_date = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")

    # Actual times when present, falling back to scheduled (needed for
    # cancelled flights, which have no actual DEP_TIME/ARR_TIME).
    dep_time_source = df["DEP_TIME"].where(df["DEP_TIME"].notna(), df["CRS_DEP_TIME"])
    arr_time_source = df["ARR_TIME"].where(df["ARR_TIME"].notna(), df["CRS_ARR_TIME"])

    df["_DEP_DT"] = fl_date + pd.to_timedelta(hhmm_to_minutes(dep_time_source), unit="m")
    # Arrival can roll past midnight; if arrival minutes < departure minutes
    # for the same scheduled leg, assume it landed the next day.
    dep_min = hhmm_to_minutes(dep_time_source)
    arr_min = hhmm_to_minutes(arr_time_source)
    arr_day_offset = (arr_min < dep_min).astype("int64")
    df["_ARR_DT"] = fl_date + pd.to_timedelta(arr_min + arr_day_offset * 1440, unit="m")
    del fl_date, dep_time_source, arr_time_source, dep_min, arr_min, arr_day_offset

    # Drop everything that was only needed to compute the two datetimes
    # above -- in particular FL_DATE (a ~20-char string per row, one of the
    # biggest hidden memory costs at 42M rows) -- before the expensive sort.
    df = df.drop(columns=["FL_DATE", "DEP_TIME", "ARR_TIME", "CRS_DEP_TIME", "CRS_ARR_TIME", "CANCELLED"])
    gc.collect()
    print(f"After trimming to essential columns: {mem_gb(df):.2f} GB", flush=True)

    print("Sorting by TAIL_NUM, then chronologically...", flush=True)
    df = df.sort_values(["TAIL_NUM", "_DEP_DT"], kind="mergesort").reset_index(drop=True)
    gc.collect()
    print(f"After sort: {mem_gb(df):.2f} GB", flush=True)

    print("Computing previous-leg features via groupby shift...", flush=True)
    # .dt.normalize() keeps a vectorized datetime64 column (comparable and
    # cheap at 42M rows); .dt.date would produce an object array of 42M
    # individual Python date objects, which is dramatically more memory.
    df["_DEP_DATE"] = df["_DEP_DT"].dt.normalize()
    g = df.groupby("TAIL_NUM", observed=True, sort=False)
    df["ROTATION_PREV_ARR_DELAY"] = g["ARR_DELAY"].shift(1)
    df["_PREV_DEST"] = g["DEST"].shift(1)
    df["_PREV_ARR_DT"] = g["_ARR_DT"].shift(1)
    df["_PREV_DEP_DATE"] = g["_DEP_DATE"].shift(1)

    df["ROTATION_TURNAROUND_MIN"] = (df["_DEP_DT"] - df["_PREV_ARR_DT"]).dt.total_seconds() / 60.0
    df["ROTATION_SAME_DAY"] = (df["_PREV_DEP_DATE"] == df["_DEP_DATE"])
    df["ROTATION_ROUTE_CONTINUOUS"] = (df["_PREV_DEST"] == df["ORIGIN"])
    df["ROTATION_HAS_PREV_LEG"] = df["_PREV_ARR_DT"].notna()
    gc.collect()
    print(f"After groupby/shift: {mem_gb(df):.2f} GB", flush=True)

    # A rotation "previous leg" more than 20 hours earlier is not a same-day
    # turnaround signal -- almost certainly the aircraft's first flight of a
    # new operational day (or a data gap). Null it out rather than carrying
    # stale delay forward.
    gap_hours = (df["_DEP_DT"] - df["_PREV_ARR_DT"]).dt.total_seconds() / 3600.0
    stale = gap_hours > 20
    df.loc[stale, ["ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN"]] = np.nan
    df.loc[stale, "ROTATION_HAS_PREV_LEG"] = False

    keep_cols = [
        "SOURCE_YEAR", "SOURCE_MONTH", "_ROW_POS",
        "ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN",
        "ROTATION_SAME_DAY", "ROTATION_ROUTE_CONTINUOUS", "ROTATION_HAS_PREV_LEG",
    ]
    return df[keep_cols]


def merge_back_and_write(rotation_features: pd.DataFrame):
    files = sorted(MONTHLY_DIR.glob("*.parquet"))
    print(f"\nPass 2: merging rotation features back into {len(files)} monthly files...", flush=True)
    rotation_features = rotation_features.set_index(["SOURCE_YEAR", "SOURCE_MONTH", "_ROW_POS"])

    for f in files:
        year_month = f.stem
        year, month = int(year_month[:4]), int(year_month[5:7])
        out_path = OUT_DIR / f.name
        if out_path.exists():
            print(f"  {year_month}: already enriched, skipping", flush=True)
            continue

        df = pd.read_parquet(f).reset_index(drop=True)
        df["_ROW_POS"] = np.arange(len(df), dtype="int32")
        df = df.set_index(["SOURCE_YEAR", "SOURCE_MONTH", "_ROW_POS"])
        df = df.join(rotation_features, how="left")
        df = df.reset_index()

        df.to_parquet(out_path, index=False, compression="snappy")
        pct_with_prev = df["ROTATION_HAS_PREV_LEG"].mean() * 100
        print(f"  {year_month}: {len(df):,} rows, {pct_with_prev:.1f}% have a usable previous leg", flush=True)
        del df
        gc.collect()


def main():
    narrow = load_narrow_all_months()
    rotation_features = compute_rotation_features(narrow)
    del narrow
    gc.collect()
    merge_back_and_write(rotation_features)
    print(f"\nDone. Enriched monthly files written to {OUT_DIR}")


if __name__ == "__main__":
    main()

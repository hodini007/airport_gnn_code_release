"""
Stage 3c: Node/edge time-window aggregation.

Groups individual flight rows into two summary tables per the methodology's
Stage 3 (4-hour bins, chosen from the doc's "hourly or 4-hour bins" option
to keep the table size manageable while still capturing daily traffic
patterns -- morning/midday/evening/overnight):

  - Node table: one row per (AIRPORT, TIME_WINDOW), combining that airport's
    departure-side and arrival-side activity in that window -- flight
    counts, mean/variance of delay, and taxi time as a congestion proxy.
  - Edge table: one row per (ORIGIN, DEST, TIME_WINDOW), the route-level
    counterpart -- flight count, mean delay, delay rate.

These are the inputs Stage 4 turns into the actual graph; this script only
builds the aggregated tables, one month at a time (no cross-month grouping
needed here, unlike the rotation feature, since a time window never spans
two months).

Reads from monthly_enriched/ (must be run after 05_build_rotation_feature.py,
so REGIME and the rotation columns are available to carry through).
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

ENRICHED_DIR = Path("./data_processed/monthly_enriched")
NODE_DIR = Path("./data_processed/node_features")
EDGE_DIR = Path("./data_processed/edge_features")
NODE_DIR.mkdir(parents=True, exist_ok=True)
EDGE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_MINUTES = 240  # 4-hour bins

NEEDED_COLS = [
    "ORIGIN", "DEST", "FL_DATE", "CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", "ARR_TIME",
    "DEP_DELAY", "ARR_DELAY", "ARR_DEL15", "TAXI_OUT", "TAXI_IN", "CANCELLED", "REGIME",
]


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    v = series.astype("float64")
    v = v.where(v != 2400, 0)
    return (v // 100) * 60 + (v % 100)


def add_window_columns(df: pd.DataFrame, fl_date: pd.Series) -> pd.DataFrame:
    dep_source = df["DEP_TIME"].where(df["DEP_TIME"].notna(), df["CRS_DEP_TIME"])
    arr_source = df["ARR_TIME"].where(df["ARR_TIME"].notna(), df["CRS_ARR_TIME"])

    dep_min = hhmm_to_minutes(dep_source)
    arr_min = hhmm_to_minutes(arr_source)
    arr_day_offset = (arr_min < dep_min).astype("int64")

    dep_window_min = (dep_min // WINDOW_MINUTES) * WINDOW_MINUTES
    arr_window_min = (arr_min // WINDOW_MINUTES) * WINDOW_MINUTES

    df["_DEP_WINDOW"] = fl_date + pd.to_timedelta(dep_window_min, unit="m")
    df["_ARR_WINDOW"] = fl_date + pd.to_timedelta(arr_window_min + arr_day_offset * 1440, unit="m")
    return df


def build_node_table(df: pd.DataFrame) -> pd.DataFrame:
    dep_side = df.groupby(["ORIGIN", "_DEP_WINDOW"], observed=True).agg(
        N_DEPARTURES=("DEP_DELAY", "size"),
        MEAN_DEP_DELAY=("DEP_DELAY", "mean"),
        STD_DEP_DELAY=("DEP_DELAY", "std"),
        MEAN_TAXI_OUT=("TAXI_OUT", "mean"),
        N_CANCELLED_DEP=("CANCELLED", "sum"),
    ).reset_index().rename(columns={"ORIGIN": "AIRPORT", "_DEP_WINDOW": "WINDOW_START"})

    arr_side = df.groupby(["DEST", "_ARR_WINDOW"], observed=True).agg(
        N_ARRIVALS=("ARR_DELAY", "size"),
        MEAN_ARR_DELAY=("ARR_DELAY", "mean"),
        STD_ARR_DELAY=("ARR_DELAY", "std"),
        ARR_DEL15_RATE=("ARR_DEL15", "mean"),
        MEAN_TAXI_IN=("TAXI_IN", "mean"),
    ).reset_index().rename(columns={"DEST": "AIRPORT", "_ARR_WINDOW": "WINDOW_START"})

    # REGIME is constant within a (date-derived) window; take it from whichever side has it.
    regime_lookup = pd.concat([
        df[["ORIGIN", "_DEP_WINDOW", "REGIME"]].rename(columns={"ORIGIN": "AIRPORT", "_DEP_WINDOW": "WINDOW_START"}),
        df[["DEST", "_ARR_WINDOW", "REGIME"]].rename(columns={"DEST": "AIRPORT", "_ARR_WINDOW": "WINDOW_START"}),
    ]).drop_duplicates(["AIRPORT", "WINDOW_START"])

    node = dep_side.merge(arr_side, on=["AIRPORT", "WINDOW_START"], how="outer")
    node = node.merge(regime_lookup, on=["AIRPORT", "WINDOW_START"], how="left")

    for col in ["N_DEPARTURES", "N_ARRIVALS", "N_CANCELLED_DEP"]:
        node[col] = node[col].fillna(0).astype("int32")

    return node


def build_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    edge = df.groupby(["ORIGIN", "DEST", "_DEP_WINDOW"], observed=True).agg(
        N_FLIGHTS=("DEP_DELAY", "size"),
        MEAN_DEP_DELAY=("DEP_DELAY", "mean"),
        MEAN_ARR_DELAY=("ARR_DELAY", "mean"),
        ARR_DEL15_RATE=("ARR_DEL15", "mean"),
        REGIME=("REGIME", "first"),
    ).reset_index().rename(columns={"_DEP_WINDOW": "WINDOW_START"})
    edge["N_FLIGHTS"] = edge["N_FLIGHTS"].astype("int32")
    return edge


def process_month(path: Path):
    month_key = path.stem
    node_out = NODE_DIR / path.name
    edge_out = EDGE_DIR / path.name
    if node_out.exists() and edge_out.exists():
        print(f"  {month_key}: already aggregated, skipping", flush=True)
        return

    df = pd.read_parquet(path, columns=NEEDED_COLS)
    fl_date = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    df = add_window_columns(df, fl_date)

    node = build_node_table(df)
    edge = build_edge_table(df)

    node.to_parquet(node_out, index=False, compression="snappy")
    edge.to_parquet(edge_out, index=False, compression="snappy")

    print(f"  {month_key}: {len(df):,} flights -> {len(node):,} node-window rows, {len(edge):,} edge-window rows", flush=True)
    del df, node, edge
    gc.collect()


def main():
    files = sorted(ENRICHED_DIR.glob("*.parquet"))
    print(f"Aggregating {len(files)} months into {WINDOW_MINUTES}-minute airport/route windows...", flush=True)
    for f in files:
        process_month(f)
    print(f"\nDone. Node tables in {NODE_DIR}, edge tables in {EDGE_DIR}")


if __name__ == "__main__":
    main()

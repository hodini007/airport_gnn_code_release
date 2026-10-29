"""
Stage 4b: Build the time-indexed graph sequence G_t = (V, E_t, X_t).

V (the 200 selected airports) and each route's static historical delay
correlation are fixed inputs from 08_select_airports_and_edge_weights.py.
This script builds, for every 4-hour window in every month:

  - X_t : a [200, 10] node feature matrix, one row per selected airport in
    a FIXED order (same airport = same row index in every window, which is
    what makes this a proper time-indexed sequence rather than a one-off
    snapshot). Features: N_DEPARTURES, N_ARRIVALS, MEAN_DEP_DELAY,
    STD_DEP_DELAY, MEAN_ARR_DELAY, STD_ARR_DELAY, ARR_DEL15_RATE,
    MEAN_TAXI_OUT, MEAN_TAXI_IN, IS_ACTIVE (1 if the airport had any
    flights in this window, 0 otherwise -- an airport with no traffic in
    a window gets an all-zero feature row rather than being dropped, so
    the node set V stays identical across every window).
  - E_t : the routes that actually had a flight in window t, restricted to
    routes between two selected airports. Weighted by two edge attributes:
    this window's flight count (varies per window) and the static
    historical delay correlation (fixed, from script 08).

Output: one .pt file per month in data_processed/graph/sequence/, each
containing a list of (window_start, torch_geometric.data.Data) tuples.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

NODE_DIR = Path("./data_processed/node_features")
EDGE_DIR = Path("./data_processed/edge_features")
GRAPH_DIR = Path("./data_processed/graph")
OUT_DIR = GRAPH_DIR / "sequence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_FEATURE_COLS = [
    "N_DEPARTURES", "N_ARRIVALS", "MEAN_DEP_DELAY", "STD_DEP_DELAY",
    "MEAN_ARR_DELAY", "STD_ARR_DELAY", "ARR_DEL15_RATE", "MEAN_TAXI_OUT", "MEAN_TAXI_IN",
]


def load_fixed_inputs():
    with open(GRAPH_DIR / "selected_airports.json") as f:
        selected = json.load(f)
    airport_list = [a["code"] for a in selected["airports"]]
    airport_index = {code: i for i, code in enumerate(airport_list)}

    corr = pd.read_parquet(GRAPH_DIR / "edge_historical_correlation.parquet")
    corr_lookup = {(r.ORIGIN, r.DEST): r.HISTORICAL_DELAY_CORR for r in corr.itertuples(index=False)}

    return airport_list, airport_index, corr_lookup


def build_node_matrix(window_rows: pd.DataFrame, airport_index: dict) -> torch.Tensor:
    n = len(airport_index)
    x = np.zeros((n, len(NODE_FEATURE_COLS) + 1), dtype="float32")  # +1 for IS_ACTIVE

    for row in window_rows.itertuples(index=False):
        idx = airport_index.get(row.AIRPORT)
        if idx is None:
            continue  # airport not in the selected top-200
        for j, col in enumerate(NODE_FEATURE_COLS):
            val = getattr(row, col)
            x[idx, j] = 0.0 if pd.isna(val) else float(val)
        x[idx, -1] = 1.0  # IS_ACTIVE

    return torch.from_numpy(x)


def build_edges(window_rows: pd.DataFrame, airport_index: dict, corr_lookup: dict):
    src, dst, attrs = [], [], []
    for row in window_rows.itertuples(index=False):
        o_idx = airport_index.get(row.ORIGIN)
        d_idx = airport_index.get(row.DEST)
        if o_idx is None or d_idx is None:
            continue  # route touches an airport outside the selected top-200
        src.append(o_idx)
        dst.append(d_idx)
        historical_corr = corr_lookup.get((row.ORIGIN, row.DEST), 0.0)
        attrs.append([float(row.N_FLIGHTS), float(historical_corr)])

    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def process_month(node_path: Path, edge_path: Path, airport_list: list, airport_index: dict, corr_lookup: dict):
    month_key = node_path.stem
    out_path = OUT_DIR / f"{month_key}.pt"
    if out_path.exists():
        print(f"  {month_key}: already built, skipping", flush=True)
        return

    node_month = pd.read_parquet(node_path)
    edge_month = pd.read_parquet(edge_path)

    node_groups = dict(tuple(node_month.groupby("WINDOW_START", observed=True)))
    edge_groups = dict(tuple(edge_month.groupby("WINDOW_START", observed=True)))
    windows = sorted(node_groups.keys())

    empty_edges = edge_month.iloc[0:0]
    graphs = []
    for window in windows:
        node_rows = node_groups[window]
        edge_rows = edge_groups.get(window, empty_edges)
        regime = node_rows["REGIME"].iloc[0] if len(node_rows) else None

        x = build_node_matrix(node_rows, airport_index)
        edge_index, edge_attr = build_edges(edge_rows, airport_index, corr_lookup)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.window_start = pd.Timestamp(window)
        data.regime = regime
        graphs.append(data)

    torch.save(graphs, out_path)
    n_edges = [g.edge_index.shape[1] for g in graphs]
    print(f"  {month_key}: {len(graphs)} windows, avg {np.mean(n_edges):.0f} edges/window", flush=True)


def main():
    airport_list, airport_index, corr_lookup = load_fixed_inputs()
    print(f"Fixed airport set: {len(airport_list)} airports", flush=True)

    with open(OUT_DIR / "airport_index.json", "w") as f:
        json.dump({"airport_order": airport_list}, f, indent=2)

    node_files = sorted(NODE_DIR.glob("*.parquet"))
    print(f"Building graph sequence for {len(node_files)} months...", flush=True)
    for node_path in node_files:
        edge_path = EDGE_DIR / node_path.name
        process_month(node_path, edge_path, airport_list, airport_index, corr_lookup)

    print(f"\nDone. Graph sequence written to {OUT_DIR}")


if __name__ == "__main__":
    main()

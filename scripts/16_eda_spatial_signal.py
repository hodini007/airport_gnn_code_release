"""
EDA requested directly in response to the GAT-vs-LSTM-only ablation finding
(README.md "Ablation baselines" section): before trying to fix GAT's
architecture, check whether the raw data actually contains exploitable
cross-airport signal that GAT specifically is failing to extract, or
whether there's little such signal to find in the first place.

scripts/08_select_airports_and_edge_weights.py builds each route's edge
weight from a CONTEMPORANEOUS correlation (corr(A[t], B[t]), same window,
no lag). That measures "these two airports tend to be delayed at the same
time" (plausibly just shared weather/holiday effects), not "A's congestion
predicts B's congestion next" -- the actual propagation relationship a
GNN would need to exploit. This script checks the LAGGED relationship
directly: for each of the busiest routes between two selected airports,
does the origin airport's delay state one window ago add any predictive
value for the destination's current delay state, beyond what the
destination's own recent history already provides?
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NODE_DIR = Path("./data_processed/node_features")
EDGE_DIR = Path("./data_processed/edge_features")
GRAPH_DIR = Path("./data_processed/graph")

N_ROUTES = 60  # busiest routes by total flight count, between selected airports


def main():
    with open(GRAPH_DIR / "selected_airports.json") as f:
        selected = json.load(f)
    selected_set = {a["code"] for a in selected["airports"]}

    print("Loading node delay time series for all selected airports...", flush=True)
    node_frames = []
    for f in sorted(NODE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["AIRPORT", "WINDOW_START", "MEAN_ARR_DELAY"])
        df = df[df["AIRPORT"].isin(selected_set)]
        node_frames.append(df)
    node_all = pd.concat(node_frames, ignore_index=True)
    delay_matrix = node_all.pivot_table(index="WINDOW_START", columns="AIRPORT", values="MEAN_ARR_DELAY", observed=True)
    delay_matrix = delay_matrix.sort_index()
    print(f"Delay time-series matrix: {delay_matrix.shape[0]:,} windows x {delay_matrix.shape[1]} airports", flush=True)

    print("Finding busiest routes between selected airports...", flush=True)
    edge_frames = []
    for f in sorted(EDGE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["ORIGIN", "DEST", "N_FLIGHTS"])
        df = df[df["ORIGIN"].isin(selected_set) & df["DEST"].isin(selected_set)]
        edge_frames.append(df)
    edge_all = pd.concat(edge_frames, ignore_index=True)
    route_volume = edge_all.groupby(["ORIGIN", "DEST"], observed=True)["N_FLIGHTS"].sum().sort_values(ascending=False)
    busiest_routes = route_volume.head(N_ROUTES).index.tolist()
    print(f"Analyzing the {len(busiest_routes)} busiest routes (by total flight count)", flush=True)

    results = []
    for origin, dest in busiest_routes:
        if origin not in delay_matrix.columns or dest not in delay_matrix.columns:
            continue
        o = delay_matrix[origin]
        d = delay_matrix[dest]
        df = pd.DataFrame({
            "d_t": d,
            "d_t_minus_1": d.shift(1),
            "o_t": o,
            "o_t_minus_1": o.shift(1),
        }).dropna()
        if len(df) < 100:
            continue

        contemp_corr = df["d_t"].corr(df["o_t"])          # what script 08's edge weight actually measures
        lagged_corr = df["d_t"].corr(df["o_t_minus_1"])    # the real propagation direction: origin's PAST -> dest's PRESENT
        dest_autocorr = df["d_t"].corr(df["d_t_minus_1"])  # dest's own persistence baseline

        # Incremental R^2: does adding the neighbor's lagged state help predict
        # dest's current state, beyond dest's own lag alone?
        from numpy.linalg import lstsq
        X_base = np.column_stack([np.ones(len(df)), df["d_t_minus_1"]])
        y = df["d_t"].values
        coef_base, *_ = lstsq(X_base, y, rcond=None)
        resid_base = y - X_base @ coef_base
        ss_res_base = (resid_base ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2_base = 1 - ss_res_base / ss_tot if ss_tot > 0 else 0.0

        X_full = np.column_stack([np.ones(len(df)), df["d_t_minus_1"], df["o_t_minus_1"]])
        coef_full, *_ = lstsq(X_full, y, rcond=None)
        resid_full = y - X_full @ coef_full
        ss_res_full = (resid_full ** 2).sum()
        r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0.0

        results.append({
            "origin": origin, "dest": dest, "n_windows": len(df),
            "contemp_corr": contemp_corr, "lagged_corr": lagged_corr, "dest_autocorr": dest_autocorr,
            "r2_dest_lag_only": r2_base, "r2_plus_neighbor_lag": r2_full, "r2_gain": r2_full - r2_base,
        })

    res_df = pd.DataFrame(results)
    print(f"\nAnalyzed {len(res_df)} routes with sufficient data.\n")
    print("=" * 80)
    print(f"Mean contemporaneous corr (what the current edge weight measures): {res_df['contemp_corr'].mean():.4f}")
    print(f"Mean lagged corr (origin[t-1] -> dest[t], the real propagation direction): {res_df['lagged_corr'].mean():.4f}")
    print(f"Mean destination autocorrelation (dest's own persistence, dest[t-1] -> dest[t]): {res_df['dest_autocorr'].mean():.4f}")
    print()
    print(f"Mean R^2 using dest's own lag alone: {res_df['r2_dest_lag_only'].mean():.4f}")
    print(f"Mean R^2 adding neighbor's lagged state: {res_df['r2_plus_neighbor_lag'].mean():.4f}")
    print(f"Mean R^2 GAIN from adding neighbor's lagged state: {res_df['r2_gain'].mean():.4f}")
    print(f"Routes where neighbor's lag gave a non-trivial gain (R^2 gain > 0.01): {(res_df['r2_gain'] > 0.01).sum()} / {len(res_df)}")
    print("=" * 80)

    out_path = GRAPH_DIR / "eda_spatial_signal_routes.csv"
    res_df.sort_values("r2_gain", ascending=False).to_csv(out_path, index=False)
    print(f"\nPer-route detail written to {out_path}")
    print("\nTop 10 routes by R^2 gain from neighbor's lagged state:")
    print(res_df.sort_values("r2_gain", ascending=False).head(10).to_string(index=False))
    print("\nBottom 10 routes by R^2 gain (most negative/least useful):")
    print(res_df.sort_values("r2_gain", ascending=True).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

"""
Stage 4a: Fixed airport set (V) + static historical delay-correlation edge weight.

Per the methodology: "V is the fixed set of airports (top 150-300 by
traffic volume, to keep computation tractable...), E_t is the set of
active routes in window t (weighted by scheduled flight count and
historical delay correlation between the endpoint pair)."

This script produces the two pieces that stay FIXED across all time
windows (unlike X_t and the per-window flight counts in E_t, which vary
by window and are handled in 09_build_graph_sequence.py):

  1. selected_airports.json -- the top 200 airports by total traffic
     volume across the full 78-month dataset. 200 is the middle of the
     methodology's stated 150-300 range: it covers the large majority of
     total U.S. domestic traffic (a small number of hub airports handle
     most volume) while keeping the graph small enough to train on.
  2. edge_historical_correlation.parquet -- for every (ORIGIN, DEST) pair
     that appears as a route at least once between two selected airports,
     the Pearson correlation between ORIGIN's MEAN_ARR_DELAY one window
     ago and DEST's MEAN_ARR_DELAY now, across all 4-hour windows in the
     dataset. This is a single static number per route, reused as one
     component of that route's edge weight in every time window (the
     other component, flight count, varies by window and is added in the
     next script).

     LAGGED, not contemporaneous, by design: scripts/16_eda_spatial_signal.py
     found the original same-window correlation (corr(origin[t], dest[t]))
     averages 0.56 on the busiest routes but is dominated by shared
     exogenous effects (weather, holidays) hitting many airports
     simultaneously, not real propagation -- the actual origin[t-1] ->
     dest[t] relationship averages only 0.087, weaker than a destination's
     own autocorrelation (0.164). Using the lagged version here means the
     edge weight -- and therefore what GAT is steered to attend to --
     reflects the propagation direction the graph is meant to model,
     instead of a mostly-confounded same-time co-movement.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

NODE_DIR = Path("./data_processed/node_features")
EDGE_DIR = Path("./data_processed/edge_features")
OUT_DIR = Path("./data_processed/graph")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_AIRPORTS = 200


def select_airports() -> list:
    print("Loading node tables to rank airports by total traffic volume...", flush=True)
    volume = {}
    for f in sorted(NODE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["AIRPORT", "N_DEPARTURES", "N_ARRIVALS"])
        month_volume = (df["N_DEPARTURES"] + df["N_ARRIVALS"]).groupby(df["AIRPORT"], observed=True).sum()
        for airport, v in month_volume.items():
            volume[airport] = volume.get(airport, 0) + int(v)

    ranked = sorted(volume.items(), key=lambda kv: -kv[1])
    total_flights_all_airports = sum(v for _, v in ranked)
    selected = ranked[:N_AIRPORTS]
    coverage = sum(v for _, v in selected) / total_flights_all_airports * 100

    print(f"Total airports seen: {len(ranked)}", flush=True)
    print(f"Selected top {N_AIRPORTS} by volume, covering {coverage:.1f}% of total traffic volume", flush=True)
    print(f"Largest: {selected[0]}  Smallest selected: {selected[-1]}", flush=True)

    out = {
        "n_airports": N_AIRPORTS,
        "traffic_coverage_pct": round(coverage, 2),
        "total_airports_in_dataset": len(ranked),
        "airports": [{"code": a, "total_volume": v} for a, v in selected],
    }
    with open(OUT_DIR / "selected_airports.json", "w") as f:
        json.dump(out, f, indent=2)
    return [a for a, _ in selected]


def compute_edge_correlations(selected: list):
    selected_set = set(selected)
    print(f"\nBuilding per-airport delay time series (restricted to {len(selected)} selected airports)...", flush=True)

    node_frames = []
    for f in sorted(NODE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["AIRPORT", "WINDOW_START", "MEAN_ARR_DELAY"])
        df = df[df["AIRPORT"].isin(selected_set)]
        node_frames.append(df)
    node_all = pd.concat(node_frames, ignore_index=True)
    del node_frames

    # Pivot to one column per airport, one row per time window.
    delay_matrix = node_all.pivot_table(index="WINDOW_START", columns="AIRPORT", values="MEAN_ARR_DELAY", observed=True)
    delay_matrix = delay_matrix.sort_index()
    print(f"Delay time-series matrix: {delay_matrix.shape[0]:,} windows x {delay_matrix.shape[1]} airports", flush=True)

    # Lag every airport's series by one window (4 hours) to get the
    # "one window ago" side of the correlation. Break the lag across any
    # gap wider than exactly 4 hours -- the dataset has whole missing
    # months (all of 2019, plus 11 more; see README's data-availability
    # limitation) and this must not be silently bridged into a fake
    # "consecutive" pair, or the correlation would mix two unrelated points
    # in time.
    time_gaps = delay_matrix.index.to_series().diff()
    valid_lag = time_gaps == pd.Timedelta(hours=4)
    lagged_matrix = delay_matrix.shift(1)
    lagged_matrix.loc[~valid_lag] = np.nan
    print(f"Valid consecutive-window pairs for lag correlation: {valid_lag.sum():,} / {len(delay_matrix):,}", flush=True)

    print("Finding all routes that appear between selected airports...", flush=True)
    route_frames = []
    for f in sorted(EDGE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["ORIGIN", "DEST"])
        df = df[df["ORIGIN"].isin(selected_set) & df["DEST"].isin(selected_set)]
        route_frames.append(df.drop_duplicates())
    all_routes = pd.concat(route_frames, ignore_index=True).drop_duplicates()
    print(f"Unique routes between selected airports: {len(all_routes):,}", flush=True)

    print("Computing pairwise lagged (origin[t-1] -> dest[t]) delay correlation per route...", flush=True)
    # One combined correlation matrix -- pandas' pairwise-complete NaN
    # handling in .corr() does the per-pair "drop rows where either side
    # is missing" work for us, same as the original contemporaneous version.
    combined = pd.concat([delay_matrix.add_suffix("_now"), lagged_matrix.add_suffix("_lag")], axis=1)
    corr_full = combined.corr()

    correlations = []
    for origin, dest in all_routes.itertuples(index=False):
        o_col, d_col = f"{origin}_lag", f"{dest}_now"
        c = corr_full.loc[o_col, d_col] if o_col in corr_full.index and d_col in corr_full.columns else np.nan
        correlations.append((origin, dest, c))

    result = pd.DataFrame(correlations, columns=["ORIGIN", "DEST", "HISTORICAL_DELAY_CORR"])
    result["HISTORICAL_DELAY_CORR"] = result["HISTORICAL_DELAY_CORR"].fillna(0.0)
    result.to_parquet(OUT_DIR / "edge_historical_correlation.parquet", index=False)
    print(f"Wrote {len(result):,} route correlations to edge_historical_correlation.parquet", flush=True)
    print(f"Correlation distribution: mean={result['HISTORICAL_DELAY_CORR'].mean():.3f}, "
          f"min={result['HISTORICAL_DELAY_CORR'].min():.3f}, max={result['HISTORICAL_DELAY_CORR'].max():.3f}")


def main():
    selected = select_airports()
    compute_edge_correlations(selected)


if __name__ == "__main__":
    main()

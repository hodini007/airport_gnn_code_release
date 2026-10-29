"""
Windowed data loading for training. One training "batch" = one time window:
the encoder runs once over that window's K-snapshot history to get all 200
airports' embeddings, then every real flight departing in that window reuses
those same embeddings to build its own flight representation. This avoids
re-running the (expensive) GAT+TCN pass separately for every one of the
~3,000 flights that typically share a single 4-hour window.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .heads import CAUSE_TO_IDX

GRAPH_DIR = Path("./data_processed/graph")
SEQUENCE_DIR = GRAPH_DIR / "sequence"
ENRICHED_DIR = Path("./data_processed/monthly_enriched")

WINDOW_MINUTES = 240  # must match scripts/07_build_node_edge_aggregation.py

FLIGHT_NUMERIC_COLS = ["ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN", "CRS_ELAPSED_TIME", "DISTANCE"]

# scripts/14_diagnose_head_c_probabilities.py found real (if losing) signal
# for WEATHER_DELAY -- true-weather flights average 3.6x the population
# baseline probability, some confidently so -- unlike SECURITY_DELAY, which
# never exceeds 0.265 probability on any test flight and isn't oversampled
# here. Repeating each weather-delay flight this many times in TRAINING
# windows only (never validation/test) gives the model more actual gradient
# terms for the class instead of one term scaled higher, which is more
# stable than pushing the loss weight further. See CAUSE_CLASS_WEIGHTS in
# model/heads.py for the matching reduction in weather's loss weight, so
# the two mechanisms don't compound.
WEATHER_OVERSAMPLE_FACTOR = 3


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    v = series.astype("float64")
    v = v.where(v != 2400, 0)
    return (v // 100) * 60 + (v % 100)


def load_airport_index() -> dict:
    with open(SEQUENCE_DIR / "airport_index.json") as f:
        order = json.load(f)["airport_order"]
    return {code: i for i, code in enumerate(order)}


def load_carrier_index() -> dict:
    """Fixed carrier vocabulary, built once from the full dataset so the
    same carrier always gets the same index regardless of which month is
    being trained on."""
    carriers = set()
    for f in sorted(ENRICHED_DIR.glob("*.parquet")):
        carriers.update(pd.read_parquet(f, columns=["OP_UNIQUE_CARRIER"])["OP_UNIQUE_CARRIER"].unique())
    return {c: i for i, c in enumerate(sorted(carriers))}


def add_departure_window(df: pd.DataFrame) -> pd.DataFrame:
    fl_date = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    dep_source = df["DEP_TIME"].where(df["DEP_TIME"].notna(), df["CRS_DEP_TIME"])
    dep_min = hhmm_to_minutes(dep_source)
    window_min = (dep_min // WINDOW_MINUTES) * WINDOW_MINUTES
    df = df.copy()
    df["WINDOW_START"] = fl_date + pd.to_timedelta(window_min, unit="m")
    return df


class MonthWindowSource:
    """
    Holds one month's graph snapshots plus enough tail history from the
    previous month (K-1 snapshots) so every window in this month has a full
    K-length history available, and the matching flight-level data with
    departure windows assigned.
    """

    def __init__(self, month_key: str, K: int = 12):
        self.month_key = month_key
        self.K = K

        graphs = torch.load(SEQUENCE_DIR / f"{month_key}.pt", weights_only=False)
        self.window_starts = [g.window_start for g in graphs]

        prev_key = self._previous_month_key(month_key)
        prev_path = SEQUENCE_DIR / f"{prev_key}.pt"
        history_pad = torch.load(prev_path, weights_only=False)[-(K - 1):] if prev_path.exists() else []

        self.full_sequence = history_pad + graphs
        self.offset = len(history_pad)  # index of this month's first real window within full_sequence

        flights = pd.read_parquet(ENRICHED_DIR / f"{month_key}.parquet")
        self.flights = add_departure_window(flights)
        self.window_to_flight_idx = {w: i for i, w in enumerate(self.window_starts)}

    @staticmethod
    def _previous_month_key(month_key: str) -> str:
        year, month = int(month_key[:4]), int(month_key[5:7])
        if month == 1:
            return f"{year - 1:04d}-12"
        return f"{year:04d}-{month - 1:02d}"

    def n_windows(self) -> int:
        return len(self.window_starts)

    def get_window(self, window_idx: int):
        """Returns (list of K PyG Data objects ending at this window, flights_df for this window), or None if not enough history exists."""
        end = self.offset + window_idx + 1
        start = end - self.K
        if start < 0:
            return None
        snapshots = self.full_sequence[start:end]
        window_start = self.window_starts[window_idx]
        flights_in_window = self.flights[self.flights["WINDOW_START"] == window_start]
        return snapshots, flights_in_window


def build_flight_tensors(flights_df: pd.DataFrame, airport_to_idx: dict, carrier_to_idx: dict, device="cpu", oversample_weather=False):
    """Converts one window's flights DataFrame into the tensors the model needs. Drops flights touching an airport outside the selected top-200.

    oversample_weather: repeat WEATHER_DELAY flights WEATHER_OVERSAMPLE_FACTOR
    times. Pass True only for training batches -- never for validation/test,
    since duplicated rows would distort the reported metrics.
    """
    df = flights_df[flights_df["ORIGIN"].isin(airport_to_idx) & flights_df["DEST"].isin(airport_to_idx)]
    if len(df) == 0:
        return None

    if oversample_weather:
        weather_rows = df[df["OWN_DOMINANT_CAUSE"].astype(str) == "WEATHER_DELAY"]
        if len(weather_rows) > 0:
            df = pd.concat([df] + [weather_rows] * (WEATHER_OVERSAMPLE_FACTOR - 1), ignore_index=True)

    origin_idx = torch.tensor([airport_to_idx[o] for o in df["ORIGIN"]], dtype=torch.long, device=device)
    dest_idx = torch.tensor([airport_to_idx[d] for d in df["DEST"]], dtype=torch.long, device=device)
    carrier_idx = torch.tensor([carrier_to_idx.get(c, 0) for c in df["OP_UNIQUE_CARRIER"]], dtype=torch.long, device=device)
    prev_cause_idx = torch.tensor([CAUSE_TO_IDX.get(str(c), CAUSE_TO_IDX["NO_PREV_LEG"]) for c in df["ROTATION_PREV_CAUSE"]], dtype=torch.long, device=device)

    numeric = df[FLIGHT_NUMERIC_COLS].fillna(0.0).values.astype("float32")
    flight_numeric = torch.tensor(numeric, dtype=torch.float32, device=device)

    true_delay = torch.tensor(df["ARR_DELAY"].fillna(0.0).values, dtype=torch.float32, device=device)
    true_risk = torch.tensor(df["ARR_DEL15"].fillna(0.0).values, dtype=torch.float32, device=device)

    own_cause = df["OWN_DOMINANT_CAUSE"].astype(str).values
    is_delayed = true_risk == 1
    cause_label = torch.tensor(
        [CAUSE_TO_IDX.get(c, -1) if c not in ("NOT_DELAYED",) else -1 for c in own_cause],
        dtype=torch.long, device=device,
    )

    return {
        "origin_idx": origin_idx, "dest_idx": dest_idx, "carrier_idx": carrier_idx,
        "prev_cause_idx": prev_cause_idx, "flight_numeric": flight_numeric,
        "true_delay": true_delay, "true_risk": true_risk,
        "cause_label": cause_label, "is_delayed": is_delayed,
    }

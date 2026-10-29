"""
Classical tabular baseline, per Methodology_Refinements_PreImplementation.docx
Section 3's baseline protocol: "Standard library implementation on the same
engineered features, no graph structure."

Trains three separate XGBoost models (one per head) on the same underlying
signal the GNN has access to -- each airport's own per-window operational
stats (the exact node features fed to GAT) plus a 12-window rolling mean as
a temporal-context substitute for the TCN, plus the same flight-level
features used by FlightRepresentation -- but with NO graph message passing
at all: ORIGIN and DEST are just flattened tabular columns.

Same time-based split as the main model (train 2019-01..2023-12, val
2024-01..2024-06, test 2025-01..2025-06), same held-out test set, same
metric set (RMSE/MAE, AUC-ROC/AUC-PR/F1, accuracy/macro-F1).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.dataset import add_departure_window, load_airport_index
from model.heads import CAUSE_VOCAB, CAUSE_CLASS_WEIGHTS, CAUSE_TO_IDX

NODE_DIR = Path("./data_processed/node_features")
ENRICHED_DIR = Path("./data_processed/monthly_enriched")

TRAIN_MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2024) for m in range(1, 13)]
VAL_MONTHS = [f"2024-{m:02d}" for m in range(1, 7)]
TEST_MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]

NODE_COLS = ["N_DEPARTURES", "N_ARRIVALS", "MEAN_DEP_DELAY", "STD_DEP_DELAY",
             "MEAN_ARR_DELAY", "STD_ARR_DELAY", "ARR_DEL15_RATE", "MEAN_TAXI_OUT", "MEAN_TAXI_IN"]
ROLL_COLS = ["MEAN_DEP_DELAY", "MEAN_ARR_DELAY", "ARR_DEL15_RATE"]
FLIGHT_NUMERIC_COLS = ["ROTATION_PREV_ARR_DELAY", "ROTATION_TURNAROUND_MIN", "CRS_ELAPSED_TIME", "DISTANCE"]
K = 12


def build_node_lookup(selected_airports: set) -> pd.DataFrame:
    """One row per (AIRPORT, WINDOW_START): the same 9 features GAT sees for
    that snapshot, plus a K=12-window trailing rolling mean of 3 key delay
    stats as a no-graph, no-GAT/TCN substitute for temporal context."""
    print("Building node feature + rolling-mean lookup table...", flush=True)
    frames = []
    for f in sorted(NODE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["AIRPORT", "WINDOW_START"] + NODE_COLS)
        df = df[df["AIRPORT"].isin(selected_airports)].copy()
        df["SRC_MONTH"] = f.stem  # boundary windows appear in two monthly files
        frames.append(df)
    node_all = pd.concat(frames, ignore_index=True)
    node_all["AIRPORT"] = node_all["AIRPORT"].astype(str)
    node_all = node_all.sort_values(["AIRPORT", "WINDOW_START"])

    roll = (
        node_all.groupby("AIRPORT")[ROLL_COLS]
        .rolling(window=K, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    roll.columns = [f"{c}_ROLL12" for c in ROLL_COLS]
    node_all = pd.concat([node_all, roll], axis=1)
    print(f"  {len(node_all):,} (airport, window) rows", flush=True)
    return node_all


def build_month_features(month_key: str, airport_to_idx: dict, node_lookup: pd.DataFrame) -> pd.DataFrame:
    flights = pd.read_parquet(ENRICHED_DIR / f"{month_key}.parquet")
    flights = add_departure_window(flights)
    flights = flights[flights["ORIGIN"].isin(airport_to_idx) & flights["DEST"].isin(airport_to_idx)]
    if len(flights) == 0:
        return flights.iloc[0:0]
    n_before = len(flights)
    # Use only this month's own node-feature file (as the neural pipeline does); the same
    # boundary window exists in two monthly files and would otherwise duplicate flights.
    node_lookup = node_lookup[node_lookup["SRC_MONTH"] == month_key]

    origin_feats = node_lookup.rename(columns={"AIRPORT": "ORIGIN"}).add_prefix("ORIGIN_")
    origin_feats = origin_feats.rename(columns={"ORIGIN_ORIGIN": "ORIGIN", "ORIGIN_WINDOW_START": "WINDOW_START"})
    dest_feats = node_lookup.rename(columns={"AIRPORT": "DEST"}).add_prefix("DEST_")
    dest_feats = dest_feats.rename(columns={"DEST_DEST": "DEST", "DEST_WINDOW_START": "WINDOW_START"})

    flights = flights.merge(origin_feats, on=["ORIGIN", "WINDOW_START"], how="left")
    flights = flights.merge(dest_feats, on=["DEST", "WINDOW_START"], how="left")

    keep_cols = (
        ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "ROTATION_PREV_CAUSE", "OWN_DOMINANT_CAUSE",
         "ARR_DELAY", "ARR_DEL15"]
        + FLIGHT_NUMERIC_COLS
        + [f"ORIGIN_{c}" for c in NODE_COLS] + [f"ORIGIN_{c}_ROLL12" for c in ROLL_COLS]
        + [f"DEST_{c}" for c in NODE_COLS] + [f"DEST_{c}_ROLL12" for c in ROLL_COLS]
    )
    assert len(flights) == n_before, f"{month_key}: merge changed the flight count ({n_before} -> {len(flights)})"
    out = flights[keep_cols].copy()
    for c in FLIGHT_NUMERIC_COLS:
        out[c] = out[c].fillna(0.0).astype("float32")
    for c in out.columns:
        if c.startswith("ORIGIN_") or c.startswith("DEST_"):
            out[c] = out[c].fillna(0.0).astype("float32")
    out["ARR_DELAY"] = out["ARR_DELAY"].fillna(0.0).astype("float32")
    out["ARR_DEL15"] = out["ARR_DEL15"].fillna(0.0).astype("int8")
    for c in ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "ROTATION_PREV_CAUSE", "OWN_DOMINANT_CAUSE"]:
        out[c] = out[c].astype(str).astype("category")
    return out


def load_split(months: list, airport_to_idx: dict, node_lookup: pd.DataFrame, label: str) -> pd.DataFrame:
    frames = []
    for month_key in months:
        t0 = time.time()
        df = build_month_features(month_key, airport_to_idx, node_lookup)
        frames.append(df)
        print(f"  [{label}] {month_key}: {len(df):,} flights ({time.time() - t0:.1f}s)", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    cat_cols = ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "ROTATION_PREV_CAUSE", "OWN_DOMINANT_CAUSE"]
    for c in cat_cols:
        combined[c] = combined[c].astype("category")
    print(f"[{label}] total: {len(combined):,} flights", flush=True)
    return combined


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "ROTATION_PREV_CAUSE"]
    numeric_cols = (
        FLIGHT_NUMERIC_COLS
        + [f"ORIGIN_{c}" for c in NODE_COLS] + [f"ORIGIN_{c}_ROLL12" for c in ROLL_COLS]
        + [f"DEST_{c}" for c in NODE_COLS] + [f"DEST_{c}_ROLL12" for c in ROLL_COLS]
    )
    return df[cat_cols + numeric_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-months", nargs="*", default=None, help="Override for a quick verification run, e.g. --train-months 2020-01 2020-02")
    parser.add_argument("--val-months", nargs="*", default=None)
    parser.add_argument("--test-months", nargs="*", default=None)
    args = parser.parse_args()
    train_months = args.train_months if args.train_months else TRAIN_MONTHS
    val_months = args.val_months if args.val_months else VAL_MONTHS
    test_months = args.test_months if args.test_months else TEST_MONTHS

    airport_to_idx = load_airport_index()
    selected = set(airport_to_idx)
    node_lookup = build_node_lookup(selected)

    train_df = load_split(train_months, airport_to_idx, node_lookup, "train")
    val_df = load_split(val_months, airport_to_idx, node_lookup, "val")
    test_df = load_split(test_months, airport_to_idx, node_lookup, "test")

    # Align categorical codes across splits (XGBoost needs consistent category sets).
    cat_cols = ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "ROTATION_PREV_CAUSE"]
    for c in cat_cols:
        all_cats = pd.api.types.union_categoricals(
            [train_df[c], val_df[c], test_df[c]]
        ).categories
        train_df[c] = train_df[c].cat.set_categories(all_cats)
        val_df[c] = val_df[c].cat.set_categories(all_cats)
        test_df[c] = test_df[c].cat.set_categories(all_cats)

    X_train, X_val, X_test = feature_matrix(train_df), feature_matrix(val_df), feature_matrix(test_df)

    common_params = dict(tree_method="hist", enable_categorical=True, device="cuda", n_estimators=300,
                          max_depth=8, learning_rate=0.1, early_stopping_rounds=20)

    # ---- Head A: delay magnitude ----
    print("\nTraining Head A (XGBRegressor)...", flush=True)
    reg = xgb.XGBRegressor(**common_params, objective="reg:squarederror", eval_metric="rmse")
    reg.fit(X_train, train_df["ARR_DELAY"], eval_set=[(X_val, val_df["ARR_DELAY"])], verbose=50)
    pred_delay = reg.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(test_df["ARR_DELAY"], pred_delay)))
    mae = float(mean_absolute_error(test_df["ARR_DELAY"], pred_delay))

    # ---- Head B: delay risk ----
    print("\nTraining Head B (XGBClassifier, binary)...", flush=True)
    clf_b = xgb.XGBClassifier(**common_params, objective="binary:logistic", eval_metric="logloss")
    clf_b.fit(X_train, train_df["ARR_DEL15"], eval_set=[(X_val, val_df["ARR_DEL15"])], verbose=50)
    pred_risk_prob = clf_b.predict_proba(X_test)[:, 1]
    pred_risk_label = (pred_risk_prob >= 0.5).astype(int)
    auc_roc = float(roc_auc_score(test_df["ARR_DEL15"], pred_risk_prob))
    auc_pr = float(average_precision_score(test_df["ARR_DEL15"], pred_risk_prob))
    f1 = float(f1_score(test_df["ARR_DEL15"], pred_risk_label))
    precision = float(precision_score(test_df["ARR_DEL15"], pred_risk_label))
    recall = float(recall_score(test_df["ARR_DEL15"], pred_risk_label))

    # ---- Head C: cause (delayed flights only), class-weighted to match the main model ----
    print("\nTraining Head C (XGBClassifier, 5-class, delayed flights only)...", flush=True)
    train_delayed = train_df[train_df["ARR_DEL15"] == 1]
    val_delayed = val_df[val_df["ARR_DEL15"] == 1]
    test_delayed = test_df[test_df["ARR_DEL15"] == 1]
    X_train_c = feature_matrix(train_delayed)
    X_val_c = feature_matrix(val_delayed)
    X_test_c = feature_matrix(test_delayed)

    # .astype(str) first: mapping a categorical Series can return a
    # categorical result, which then can't be compared with `< 5` below.
    y_train_c = train_delayed["OWN_DOMINANT_CAUSE"].astype(str).map(CAUSE_TO_IDX).astype(float)
    y_val_c = val_delayed["OWN_DOMINANT_CAUSE"].astype(str).map(CAUSE_TO_IDX).astype(float)
    y_test_c = test_delayed["OWN_DOMINANT_CAUSE"].astype(str).map(CAUSE_TO_IDX).astype(float)
    keep_train = y_train_c.notna() & (y_train_c < 5)
    keep_val = y_val_c.notna() & (y_val_c < 5)
    keep_test = y_test_c.notna() & (y_test_c < 5)
    X_train_c, y_train_c = X_train_c[keep_train], y_train_c[keep_train].astype(int)
    X_val_c, y_val_c = X_val_c[keep_val], y_val_c[keep_val].astype(int)
    X_test_c, y_test_c = X_test_c[keep_test], y_test_c[keep_test].astype(int)

    sample_weight = y_train_c.map(lambda i: CAUSE_CLASS_WEIGHTS[i]).values
    clf_c = xgb.XGBClassifier(**common_params, objective="multi:softprob", num_class=5, eval_metric="mlogloss")
    clf_c.fit(X_train_c, y_train_c, sample_weight=sample_weight, eval_set=[(X_val_c, y_val_c)], verbose=50)
    pred_cause = clf_c.predict(X_test_c)

    cause_accuracy = float(accuracy_score(y_test_c, pred_cause))
    cause_report = classification_report(y_test_c, pred_cause, target_names=CAUSE_VOCAB[:5], output_dict=True, zero_division=0)
    macro_f1 = float(cause_report["macro avg"]["f1-score"])

    print("\n" + "=" * 60)
    print(f"XGBOOST BASELINE -- test months {test_months}")
    print("=" * 60)
    print(f"\nHEAD A -- Delay magnitude (regression)")
    print(f"  RMSE: {rmse:.2f} minutes")
    print(f"  MAE:  {mae:.2f} minutes")
    print(f"\nHEAD B -- Delay risk (binary classification)")
    print(f"  AUC-ROC:          {auc_roc:.4f}")
    print(f"  AUC-PR:           {auc_pr:.4f}")
    print(f"  F1 (@ 0.5):       {f1:.4f}")
    print(f"  Precision (@0.5): {precision:.4f}")
    print(f"  Recall (@0.5):    {recall:.4f}")
    print(f"\nHEAD C -- Dominant cause (5-class, delayed flights only)")
    print(f"  Accuracy:  {cause_accuracy:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    for cause in CAUSE_VOCAB[:5]:
        stats = cause_report.get(cause, {})
        print(f"    {cause:22s} precision={stats.get('precision', 0):.3f} recall={stats.get('recall', 0):.3f} "
              f"f1={stats.get('f1-score', 0):.3f} support={int(stats.get('support', 0))}")
    print("=" * 60)

    report = {
        "test_months": test_months,
        "head_a": {"rmse_minutes": rmse, "mae_minutes": mae},
        "head_b": {
            "auc_roc": auc_roc, "auc_pr": auc_pr, "f1_at_0.5": f1,
            "precision_at_0.5": precision, "recall_at_0.5": recall,
        },
        "head_c": {
            "accuracy": cause_accuracy, "macro_f1": macro_f1,
            "per_class": {
                cause: {
                    "precision": cause_report.get(cause, {}).get("precision", 0),
                    "recall": cause_report.get(cause, {}).get("recall", 0),
                    "f1": cause_report.get(cause, {}).get("f1-score", 0),
                    "support": int(cause_report.get(cause, {}).get("support", 0)),
                }
                for cause in CAUSE_VOCAB[:5]
            },
        },
    }
    out_path = Path("./data_processed/checkpoints/xgboost_baseline_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

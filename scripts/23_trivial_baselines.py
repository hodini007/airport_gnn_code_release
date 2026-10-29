"""
Trivial reference baselines for Table 2, computed on exactly the population the
neural models are scored on (flights between two top-200 airports whose
departure window exists in the graph sequence; same filtering as
model.dataset.build_flight_tensors).

  Head A: always predict the training-set mean arrival delay.
  Head B: a constant score carries no ranking information (AUC-ROC 0.5,
          AUC-PR = base rate); at the 0.5 threshold a constant below 0.5
          predicts "on time" for everyone (F1 = 0).
  Head C: always predict the most common cause among delayed flights.

Also reports the XGBoost-style population (all top-200-airport flights, no
window requirement) so the size difference between the two test sets is explicit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.dataset import MonthWindowSource, build_flight_tensors, load_airport_index, load_carrier_index, ENRICHED_DIR
from model.heads import CAUSE_VOCAB

TRAIN_MONTHS = [f"{y}-{m:02d}" for y in range(2019, 2024) for m in range(1, 13)]
TEST_MONTHS = [f"2025-{m:02d}" for m in range(1, 7)]
OUT_PATH = Path("data_processed/checkpoints/trivial_baselines_report.json")


def main():
    airport_to_idx = load_airport_index()
    carrier_to_idx = load_carrier_index()

    s, n = 0.0, 0
    for m in TRAIN_MONTHS:
        df = pd.read_parquet(ENRICHED_DIR / f"{m}.parquet", columns=["ORIGIN", "DEST", "ARR_DELAY"])
        df = df[df["ORIGIN"].isin(airport_to_idx) & df["DEST"].isin(airport_to_idx)]
        v = df["ARR_DELAY"].fillna(0.0).astype("float64")
        s += float(v.sum()); n += len(v)
    train_mean = s / n
    print(f"training mean arrival delay: {train_mean:.3f} min over {n:,} flights", flush=True)

    delay, risk, cause = [], [], []
    per_month_neural, per_month_all = {}, {}
    for m in TEST_MONTHS:
        raw = pd.read_parquet(ENRICHED_DIR / f"{m}.parquet", columns=["ORIGIN", "DEST"])
        per_month_all[m] = int((raw["ORIGIN"].isin(airport_to_idx) & raw["DEST"].isin(airport_to_idx)).sum())
        src = MonthWindowSource(m, K=12)
        cnt = 0
        for w in range(src.n_windows()):
            r = src.get_window(w)
            if r is None or len(r[1]) == 0:
                continue
            b = build_flight_tensors(r[1], airport_to_idx, carrier_to_idx, device="cpu")
            if b is None:
                continue
            delay.append(b["true_delay"].numpy()); risk.append(b["true_risk"].numpy())
            keep = (b["is_delayed"] & (b["cause_label"] >= 0)).numpy()
            cause.append(b["cause_label"].numpy()[keep]); cnt += len(b["true_delay"])
        per_month_neural[m] = cnt
        print(f"  {m}: neural population {cnt:,} | all top-200 flights {per_month_all[m]:,}", flush=True)

    delay = np.concatenate(delay); risk = np.concatenate(risk); cause = np.concatenate(cause)
    err = delay - train_mean
    rmse = float(np.sqrt(np.mean(err ** 2))); mae = float(np.mean(np.abs(err)))
    base_rate = float(risk.mean())
    counts = np.bincount(cause, minlength=5)
    maj = int(counts.argmax())
    pred = np.full_like(cause, maj)
    acc = float(accuracy_score(cause, pred))
    macro = float(f1_score(cause, pred, average="macro", labels=list(range(5)), zero_division=0))
    report = {
        "train_mean_delay_min": train_mean,
        "n_test_flights_neural_population": int(len(delay)),
        "n_test_flights_all_top200": int(sum(per_month_all.values())),
        "per_month_neural": per_month_neural, "per_month_all_top200": per_month_all,
        "head_a_mean_predictor": {"rmse_minutes": rmse, "mae_minutes": mae},
        "head_b_constant": {"auc_roc": 0.5, "auc_pr": base_rate, "base_rate": base_rate, "f1_at_0.5": 0.0},
        "head_c_majority": {"majority_class": CAUSE_VOCAB[maj], "accuracy": acc, "macro_f1": macro,
                            "n_delayed_with_cause": int(len(cause)), "class_counts": counts.tolist()},
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

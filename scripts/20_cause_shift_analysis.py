"""
RQ2 direct measurement: "As delay propagates from one flight to the next,
does the dominant cause category itself change along the propagation
chain?" (README.md, Research questions).

Everything built so far around cause-shift (scripts/06_cause_label_
consistency_check.py, scripts/10_add_previous_cause_feature.py, the
previous-cause input to Head C) either validates the labels or *uses*
cause-shift as a helpful model input -- none of it directly answers RQ2 as
its own reported result. This script does that: a transition-matrix
analysis, run once on the full dataset (a data-description question, not a
model-evaluation one, so it uses every available month, the same
precedent as the cause-label consistency check).

Population: consecutive same-aircraft legs (TAIL_NUM rotation, same
20-hour staleness rule as scripts/05 and 10) where BOTH the previous leg
and the current leg were actually delayed with a real attributed cause --
i.e. a delay genuinely propagated from one leg to the next, so it is
meaningful to ask whether the attributed *cause* stayed the same or
changed. Rows where either leg was on time, or had no trackable previous
leg, are excluded -- there is nothing to "shift" if nothing propagated.

Reports:
  - The 5x5 transition matrix P(current cause = j | previous cause = i).
  - The unconditional (marginal) distribution of the current cause in this
    same population, for comparison against each transition row.
  - A chi-square test of independence between previous and current cause --
    if cause were unrelated to what happened on the previous leg, this
    would come back non-significant.
  - The same-cause persistence rate (diagonal of the transition matrix) as
    the single headline number.
  - The one pattern hypothesized in advance: how often a non-LATE_AIRCRAFT
    previous cause resolves to LATE_AIRCRAFT_DELAY on the current leg --
    BTS attributes cause per-leg, so any prior delay to the same aircraft,
    whatever its root cause, mechanically tends to show up downstream as
    "late aircraft" rather than carrying the original label forward.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ENRICHED_DIR = Path("./data_processed/monthly_enriched")
OUT_PATH = Path("./data_processed/cause_shift_report.json")

REAL_CAUSES = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
NEEDED_COLS = ["OWN_DOMINANT_CAUSE", "ROTATION_PREV_CAUSE"]


def load_propagating_chains() -> pd.DataFrame:
    files = sorted(ENRICHED_DIR.glob("*.parquet"))
    print(f"Loading cause columns from {len(files)} months...", flush=True)
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=NEEDED_COLS)
        df = df[df["OWN_DOMINANT_CAUSE"].isin(REAL_CAUSES) & df["ROTATION_PREV_CAUSE"].isin(REAL_CAUSES)]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Chains where both the previous and current leg were delayed with a real cause: {len(combined):,}", flush=True)
    return combined


def main():
    df = load_propagating_chains()

    prev = df["ROTATION_PREV_CAUSE"].astype(str)
    curr = df["OWN_DOMINANT_CAUSE"].astype(str)

    counts = pd.crosstab(prev, curr, dropna=False).reindex(index=REAL_CAUSES, columns=REAL_CAUSES, fill_value=0)
    transition = counts.div(counts.sum(axis=1), axis=0)

    marginal = curr.value_counts(normalize=True).reindex(REAL_CAUSES)

    chi2, p_value, dof, _ = chi2_contingency(counts.values)

    same_cause_mask = (curr == prev).values
    same_cause_rate = float(same_cause_mask.mean())

    non_late_aircraft_prev_mask = (prev != "LATE_AIRCRAFT_DELAY").values
    becomes_late_aircraft_rate = float((curr[non_late_aircraft_prev_mask] == "LATE_AIRCRAFT_DELAY").mean())

    print("\n" + "=" * 70)
    print("TRANSITION MATRIX  P(current cause = column | previous cause = row)")
    print(transition.round(3).to_string())
    print("\nMarginal (unconditional) distribution of the current cause:")
    print(marginal.round(3).to_string())
    print(f"\nChi-square test of independence: chi2={chi2:.1f}, dof={dof}, p={p_value:.3e}")
    print(f"  -> {'REJECTS' if p_value < 0.001 else 'does not reject'} independence: previous-leg cause "
          f"{'does' if p_value < 0.001 else 'does not'} predict current-leg cause beyond chance.")
    print(f"\nSame-cause persistence rate (diagonal): {same_cause_rate:.1%}")
    print(f"  -> the attributed cause CHANGES in {1 - same_cause_rate:.1%} of propagating chains.")
    print(f"\nNon-LATE_AIRCRAFT previous cause -> LATE_AIRCRAFT_DELAY current cause: {becomes_late_aircraft_rate:.1%}")
    print("=" * 70)

    report = {
        "n_propagating_chains": int(len(df)),
        "transition_matrix": transition.round(6).to_dict(orient="index"),
        "marginal_current_cause_distribution": marginal.round(6).to_dict(),
        "chi_square_test": {"chi2": float(chi2), "dof": int(dof), "p_value": float(p_value)},
        "same_cause_persistence_rate": same_cause_rate,
        "cause_shift_rate": 1 - same_cause_rate,
        "non_late_aircraft_prev_becomes_late_aircraft_rate": becomes_late_aircraft_rate,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()

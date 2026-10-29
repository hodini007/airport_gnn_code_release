"""
Stage 1-2: Data Acquisition, Cleaning & Schema Alignment.

Reads every monthly BTS On-Time Performance CSV in 9999/ (and its stray
333333333333/ subfolder), resolves the two known duplicate-month conflicts,
verifies each file's FL_DATE actually matches what its filename implies,
reconciles the 3 schema variants found (see scripts/01_inventory_bts_files.py),
recomputes DEP_DELAY where BTS's own field is absent, tags each row with a
COVID regime label, and writes a single cleaned Parquet dataset plus a
JSON cleaning report.

Run from WSL: python 02_clean_and_merge_bts.py
"""
import gc
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

ROOT = Path("./9999")
SUPPLEMENTAL_DIR = Path("./bts_supplemental_converted")
OUT_DIR = Path("./data_processed")
MONTHLY_DIR = OUT_DIR / "monthly"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

# Observed consistently across every schema variant: "1/1/2020 12:00:00 AM".
# Giving pandas this explicit format avoids the slow per-row dateutil fallback,
# which is what caused the WSL 9.7GB-RAM box to OOM on the first full run.
FL_DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

FNAME_RE = re.compile(r"^(\d+)1(\d{4})$")

# Columns we actually need downstream (methodology Section 2.1), read only these.
NEEDED_COLS = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "TAIL_NUM",
    "ORIGIN", "DEST", "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY",
    "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY", "ARR_DEL15",
    "TAXI_OUT", "TAXI_IN", "CANCELLED", "CANCELLATION_CODE", "DIVERTED",
    "CRS_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]

DTYPES = {
    "OP_UNIQUE_CARRIER": "category", "ORIGIN": "category", "DEST": "category",
    "TAIL_NUM": "category", "CANCELLATION_CODE": "category",
    "OP_CARRIER_FL_NUM": "Int32", "CRS_DEP_TIME": "Int32", "DEP_TIME": "Int32",
    "CRS_ARR_TIME": "Int32", "ARR_TIME": "Int32",
    "DEP_DELAY": "float32", "ARR_DELAY": "float32", "ARR_DEL15": "float32",
    "TAXI_OUT": "float32", "TAXI_IN": "float32", "CANCELLED": "float32", "DIVERTED": "float32",
    "CRS_ELAPSED_TIME": "float32", "AIR_TIME": "float32", "DISTANCE": "float32",
    "CARRIER_DELAY": "float32", "WEATHER_DELAY": "float32", "NAS_DELAY": "float32",
    "SECURITY_DELAY": "float32", "LATE_AIRCRAFT_DELAY": "float32",
}


def display_path(path: Path) -> str:
    for base in (ROOT, SUPPLEMENTAL_DIR):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def decode_filename(stem: str):
    m = FNAME_RE.match(stem)
    if not m:
        return None
    month, year = int(m.group(1)), int(m.group(2))
    if 1 <= month <= 12 and 2015 <= year <= 2026:
        return (year, month)
    return None


def decode_supplemental_filename(stem: str):
    # bts_supplemental_converted/ uses plain YYYY_MM.csv naming (see
    # scripts/04_convert_prezip_to_9999_format.py) -- these are 2019 backfill
    # months plus the 5 gap months missing from 9999/ (2021-01, 2021-11,
    # 2022-10, 2023-09, 2023-10), already rewritten to 9999/'s exact column
    # names/order/date-format and verified against the official BTS source.
    m = re.match(r"^(\d{4})_(\d{2})$", stem)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if 1 <= month <= 12:
        return (year, month)
    return None


def build_canonical_file_list():
    files = [(f, ROOT) for f in list(ROOT.glob("*.csv")) + list(ROOT.glob("*/*.csv"))]
    by_key = {}
    for f, base in files:
        key = decode_filename(f.stem)
        if key is None:
            continue
        by_key.setdefault(key, []).append(f)

    canonical = {}
    resolution_notes = []
    for key, paths in by_key.items():
        if len(paths) == 1:
            canonical[key] = paths[0]
        else:
            # Known duplicates: (2024,1) identical, (2025,1) differ only by an
            # extra unneeded column. Prefer the top-level copy (not the stray
            # 333333333333/ subfolder) in both cases.
            top_level = [p for p in paths if p.parent == ROOT]
            chosen = top_level[0] if top_level else paths[0]
            canonical[key] = chosen
            resolution_notes.append({
                "year": key[0], "month": key[1],
                "candidates": [str(p.relative_to(ROOT)) for p in paths],
                "chosen": str(chosen.relative_to(ROOT)),
            })

    # Fill in 2019 backfill + gap months from the converted-format supplemental
    # source. Never overrides a month already found in 9999/.
    if SUPPLEMENTAL_DIR.exists():
        added_from_supplemental = []
        for f in sorted(SUPPLEMENTAL_DIR.glob("*.csv")):
            key = decode_supplemental_filename(f.stem)
            if key is None or key in canonical:
                continue
            canonical[key] = f
            added_from_supplemental.append(f"{key[0]}-{key[1]:02d}")
        if added_from_supplemental:
            resolution_notes.append({
                "source": "bts_supplemental_converted",
                "months_added": added_from_supplemental,
            })

    return canonical, resolution_notes


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    v = series.astype("float64")
    v = v.where(v != 2400, 0)  # BTS uses 2400 for midnight
    hours = (v // 100)
    mins = v % 100
    return hours * 60 + mins


def recompute_dep_delay(df: pd.DataFrame) -> pd.Series:
    dep = hhmm_to_minutes(df["DEP_TIME"])
    crs = hhmm_to_minutes(df["CRS_DEP_TIME"])
    diff = dep - crs
    # Midnight wraparound: a scheduled departure late at night with an actual
    # departure just after midnight looks like a huge negative diff, and vice versa.
    diff = np.where(diff < -720, diff + 1440, diff)
    diff = np.where(diff > 720, diff - 1440, diff)
    return pd.Series(diff, index=df.index, dtype="float32")


def load_one_month(path: Path, year: int, month: int, report: dict):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in NEEDED_COLS if c in header]
    missing_needed = [c for c in NEEDED_COLS if c not in header]

    dtypes_here = {k: v for k, v in DTYPES.items() if k in usecols}
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes_here, low_memory=False)

    # FL_DATE verification against filename-implied (year, month).
    fl_date = pd.to_datetime(df["FL_DATE"], format=FL_DATE_FORMAT, errors="coerce")
    unparsed = fl_date.isna() & df["FL_DATE"].notna()
    if unparsed.any():
        # A handful of rows in a different format is plausible; fall back to the
        # general parser only for those, not the whole column.
        fl_date.loc[unparsed] = pd.to_datetime(df.loc[unparsed, "FL_DATE"], errors="coerce")
    bad_date_mask = fl_date.isna()
    mismatch_mask = (~bad_date_mask) & ((fl_date.dt.year != year) | (fl_date.dt.month != month))
    n_bad_date = int(bad_date_mask.sum())
    n_mismatch = int(mismatch_mask.sum())

    df["FL_DATE"] = fl_date
    df = df.loc[~bad_date_mask & ~mismatch_mask].copy()

    # Recompute DEP_DELAY where BTS's own field is absent (schema variant #2, 2022 Jan-Sep).
    dep_delay_recomputed = 0
    if "DEP_DELAY" not in df.columns:
        df["DEP_DELAY"] = recompute_dep_delay(df)
        dep_delay_recomputed = len(df)
    else:
        na_mask = df["DEP_DELAY"].isna() & df["DEP_TIME"].notna() & df["CRS_DEP_TIME"].notna()
        if na_mask.any():
            df.loc[na_mask, "DEP_DELAY"] = recompute_dep_delay(df.loc[na_mask])
            dep_delay_recomputed = int(na_mask.sum())

    df["SOURCE_YEAR"] = np.int16(year)
    df["SOURCE_MONTH"] = np.int8(month)

    report["files"][f"{year}-{month:02d}"] = {
        "path": display_path(path),
        "rows_read": int(len(df)) + n_bad_date + n_mismatch,
        "rows_kept": int(len(df)),
        "rows_dropped_bad_date": n_bad_date,
        "rows_dropped_date_mismatch": n_mismatch,
        "dep_delay_recomputed_rows": dep_delay_recomputed,
        "missing_columns_in_source": missing_needed,
    }
    return df


def label_regime(year: int) -> str:
    if year == 2019:
        return "Pre-COVID"
    if year in (2020, 2021):
        return "COVID-Disrupted"
    if year in (2022, 2023):
        return "Recovery"
    return "Stabilized"


def main():
    canonical, dup_notes = build_canonical_file_list()
    report = {"generated_at": datetime.now().isoformat(), "files": {}, "duplicate_resolution": dup_notes}

    print(f"Processing {len(canonical)} monthly files (one at a time, streamed to disk)...", flush=True)

    running = {
        "rows_before_exact_dedup": 0, "rows_after_exact_dedup": 0,
        "rows_dropped_corrupted_keys": 0, "final_row_count": 0,
        "missing_tail_num_count": 0, "regime_distribution": {},
        "carriers": set(), "airports": set(),
        "date_min": None, "date_max": None,
    }

    for (year, month), path in sorted(canonical.items()):
        key = f"{year}-{month:02d}"
        out_path = MONTHLY_DIR / f"{year:04d}-{month:02d}.parquet"

        if out_path.exists():
            # Resuming after an earlier crash: this month's cleaned output already
            # exists. Read it back (small, already-cleaned Parquet, not the raw
            # CSV) to recompute the aggregate stats cheaply instead of reprocessing.
            print(f"  {key}  {path.name}  (already done, reusing cached output)", flush=True)
            df = pd.read_parquet(out_path)
            rows_before_dedup = rows_after_dedup = len(df)  # not re-derivable from cache; treat as final
            n_corrupted = 0
            report["files"][key] = {
                "path": display_path(path), "rows_kept": int(len(df)),
                "resumed_from_cache": True,
            }
        else:
            print(f"  {key}  {path.name}", flush=True)
            df = load_one_month(path, year, month, report)

            rows_before_dedup = len(df)
            df = df.drop_duplicates()
            rows_after_dedup = len(df)

            key_cols = ["ORIGIN", "DEST", "FL_DATE"]
            corrupted_mask = df[key_cols].isna().any(axis=1)
            n_corrupted = int(corrupted_mask.sum())
            df = df.loc[~corrupted_mask].copy()

            df["HAS_TAIL_NUM"] = df["TAIL_NUM"].notna()
            df["REGIME"] = df["SOURCE_YEAR"].map(label_regime).astype("category")

            df.to_parquet(out_path, index=False, compression="snappy")

            report["files"][key]["rows_before_exact_dedup"] = rows_before_dedup
            report["files"][key]["exact_duplicates_removed"] = rows_before_dedup - rows_after_dedup
            report["files"][key]["rows_dropped_corrupted_keys"] = n_corrupted
            report["files"][key]["final_row_count"] = int(len(df))

        running["rows_before_exact_dedup"] += rows_before_dedup
        running["rows_after_exact_dedup"] += rows_after_dedup
        running["rows_dropped_corrupted_keys"] += n_corrupted
        running["final_row_count"] += len(df)
        running["missing_tail_num_count"] += int((~df["HAS_TAIL_NUM"]).sum())
        for regime, count in df["REGIME"].value_counts().items():
            running["regime_distribution"][str(regime)] = running["regime_distribution"].get(str(regime), 0) + int(count)
        running["carriers"].update(df["OP_UNIQUE_CARRIER"].dropna().unique().tolist())
        running["airports"].update(df["ORIGIN"].dropna().unique().tolist())
        running["airports"].update(df["DEST"].dropna().unique().tolist())
        dmin, dmax = df["FL_DATE"].min(), df["FL_DATE"].max()
        if running["date_min"] is None or dmin < running["date_min"]:
            running["date_min"] = dmin
        if running["date_max"] is None or dmax > running["date_max"]:
            running["date_max"] = dmax

        del df
        gc.collect()  # keep peak memory bounded on the 9.7GB WSL box

        # Persist progress after every file, so a crash mid-run loses at most
        # one file's worth of work instead of the whole run.
        with open(OUT_DIR / "cleaning_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

    report["summary"] = {
        "total_rows_after_load": sum(v["rows_kept"] for v in report["files"].values()),
        "rows_before_exact_dedup": running["rows_before_exact_dedup"],
        "rows_after_exact_dedup": running["rows_after_exact_dedup"],
        "exact_duplicates_removed": running["rows_before_exact_dedup"] - running["rows_after_exact_dedup"],
        "rows_dropped_corrupted_keys": running["rows_dropped_corrupted_keys"],
        "final_row_count": running["final_row_count"],
        "missing_tail_num_rate": running["missing_tail_num_count"] / running["final_row_count"] if running["final_row_count"] else 0,
        "regime_distribution": running["regime_distribution"],
        "unique_carriers": len(running["carriers"]),
        "unique_airports": len(running["airports"]),
        "date_range": [str(running["date_min"]), str(running["date_max"])],
        "output": f"Partitioned monthly Parquet files in {MONTHLY_DIR} (one per month, not a single combined file, to stay within WSL's 9.7GB RAM cap)",
    }

    report_path = OUT_DIR / "cleaning_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {len(canonical)} monthly Parquet files to {MONTHLY_DIR}")
    print(f"Wrote {report_path}")
    print(json.dumps(report["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()

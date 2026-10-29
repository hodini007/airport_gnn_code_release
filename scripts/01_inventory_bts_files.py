"""
Stage 1 (pre-step): Build a canonical inventory of the raw BTS monthly CSVs.

The files on disk are named with an undocumented pattern (e.g. "1012020.csv",
"112020.csv") and are split across the top-level 9999/ directory and a stray
subdirectory named "333333333333/". This script:
  1. Decodes (month, year) from each filename.
  2. Flags any (month, year) key that appears more than once (duplicates,
     possibly conflicting).
  3. Peeks at FL_DATE in each file to verify the decoded month/year actually
     matches the data inside (per the methodology's Stage 1-2 plan).

Run from WSL: python 01_inventory_bts_files.py
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("./9999")

FNAME_RE = re.compile(r"^(\d+)1(\d{4})$")  # month(no pad) + literal "1" + year


def decode_filename(stem: str):
    m = FNAME_RE.match(stem)
    if not m:
        return None
    month, year = int(m.group(1)), int(m.group(2))
    if 1 <= month <= 12 and 2015 <= year <= 2026:
        return (year, month)
    return None


def find_all_csvs(root: Path):
    return list(root.glob("*.csv")) + list(root.glob("*/*.csv"))


def parse_year_month(v: str):
    """Handle 'YYYY-MM-DD', 'M/D/YYYY', and 'M/D/YYYY 12:00:00 AM' style FL_DATE values."""
    v = v.strip()
    if not v:
        return None
    v = v.split(" ")[0]  # drop any trailing time-of-day component
    if "-" in v:
        parts = v.split("-")
        if len(parts) == 3 and len(parts[0]) == 4:
            return f"{parts[0]}-{int(parts[1]):02d}"
    if "/" in v:
        parts = v.split("/")
        if len(parts) == 3 and len(parts[2]) == 4:
            return f"{parts[2]}-{int(parts[0]):02d}"
    return None


def peek_fl_date_range(path: Path, sample_rows=5000):
    dates = set()
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fl_date_col = None
        for candidate in ("FL_DATE", "FlightDate"):
            if candidate in (reader.fieldnames or []):
                fl_date_col = candidate
                break
        if fl_date_col is None:
            return None, reader.fieldnames
        for i, row in enumerate(reader):
            if i >= sample_rows:
                break
            v = row.get(fl_date_col)
            ym = parse_year_month(v) if v else None
            if ym:
                dates.add(ym)
    return dates, reader.fieldnames


def compare_duplicate_pair(paths):
    """Row count + full FL_DATE range + a content hash, to decide if two files
    with the same decoded (year, month) are identical, overlapping, or distinct."""
    import hashlib

    results = []
    for p in paths:
        h = hashlib.md5()
        row_count = 0
        min_date, max_date = None, None
        with open(p, "rb") as fb:
            for chunk in iter(lambda: fb.read(1 << 20), b""):
                h.update(chunk)
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            fl_date_col = next((c for c in ("FL_DATE", "FlightDate") if c in (reader.fieldnames or [])), None)
            for row in reader:
                row_count += 1
                if fl_date_col:
                    ym = parse_year_month(row.get(fl_date_col, ""))
                    if ym:
                        if min_date is None or ym < min_date:
                            min_date = ym
                        if max_date is None or ym > max_date:
                            max_date = ym
        results.append({
            "path": p, "rows": row_count, "md5": h.hexdigest(),
            "date_range": (min_date, max_date),
        })
    return results


def main():
    files = find_all_csvs(ROOT)
    by_key = defaultdict(list)
    undecoded = []

    for f in files:
        key = decode_filename(f.stem)
        if key is None:
            undecoded.append(f)
        else:
            by_key[key].append(f)

    print(f"Total CSVs found: {len(files)}")
    print(f"Undecoded filenames: {len(undecoded)}")
    for f in undecoded:
        print(f"  UNDECODED: {f.relative_to(ROOT)}")

    print(f"\nUnique (year, month) keys: {len(by_key)}")
    dupes = {k: v for k, v in by_key.items() if len(v) > 1}
    print(f"Keys with duplicate files: {len(dupes)}")
    for key, paths in sorted(dupes.items()):
        print(f"\n  DUPLICATE (year={key[0]}, month={key[1]:02d}):")
        for p in paths:
            size = p.stat().st_size
            print(f"    {p.relative_to(ROOT)}  ({size:,} bytes)")

    print("\nCoverage check (expect Jan 2019 - some 2025 month, per methodology):")
    years = sorted(set(k[0] for k in by_key))
    for y in years:
        months = sorted(k[1] for k in by_key if k[0] == y)
        missing = sorted(set(range(1, 13)) - set(months))
        print(f"  {y}: {len(months)} months present -> {months}" + (f"  MISSING: {missing}" if missing else ""))

    print("\nFull comparison of duplicate-key file pairs (row count, MD5, full FL_DATE range):")
    for key, paths in sorted(dupes.items()):
        print(f"\n  (year={key[0]}, month={key[1]:02d}) expected: {key[0]}-{key[1]:02d}")
        results = compare_duplicate_pair(paths)
        for r in results:
            print(f"    {r['path'].relative_to(ROOT)}")
            print(f"      rows={r['rows']:,}  md5={r['md5']}  date_range={r['date_range']}")
        if len(set(r["md5"] for r in results)) == 1:
            print("      -> IDENTICAL FILES (safe to drop one)")
        elif len(set(r["rows"] for r in results)) > 1:
            print("      -> DIFFERENT ROW COUNTS: not a simple duplicate, needs manual resolution")

    print("\nFull column-schema diff across all 61 (year, month) files:")
    schema_by_key = {}
    for key, paths in by_key.items():
        p = paths[0]  # first copy is enough to characterize the schema for this month
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            header = f.readline().strip().split(",")
        schema_by_key[key] = tuple(header)

    schema_variants = defaultdict(list)
    for key, cols in schema_by_key.items():
        schema_variants[cols].append(key)

    print(f"  Distinct column-schemas found: {len(schema_variants)}")
    all_cols = set()
    for cols in schema_variants:
        all_cols |= set(cols)
    common_cols = set.intersection(*(set(c) for c in schema_variants)) if schema_variants else set()
    for i, (cols, keys) in enumerate(sorted(schema_variants.items(), key=lambda kv: -len(kv[1]))):
        extra = set(cols) - common_cols
        print(f"  Schema #{i+1} ({len(keys)} months): {sorted(keys)}")
        if extra:
            print(f"    Columns not shared by every schema: {sorted(extra)}")
    print(f"\n  Columns common to ALL {len(schema_variants)} schema variants ({len(common_cols)}): {sorted(common_cols)}")
    if all_cols - common_cols:
        print(f"  Columns present in SOME but not all files ({len(all_cols - common_cols)}): {sorted(all_cols - common_cols)}")


if __name__ == "__main__":
    main()

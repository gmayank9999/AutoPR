"""
Extracts data/raw/dataset.zip and verifies all 3 CSVs are present with the
expected schema and row counts.

Does NOT download from Drive programmatically (Drive rate-limits scripted access
to folders; manual one-time download is more reliable and takes 1 minute).

The CSVs may also be placed directly into data/raw/ if already extracted.

Usage:
    python src/download_data.py
"""
import csv
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = RAW_DIR / "dataset.zip"

EXPECTED_FILES = {
    "train.pr_commits_20_400_100_0.5_nltk.csv": 33466,
    "valid.pr_commits_20_400_100_0.5_nltk.csv":  4183,
    "test.pr_commits_20_400_100_0.5_nltk.csv":   4183,
}

MANUAL_INSTRUCTIONS = f"""
ERROR: dataset.zip not found at {ZIP_PATH}

One-time manual download (~1 minute):
  1. Open in browser:
     https://drive.google.com/drive/folders/1VMByXOEmJDQL_JQY6l63NRiveUySY-Sq
  2. Download the file named  dataset.zip  (~10 MB)
  3. Save it to: {ZIP_PATH.resolve()}
  4. Re-run this script.

Alternative (sometimes works for small public Drive zips):
  pip install gdown
  gdown --folder "https://drive.google.com/drive/folders/1VMByXOEmJDQL_JQY6l63NRiveUySY-Sq" -O data/raw/

NOTE: If the CSVs are already present in data/raw/, the script will verify them
without needing the zip.
"""


def extract_if_needed():
    # If CSVs already present, skip extraction
    all_present = all((RAW_DIR / f).exists() for f in EXPECTED_FILES)
    if all_present:
        print("CSVs already present in data/raw/ — skipping extraction.")
        return

    if not ZIP_PATH.exists():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in z.namelist():
            # Skip macOS metadata directories
            if name.startswith("__MACOSX") or name.endswith("/"):
                continue
            # Flatten any subdirectories - extract directly to RAW_DIR
            dest = RAW_DIR / Path(name).name
            if not dest.exists():
                dest.write_bytes(z.read(name))

    print(f"Extracted to: {RAW_DIR.resolve()}")


def verify():
    csv.field_size_limit(10_000_000)  # some PRs have very long articles
    ok = True
    for fname, expected_n in EXPECTED_FILES.items():
        p = RAW_DIR / fname
        if not p.exists():
            print(f"  MISSING: {p}")
            ok = False
            continue

        with open(p, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            n = sum(1 for _ in reader)

        if cols != ["id", "abstract", "article"]:
            print(f"  BAD SCHEMA in {fname}: got {cols}, expected ['id', 'abstract', 'article']")
            ok = False
            continue

        if n != expected_n:
            print(f"  WARN: {fname} has {n} rows, expected {expected_n}. Continuing.")
        print(f"  {fname}: cols={cols}  rows={n}")

    if not ok:
        sys.exit(2)
    print("OK — all CSVs present with correct schema.")


if __name__ == "__main__":
    extract_if_needed()
    verify()

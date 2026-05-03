"""
Normalizes the shipped CSVs and emits JSONL with the standard schema:

    {
      "id": str,
      "source": str,         # was "article" in CSV (commit messages + code comments)
      "target": str,         # was "abstract" in CSV (PR description)
      "num_commits": int,
      "src_len_words": int,
      "tgt_len_words": int
    }

CRITICAL column mapping (do not flip):
    CSV 'article'   ->  our 'source'   (input — commit messages)
    CSV 'abstract'  ->  our 'target'   (output — PR description)

Usage:
    python src/preprocess.py
"""
import csv
import json
import re
import sys
import statistics as st
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# Some 'article' fields are very long; default csv field limit will choke
csv.field_size_limit(10_000_000)

FILES = {
    "train": "train.pr_commits_20_400_100_0.5_nltk.csv",
    "val":   "valid.pr_commits_20_400_100_0.5_nltk.csv",
    "test":  "test.pr_commits_20_400_100_0.5_nltk.csv",
}

MULTI_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("<cm-sep>", " [COMMIT_SEP] ")
    text = text.replace("<nl>", " ")
    text = MULTI_WS.sub(" ", text).strip()
    return text


def process(split: str, fname: str) -> int:
    in_path = RAW / fname
    out_path = OUT / f"{split}.jsonl"

    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run src/download_data.py first.")
        sys.exit(1)

    kept = dropped = 0

    with open(in_path, encoding="utf-8", newline="") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        assert reader.fieldnames == ["id", "abstract", "article"], \
            f"Unexpected columns in {fname}: {reader.fieldnames}"

        for row in reader:
            src = normalize(row["article"])      # article -> source
            tgt = normalize(row["abstract"])     # abstract -> target

            src_words = src.split()
            tgt_words = tgt.split()

            if len(src_words) < 3 or len(tgt_words) < 3:
                dropped += 1
                continue

            rec = {
                "id": row["id"],
                "source": src,
                "target": tgt,
                "num_commits": src.count("[COMMIT_SEP]") + 1,
                "src_len_words": len(src_words),
                "tgt_len_words": len(tgt_words),
            }
            fout.write(json.dumps(rec) + "\n")
            kept += 1

    print(f"{split}: kept={kept:,}  dropped={dropped:,}  -> {out_path}")
    return kept


def main():
    total = 0
    for split, fname in FILES.items():
        total += process(split, fname)

    # Reproduce paper-style length stats on the training split
    src_lens, tgt_lens = [], []
    with open(OUT / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src_lens.append(r["src_len_words"])
            tgt_lens.append(r["tgt_len_words"])

    print(f"\nTrain avg source length (words): {st.mean(src_lens):.2f}   (paper reports ~76)")
    print(f"Train avg target length (words): {st.mean(tgt_lens):.2f}   (paper reports ~36)")
    print(f"Total records across splits: {total:,}")


if __name__ == "__main__":
    main()

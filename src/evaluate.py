"""
Computes ROUGE-1/2/L, BLEU-4, METEOR, BERTScore-F1, and length ratio
for every predictions file under experiments/results/.

Writes:  experiments/results/metrics_all.csv

Usage:
    python src/evaluate.py
    python src/evaluate.py --glob "e*_preds.jsonl"
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import nltk

# Quietly fetch what we need on first run
for resource in ["wordnet", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

RESULTS_DIR = Path("experiments/results")


def eval_file(path: Path):
    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouges = {"rouge1": [], "rouge2": [], "rougeL": []}
    preds, refs, meteors, ratios = [], [], [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p = (r.get("prediction") or "").strip()
            t = (r.get("target") or "").strip()
            if not t:
                continue

            s = rs.score(t, p)
            rouges["rouge1"].append(s["rouge1"].fmeasure)
            rouges["rouge2"].append(s["rouge2"].fmeasure)
            rouges["rougeL"].append(s["rougeL"].fmeasure)

            meteors.append(meteor_score([t.split()], p.split()))
            ratios.append(len(p.split()) / max(len(t.split()), 1))

            preds.append(p)
            refs.append(t)

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    P, R, F = bert_score(preds, refs, lang="en", verbose=False, batch_size=64)

    return {
        "file": path.name,
        "n": len(preds),
        "rouge1":       round(mean(rouges["rouge1"]) * 100, 2) if rouges["rouge1"] else 0.0,
        "rouge2":       round(mean(rouges["rouge2"]) * 100, 2) if rouges["rouge2"] else 0.0,
        "rougeL":       round(mean(rouges["rougeL"]) * 100, 2) if rouges["rougeL"] else 0.0,
        "bleu4":        round(bleu, 2),
        "meteor":       round(mean(meteors) * 100, 2) if meteors else 0.0,
        "bertscore_f1": round(F.mean().item() * 100, 2),
        "len_ratio":    round(mean(ratios), 3) if ratios else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*_preds.jsonl")
    args = ap.parse_args()

    files = sorted(RESULTS_DIR.glob(args.glob))
    assert files, f"No files match {args.glob} in {RESULTS_DIR}"

    rows = [eval_file(f) for f in files]

    out = RESULTS_DIR / "metrics_all.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {out}\n")
    header = f"{'file':50s}  {'R1':>5}  {'R2':>5}  {'RL':>5}  {'BLEU':>6}  {'METEOR':>7}  {'BERTS':>6}  {'lenR':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"  {r['file']:48s}  {r['rouge1']:5.2f}  {r['rouge2']:5.2f}  "
            f"{r['rougeL']:5.2f}  {r['bleu4']:6.2f}  {r['meteor']:7.2f}  "
            f"{r['bertscore_f1']:6.2f}  {r['len_ratio']:5.3f}"
        )


if __name__ == "__main__":
    main()

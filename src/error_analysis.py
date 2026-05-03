"""
Reads the predictions of our main model (E3 by default) and produces:
  - experiments/results/error_table.csv      (every test example with per-row metrics + flags)
  - experiments/results/error_summary.md     (top-5 best + top-5 worst + aggregate stats)

After running, open error_summary.md and manually fill in the "YOUR LABEL" lines
for the 5 worst cases with one of: hallucination / under-gen / over-gen / wrong-focus / hedging

Usage:
    python src/error_analysis.py --preds experiments/results/e3_lora_codet5p_preds.jsonl
"""
import argparse
import csv
import json
import re
from pathlib import Path

from rouge_score import rouge_scorer

RES = Path("experiments/results")
RES.mkdir(parents=True, exist_ok=True)

# Heuristic identifier matcher: CamelCase OR snake_case
IDENT_RE = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:_[a-z]+)+")


def idents(text: str) -> set:
    return set(IDENT_RE.findall(text or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(RES / "e3_lora_codet5p_preds.jsonl"))
    args = ap.parse_args()

    preds_path = Path(args.preds)
    assert preds_path.exists(), f"Predictions file not found: {preds_path}"

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    with open(preds_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    # Load test sources (predictions file only has id+prediction+target)
    src_map = {}
    with open("data/processed/test.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            src_map[o["id"]] = o["source"]

    records = []
    halluc = 0
    for r in rows:
        src = src_map.get(r["id"], "")
        rl = rs.score(r["target"] or "", r["prediction"] or "")["rougeL"].fmeasure

        src_ids = idents(src)
        pred_ids = idents(r["prediction"] or "")
        novel = pred_ids - src_ids

        # Heuristic: prediction introduces an identifier not in the source AND ROUGE-L is low
        is_halluc = bool(novel) and rl < 0.20
        halluc += int(is_halluc)

        records.append({
            "id": r["id"],
            "rougeL": round(rl, 4),
            "novel_idents": "|".join(sorted(novel)),
            "halluc_flag": is_halluc,
            "pred": r["prediction"],
            "target": r["target"],
            "source": src[:300],
        })

    # Write the full error table CSV
    table_path = RES / "error_table.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)

    # Top-5 best + bottom-5 worst + aggregate stats
    records.sort(key=lambda x: x["rougeL"], reverse=True)
    best = records[:5]
    worst = records[-5:]

    lines = [
        f"# Error analysis summary — {preds_path.name}\n",
        f"- Total test examples: **{len(records):,}**",
        f"- Heuristic hallucination flag rate: **{halluc / len(records) * 100:.2f}%**",
        "",
        "(Hallucination flag = prediction contains an identifier-shaped token not in the source AND ROUGE-L < 0.20.)",
        "",
        "## Top 5 best predictions\n",
    ]
    for r in best:
        lines.append(f"### `id = {r['id']}`  (ROUGE-L = {r['rougeL']})")
        lines.append(f"- **Target:** {r['target']}")
        lines.append(f"- **Prediction:** {r['pred']}")
        lines.append("")

    lines.append(
        "## Top 5 worst predictions\n"
        "> Manually label each with one of: hallucination / under-gen / over-gen / wrong-focus / hedging\n"
    )
    for r in worst:
        lines.append(f"### `id = {r['id']}`  (ROUGE-L = {r['rougeL']})")
        lines.append(f"- **Source (first 300 chars):** {r['source']}")
        lines.append(f"- **Target:** {r['target']}")
        lines.append(f"- **Prediction:** {r['pred']}")
        lines.append(f"- **Novel identifiers:** {r['novel_idents'] or '(none)'}")
        lines.append(f"- **Hallucination flag:** {r['halluc_flag']}")
        lines.append(f"- **YOUR LABEL:** _____________________")
        lines.append("")

    summ_path = RES / "error_summary.md"
    summ_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote:")
    print(f"  {table_path}")
    print(f"  {summ_path}")
    print(f"\nAggregate hallucination flag rate: {halluc / len(records) * 100:.2f}%")


if __name__ == "__main__":
    main()

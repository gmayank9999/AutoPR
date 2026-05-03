"""
Zero-shot T5-base baseline (E7).

Uses HuggingFace t5-base directly with the prefix "summarize: " and no fine-tuning.
Demonstrates the floor that fine-tuning has to beat.

Usage:
    python src/baselines/zeroshot_t5.py
"""
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

TEST = Path("data/processed/test.jsonl")
OUT = Path("experiments/results/e7_zeroshot_t5_preds.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    tok = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda").eval()

    with open(TEST, encoding="utf-8") as fin:
        rows = [json.loads(l) for l in fin]

    BATCH = 32
    with open(OUT, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(rows), BATCH), desc="Zero-shot T5"):
            chunk = rows[i:i+BATCH]
            prompts = ["summarize: " + r["source"] for r in chunk]
            inputs = tok(
                prompts,
                return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to("cuda")

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            preds = tok.batch_decode(out, skip_special_tokens=True)
            for r, p in zip(chunk, preds):
                fout.write(json.dumps({"id": r["id"], "prediction": p, "target": r["target"]}) + "\n")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

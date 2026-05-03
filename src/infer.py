"""
Batch inference on the test set, for either:
  - A LoRA adapter on top of a base model (E2-E5):
        python src/infer.py --base Salesforce/codet5p-220m \
                            --adapter experiments/runs/lora_codet5p-220m/best \
                            --out experiments/results/e3_lora_codet5p_preds.jsonl
  - A full-FT checkpoint (no adapter) (E1):
        python src/infer.py --base experiments/runs/e1_full_ft_bart_base/best \
                            --out experiments/results/e1_full_ft_bart_preds.jsonl

Output: JSONL with {id, prediction, target} per line.
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_with_adapter(base, adapter):
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(adapter)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base).to("cuda")
    model = PeftModel.from_pretrained(base_model, adapter).to("cuda").eval()
    return tok, model


def load_full_ft(checkpoint):
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda").eval()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="HuggingFace base model id, OR path to a full-FT checkpoint")
    ap.add_argument("--adapter", default=None,
                    help="Path to a saved LoRA adapter. Omit for full-FT inference.")
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()

    if args.adapter:
        tok, model = load_with_adapter(args.base, args.adapter)
        # T5-family needs the prefix; BART does not
        use_prefix = "t5" in args.base.lower()
    else:
        tok, model = load_full_ft(args.base)
        use_prefix = False  # E1 uses BART; no prefix

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.test, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    prefix = "summarize: " if use_prefix else ""

    with open(args.out, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(rows), args.batch), desc="infer"):
            chunk = rows[i:i+args.batch]
            inputs = tok(
                [prefix + r["source"] for r in chunk],
                return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            preds = tok.batch_decode(out, skip_special_tokens=True)
            for r, p in zip(chunk, preds):
                fout.write(json.dumps({"id": r["id"], "prediction": p, "target": r["target"]}) + "\n")

    print(f"Wrote {args.out} ({len(rows)} predictions)")


if __name__ == "__main__":
    main()

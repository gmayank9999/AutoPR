"""
Few-shot prompted FLAN-T5-base baseline (E8). Free, runs locally.

FLAN-T5-base is instruction-tuned, so it natively understands prompts with
in-context examples — the same setup you'd use with a paid model like GPT-4o-mini,
but local and free.

We use 2 fixed in-context examples from the training set (deterministic: first 2 rows).

Usage:
    python src/baselines/flan_t5_fewshot.py
"""
import json
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
TRAIN = Path("data/processed/train.jsonl")
TEST = Path("data/processed/test.jsonl")
OUT = Path("experiments/results/e8_flan_t5_fewshot_preds.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

INSTRUCTION = (
    "You are a developer writing a pull request description. "
    "Given commit messages, write ONE short paragraph (1-3 sentences) summarizing the changes. "
    "Match the style of the examples."
)


def load_fewshot_examples(n: int = 2):
    examples = []
    with open(TRAIN, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < n:
                examples.append(json.loads(line))
            else:
                break
    return examples


def build_prompt(test_src: str, fs_examples: list) -> str:
    parts = [INSTRUCTION, ""]
    for i, ex in enumerate(fs_examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Commits: {ex['source']}")
        parts.append(f"Description: {ex['target']}")
        parts.append("")
    parts.append("Now your turn:")
    parts.append(f"Commits: {test_src}")
    parts.append("Description:")
    return "\n".join(parts)


def main():
    fs_examples = load_fewshot_examples(2)
    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda").eval()

    with open(TEST, encoding="utf-8") as fin:
        rows = [json.loads(l) for l in fin]

    BATCH = 16
    with open(OUT, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(rows), BATCH), desc="FLAN-T5 few-shot"):
            chunk = rows[i:i+BATCH]
            prompts = [build_prompt(r["source"], fs_examples) for r in chunk]
            inputs = tok(
                prompts,
                return_tensors="pt", padding=True, truncation=True, max_length=1024,
            ).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            preds = tok.batch_decode(out, skip_special_tokens=True)
            for r, p in zip(chunk, preds):
                fout.write(json.dumps({"id": r["id"], "prediction": p.strip(), "target": r["target"]}) + "\n")

    print(f"Wrote {OUT}  ({len(rows)} predictions)")


if __name__ == "__main__":
    main()

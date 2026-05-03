"""
Full fine-tuning of facebook/bart-base on the PR description task.
Reproduces Saini et al. (2025)'s best configuration:
    lr = 2.736e-5, weight_decay = 0.1, BART-base, 5 epochs

This is the VALIDATION GATE. Run this before any LoRA experiment.
If test/rougeL lands at ~25.0 ± 2.0, the pipeline is faithful.

Usage:
    python src/train_full_ft.py
"""
import json
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from rouge_score import rouge_scorer

from src.utils import set_seed

MODEL = "facebook/bart-base"
OUT_DIR = Path("experiments/runs/e1_full_ft_bart_base")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    set_seed(42)

    tok = AutoTokenizer.from_pretrained(MODEL)
    raw = load_dataset("json", data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/val.jsonl",
        "test": "data/processed/test.jsonl",
    })

    def tokenize(batch):
        m = tok(batch["source"], max_length=512, truncation=True)
        with tok.as_target_tokenizer():
            l = tok(batch["target"], max_length=64, truncation=True)
        m["labels"] = l["input_ids"]
        return m

    ds = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        pt = tok.batch_decode(preds, skip_special_tokens=True)
        lt = tok.batch_decode(labels, skip_special_tokens=True)
        r1 = r2 = rL = 0.0
        for p, lb in zip(pt, lt):
            s = scorer.score(lb, p)
            r1 += s["rouge1"].fmeasure
            r2 += s["rouge2"].fmeasure
            rL += s["rougeL"].fmeasure
        n = max(len(pt), 1)
        return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rL / n}

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2.736e-5,            # Saini et al. (2025) optimum from W&B sweep
        weight_decay=0.1,                  # Saini et al. (2025) optimum
        warmup_ratio=0.06,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=64,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute,
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR / "best"))
    tok.save_pretrained(str(OUT_DIR / "best"))

    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n=== E1 (BART-base full FT, Saini reproduction) ===")
    print(json.dumps(test_metrics, indent=2))
    print("\nSaini et al. (2025) reported: rouge1=0.2803, rouge2=0.1685, rougeL=0.2549")
    print("Pipeline is OK if our test/rougeL is within ~0.02 of 0.2549.\n")


if __name__ == "__main__":
    main()

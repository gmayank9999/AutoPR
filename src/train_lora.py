"""
LoRA fine-tuning for Seq2Seq PR description generation.
Used for experiments E2-E5 (vary model.name in config/config.yaml between runs).

Experiment mapping:
    E2: facebook/bart-base
    E3: Salesforce/codet5p-220m  (headline experiment)
    E4: google/flan-t5-base
    E5: t5-base

Usage:
    python src/train_lora.py --config config/config.yaml
"""
import argparse
import json
import yaml
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from rouge_score import rouge_scorer

from src.utils import set_seed


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_target_modules(model_name: str):
    """LoRA target module names differ between architectures."""
    n = model_name.lower()
    if "bart" in n:
        return ["q_proj", "k_proj", "v_proj", "out_proj"]
    # T5, FLAN-T5, CodeT5+ all use the same projection names
    return ["q", "k", "v", "o"]


def build_ds(cfg, tok):
    raw = load_dataset("json", data_files={
        "train": cfg["paths"]["train"],
        "validation": cfg["paths"]["val"],
        "test": cfg["paths"]["test"],
    })

    # T5 family expects a "summarize: " prefix; BART does not.
    use_prefix = "t5" in cfg["model"]["name"].lower()
    prefix = "summarize: " if use_prefix else ""

    def tokenize(batch):
        inputs = [prefix + s for s in batch["source"]]
        m = tok(inputs, max_length=cfg["model"]["max_src_len"], truncation=True)
        with tok.as_target_tokenizer():
            lb = tok(batch["target"], max_length=cfg["model"]["max_tgt_len"], truncation=True)
        m["labels"] = lb["input_ids"]
        return m

    return raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)


def build_model(cfg):
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["name"])
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=get_target_modules(cfg["model"]["name"]),
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def make_metrics(tok):
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

    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    set_seed(cfg["train"]["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    ds = build_ds(cfg, tok)
    model = build_model(cfg)

    short = cfg["model"]["name"].split("/")[-1]
    run_name = f"lora_{short}"
    out_dir = Path(cfg["paths"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    args_tr = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        evaluation_strategy=cfg["train"]["eval_strategy"],
        save_strategy=cfg["train"]["save_strategy"],
        logging_steps=cfg["train"]["logging_steps"],
        fp16=cfg["train"]["fp16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        predict_with_generate=True,
        generation_num_beams=cfg["generation"]["num_beams"],
        generation_max_length=cfg["generation"]["max_new_tokens"],
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=cfg["train"]["seed"],
    )

    collator = DataCollatorForSeq2Seq(tok, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args_tr,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=make_metrics(tok),
    )

    trainer.train()
    trainer.save_model(str(out_dir / "best"))
    tok.save_pretrained(str(out_dir / "best"))

    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n=== LoRA fine-tuning of {cfg['model']['name']} done ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"Adapter saved to: {out_dir / 'best'}")


if __name__ == "__main__":
    main()

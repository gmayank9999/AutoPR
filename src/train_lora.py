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
    TrainerCallback, TrainerState, TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType
from rouge_score import rouge_scorer

from src.utils import set_seed, get_logger


class JsonlMetricsCallback(TrainerCallback):
    """Writes per-evaluation metrics as JSON lines to a log file."""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            row = {"step": state.global_step, "epoch": state.epoch, **metrics}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")


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


def use_fp16(model_name: str) -> bool:
    """T5-family models have known NaN loss issues with fp16; disable it."""
    n = model_name.lower()
    # BART and CodeT5+ are fine with fp16; plain T5 / FLAN-T5 are not
    if "bart" in n or "codet5" in n:
        return True
    return False  # t5-base, flan-t5 → fp32


def safe_lr(model_name: str, cfg_lr: float) -> float:
    """T5-family needs a lower LR to avoid gradient explosion with LoRA."""
    n = model_name.lower()
    if "bart" in n or "codet5" in n:
        return cfg_lr
    return min(cfg_lr, 1e-4)  # cap at 1e-4 for T5/FLAN-T5


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

    short = cfg["model"]["name"].split("/")[-1]
    run_name = f"lora_{short}"
    out_dir = Path(cfg["paths"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(run_name, log_file=str(log_dir / "train.log"))
    log.info("Starting %s  model=%s", run_name, cfg["model"]["name"])

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    log.info("Tokenizer loaded. Building dataset …")
    ds = build_ds(cfg, tok)
    log.info("Dataset ready. Train=%d  Val=%d  Test=%d",
             len(ds["train"]), len(ds["validation"]), len(ds["test"]))
    model = build_model(cfg)
    log.info("Model ready. Starting training …")

    args_tr = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["per_device_batch"],
        per_device_eval_batch_size=cfg["train"]["per_device_batch"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=safe_lr(cfg["model"]["name"], cfg["train"]["lr"]),
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        evaluation_strategy=cfg["train"]["eval_strategy"],
        save_strategy=cfg["train"]["save_strategy"],
        logging_steps=cfg["train"]["logging_steps"],
        logging_dir=str(log_dir / "tb"),       # TensorBoard event files
        fp16=use_fp16(cfg["model"]["name"]),
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        predict_with_generate=True,
        generation_num_beams=cfg["generation"]["num_beams"],
        generation_max_length=cfg["generation"]["max_new_tokens"],
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=3,               # keep best + last 2 checkpoints
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
        callbacks=[JsonlMetricsCallback(log_dir / "metrics.jsonl")],
    )

    trainer.train()
    log.info("Training done. Saving adapter to %s", out_dir / "best")
    trainer.save_model(str(out_dir / "best"))
    tok.save_pretrained(str(out_dir / "best"))

    log.info("Evaluating on test set …")
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(log_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"split": "test", **test_metrics}) + "\n")

    log.info("=== %s results ===\n%s", run_name, json.dumps(test_metrics, indent=2))
    print(f"\n=== LoRA fine-tuning of {cfg['model']['name']} done ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"Adapter saved to  : {out_dir / 'best'}")
    print(f"Training log      → {log_dir / 'train.log'}")
    print(f"Per-epoch metrics → {log_dir / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()

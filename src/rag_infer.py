"""
RAG variant for E6: retrieve k=3 similar training PRs from Chroma and prepend
them as in-context examples, then run the fine-tuned CodeT5+ LoRA adapter.

Usage:
    python src/rag_infer.py \
        --base Salesforce/codet5p-220m \
        --adapter experiments/runs/lora_codet5p-220m/best \
        --out experiments/results/e6_rag_codet5p_preds.jsonl
"""
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def trim(s: str, n_words: int) -> str:
    return " ".join(s.split()[:n_words])


def build_prompt(test_src: str, neighbors) -> str:
    parts = []
    docs = neighbors["documents"][0]
    metas = neighbors["metadatas"][0]
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        parts.append(
            f"EXAMPLE {i}\n"
            f"Commits: {trim(doc, 80)}\n"
            f"Description: {trim(meta['target'], 40)}"
        )
    ctx = "\n\n".join(parts)
    return (
        f"Here are similar past pull requests and their descriptions:\n\n{ctx}\n\n"
        f"Now summarize this pull request:\n"
        f"Commits: {test_src}\n"
        f"Description:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Salesforce/codet5p-220m")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", default="data/processed/test.jsonl")
    ap.add_argument("--out", default="experiments/results/e6_rag_codet5p_preds.jsonl")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda").eval()

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    client = chromadb.PersistentClient(path="data/db/chroma")
    coll = client.get_collection("train_prs")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.test, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in tqdm(f, desc="RAG inference"):
            r = json.loads(line)

            q_emb = emb.encode([r["source"]], normalize_embeddings=True, show_progress_bar=False).tolist()
            neigh = coll.query(
                query_embeddings=q_emb, n_results=args.k,
                include=["documents", "metadatas"],
            )

            prompt = build_prompt(r["source"], neigh)

            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            pred = tok.decode(out[0], skip_special_tokens=True)
            g.write(json.dumps({"id": r["id"], "prediction": pred, "target": r["target"]}) + "\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

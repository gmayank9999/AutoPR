"""
FastAPI server. Loads zero-shot T5, fine-tuned CodeT5+ (LoRA), and the SBERT encoder
once at startup. The Streamlit UI talks to this service.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000

NOTE: Run this AFTER training E3. The adapter path below must match your run folder.
"""
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5Tokenizer, T5ForConditionalGeneration,
)
from peft import PeftModel

import chromadb
from sentence_transformers import SentenceTransformer

# === Configuration ===
ADAPTER = "experiments/runs/lora_codet5p-220m/best"
BASE    = "Salesforce/codet5p-220m"

# === Load fine-tuned CodeT5+ + LoRA ===
print("Loading CodeT5+ + LoRA adapter ...")
tok = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForSeq2SeqLM.from_pretrained(BASE).to("cuda")
lora_model = PeftModel.from_pretrained(base, ADAPTER).to("cuda").eval()

# === Load zero-shot T5-base ===
print("Loading zero-shot T5-base ...")
zs_tok = T5Tokenizer.from_pretrained("t5-base")
zs_model = T5ForConditionalGeneration.from_pretrained("t5-base").to("cuda").eval()

# === Load SBERT and Chroma for RAG ===
print("Loading SBERT + Chroma ...")
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
coll = chromadb.PersistentClient(path="data/db/chroma").get_collection("train_prs")

print("Server ready.")

app = FastAPI(title="PR Description Generator")


class GenIn(BaseModel):
    source: str
    max_new_tokens: int = 64
    num_beams: int = 4


def gen_lora(src: str, n_tok: int = 64, nb: int = 4) -> str:
    inp = tok("summarize: " + src, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = lora_model.generate(
            **inp, max_new_tokens=n_tok, num_beams=nb,
            early_stopping=True, no_repeat_ngram_size=3,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def gen_zeroshot(src: str, n_tok: int = 64, nb: int = 4) -> str:
    inp = zs_tok("summarize: " + src, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        out = zs_model.generate(
            **inp, max_new_tokens=n_tok, num_beams=nb,
            early_stopping=True, no_repeat_ngram_size=3,
        )
    return zs_tok.decode(out[0], skip_special_tokens=True)


def gen_rag(src: str, k: int = 3, n_tok: int = 64, nb: int = 4) -> str:
    q = emb.encode([src], normalize_embeddings=True, show_progress_bar=False).tolist()
    n = coll.query(query_embeddings=q, n_results=k, include=["documents", "metadatas"])
    exs = []
    for i, (d, m) in enumerate(zip(n["documents"][0], n["metadatas"][0]), 1):
        exs.append(
            f"EXAMPLE {i}\n"
            f"Commits: {' '.join(d.split()[:80])}\n"
            f"Description: {' '.join(m['target'].split()[:40])}"
        )
    prompt = (
        "Here are similar past pull requests and their descriptions:\n\n"
        + "\n\n".join(exs)
        + f"\n\nNow summarize this pull request:\nCommits: {src}\nDescription:"
    )
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    with torch.no_grad():
        out = lora_model.generate(
            **inp, max_new_tokens=n_tok, num_beams=nb,
            early_stopping=True, no_repeat_ngram_size=3,
        )
    return tok.decode(out[0], skip_special_tokens=True)


@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.post("/generate")
def generate(x: GenIn):
    return {
        "model": "codet5p+lora",
        "prediction": gen_lora(x.source, x.max_new_tokens, x.num_beams),
    }


@app.post("/generate_rag")
def generate_rag(x: GenIn):
    return {
        "model": "codet5p+lora+rag",
        "prediction": gen_rag(x.source, 3, x.max_new_tokens, x.num_beams),
    }


@app.post("/generate_compare")
def generate_compare(x: GenIn):
    return {
        "zero_shot_t5":     gen_zeroshot(x.source, x.max_new_tokens, x.num_beams),
        "codet5p_lora":     gen_lora(x.source, x.max_new_tokens, x.num_beams),
        "codet5p_lora_rag": gen_rag(x.source, 3, x.max_new_tokens, x.num_beams),
    }

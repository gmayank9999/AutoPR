"""
Builds:
  - data/db/pr.sqlite           (all records from all splits, plus an empty predictions table)
  - data/db/chroma/             (embeddings of TRAIN sources only;
                                 each record carries the train target as metadata)

Run once after preprocessing. Re-runnable: drops and recreates the chroma collection.

Usage:
    python src/build_db.py
"""
import json
import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROC = Path("data/processed")
DB_DIR = Path("data/db")
DB_DIR.mkdir(parents=True, exist_ok=True)
SQLITE = DB_DIR / "pr.sqlite"
CHROMA_DIR = DB_DIR / "chroma"


def build_sqlite():
    if SQLITE.exists():
        SQLITE.unlink()  # clean rebuild

    conn = sqlite3.connect(SQLITE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE pr (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            orig_id TEXT NOT NULL,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            num_commits INTEGER,
            src_len INTEGER,
            tgt_len INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE predictions (
            pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            prediction TEXT,
            rouge1 REAL,
            rouge2 REAL,
            rougeL REAL,
            bleu REAL,
            meteor REAL,
            bertscore REAL,
            FOREIGN KEY (row_id) REFERENCES pr(row_id)
        )
    """)

    c.execute("CREATE INDEX idx_pr_split ON pr(split)")
    c.execute("CREATE INDEX idx_pr_origid ON pr(orig_id)")
    c.execute("CREATE INDEX idx_pred_model ON predictions(model_name)")

    for split in ["train", "val", "test"]:
        with open(PROC / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                c.execute(
                    "INSERT INTO pr(split, orig_id, source, target, num_commits, src_len, tgt_len) VALUES (?,?,?,?,?,?,?)",
                    (split, r["id"], r["source"], r["target"],
                     r["num_commits"], r["src_len_words"], r["tgt_len_words"])
                )

    conn.commit()
    conn.close()
    print(f"SQLite populated: {SQLITE}")


def build_chroma():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Clean rebuild
    try:
        client.delete_collection("train_prs")
    except Exception:
        pass

    coll = client.create_collection("train_prs", metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    with open(PROC / "train.jsonl", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]

    BATCH = 128
    for i in tqdm(range(0, len(recs), BATCH), desc="Indexing train PRs"):
        chunk = recs[i:i+BATCH]
        embs = model.encode(
            [r["source"] for r in chunk],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        coll.add(
            ids=[f"train_{i+j}" for j in range(len(chunk))],
            embeddings=embs,
            documents=[r["source"] for r in chunk],
            metadatas=[{"target": r["target"], "orig_id": r["id"]} for r in chunk],
        )

    print(f"Chroma built. Collection size: {coll.count()}")


if __name__ == "__main__":
    build_sqlite()
    build_chroma()

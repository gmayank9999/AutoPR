# AutoPR — Automatic Pull Request Description Generation

GenAI course end-term project. Fine-tunes CodeT5+ with LoRA on the Liu et al. 2019 PR dataset to automatically generate pull request descriptions from commit messages.

**Primary benchmark to beat:** Saini et al. (2025) — BART-base full fine-tuning, ROUGE-1 F1 = 28.03.

## Project overview

Given the commit messages of a GitHub PR, the model generates a short, readable description (1–3 sentences). The core contribution is using LoRA (PEFT) on a code-aware backbone (CodeT5+), which achieves competitive or better ROUGE scores while training only ~1% of the model's parameters. We also add a RAG variant that retrieves similar past PRs from a ChromaDB vector store and prepends them as in-context examples.

## Repository structure

```
AutoPR/
├── config/config.yaml          # all hyperparameters
├── src/
│   ├── download_data.py        # verify/extract dataset CSVs
│   ├── preprocess.py           # CSV -> JSONL normalization
│   ├── build_db.py             # SQLite + ChromaDB setup
│   ├── baselines/
│   │   ├── lexrank_baseline.py # E0 — extractive baseline
│   │   ├── zeroshot_t5.py      # E7 — zero-shot T5
│   │   └── flan_t5_fewshot.py  # E8 — few-shot FLAN-T5
│   ├── train_full_ft.py        # E1 — Saini reproduction (BART full FT)
│   ├── train_lora.py           # E2-E5 — LoRA fine-tuning
│   ├── infer.py                # batch inference (LoRA + full FT)
│   ├── rag_infer.py            # E6 — RAG inference
│   ├── evaluate.py             # ROUGE/BLEU/METEOR/BERTScore
│   ├── error_analysis.py       # hallucination + error taxonomy
│   └── utils.py                # seed, logging helpers
├── api/server.py               # FastAPI inference service
├── ui/streamlit_app.py         # Streamlit demo UI
├── github_action/
│   └── suggest_pr_desc.py      # real GitHub PR → generate description
├── notebooks/00_eda.ipynb      # EDA: length histograms, sample rows
├── report/                     # final report + figures
└── requirements.txt
```

## Quick start

```powershell
# 1. Create environment
conda create -n prgen python=3.10 -y
conda activate prgen

# 2. Install PyTorch with CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Running the pipeline

The CSVs are already in `data/raw/`. Run these in order:

```powershell
# Data prep
python src/download_data.py     # verify CSVs
python src/preprocess.py        # CSV -> JSONL
python src/build_db.py          # SQLite + ChromaDB

# Baselines
python src/baselines/lexrank_baseline.py
python src/baselines/zeroshot_t5.py
python src/baselines/flan_t5_fewshot.py

# E1 — validation gate (Saini reproduction, ~80 min)
python src/train_full_ft.py
python src/infer.py --base experiments/runs/e1_full_ft_bart_base/best `
                    --out experiments/results/e1_full_ft_bart_preds.jsonl

# Check E1 ROUGE-L is ~25. Then continue:

# E2 — BART + LoRA (edit config.yaml: model.name: facebook/bart-base)
python src/train_lora.py --config config/config.yaml

# E3 — CodeT5+ + LoRA (edit config.yaml: model.name: Salesforce/codet5p-220m)
python src/train_lora.py --config config/config.yaml

# ... (see pr_description_generation_implementation_plan.md for full grid)

# Evaluate all experiments
python src/evaluate.py

# Error analysis
python src/error_analysis.py --preds experiments/results/e3_lora_codet5p_preds.jsonl

# Live demo (two terminals)
uvicorn api.server:app --host 0.0.0.0 --port 8000
streamlit run ui/streamlit_app.py

# Real GitHub PR demo
python github_action/suggest_pr_desc.py --url https://github.com/microsoft/vscode/pull/200000
```

## Experiment grid

| Exp | Model | Method | Purpose |
|:--:|---|---|---|
| E0 | LexRank | extractive | Saini baseline (ROUGE-1 = 24.11) |
| **E1** | **BART-base** | **full FT** | **Pipeline validation gate. Target ≈ Saini 28.03** |
| E2 | BART-base | LoRA | RQ1: does PEFT match full FT? |
| **E3** | **CodeT5+-220m** | **LoRA** | **RQ2: code-aware backbone (main contribution)** |
| E4 | FLAN-T5-base | LoRA | backbone ablation |
| E5 | T5-base | LoRA | backbone ablation |
| E6 | CodeT5+-220m | LoRA + RAG | RQ3: does retrieval add lift? |
| E7 | T5-base | zero-shot | sanity floor |
| E8 | FLAN-T5-base | few-shot prompted | RQ4: prompting vs fine-tuning |

## References

- Saini et al. (2025) — Generation of Pull Request Description using Transformers. ResearchSquare. DOI: 10.21203/rs.3.rs-7089220/v1
- Sakib et al. (2024) — Automatic PR Description Generation Using LLMs. IEEE AIBThings.
- Liu et al. (2019) — Automatic Generation of Pull Request Descriptions. ASE 2019.
- Hu et al. (2022) — LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Wang et al. (2023) — CodeT5+. EMNLP 2023.

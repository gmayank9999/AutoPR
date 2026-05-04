"""
Generate result figures from metrics_all.csv -> report/figs/

Usage:
    python report/plot_results.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("report/figs")
OUT.mkdir(parents=True, exist_ok=True)

CSV = Path("experiments/results/metrics_all.csv")
df = pd.read_csv(CSV)

# Clean labels
LABELS = {
    "e0_lexrank_preds.jsonl":       "E0 LexRank",
    "e1_full_ft_bart_preds.jsonl":  "E1 BART Full FT",
    "e2_lora_bart_preds.jsonl":     "E2 BART LoRA",
    "e3_lora_codet5p_preds.jsonl":  "E3 CodeT5+ LoRA",
    "e4_lora_flan_t5_preds.jsonl":  "E4 FLAN-T5 LoRA",
    "e5_lora_t5_preds.jsonl":       "E5 T5-base LoRA",
    "e6_rag_codet5p_preds.jsonl":   "E6 RAG CodeT5+",
    "e7_zeroshot_t5_preds.jsonl":   "E7 Zero-shot T5",
    "e8_flan_t5_fewshot_preds.jsonl":"E8 FLAN-T5 Few-shot",
}

COLORS = {
    "E0 LexRank":            "#adb5bd",   # grey  — baseline
    "E1 BART Full FT":       "#2196F3",   # blue  — full FT
    "E2 BART LoRA":          "#4CAF50",   # green — LoRA
    "E3 CodeT5+ LoRA":       "#FF9800",   # orange— LoRA (code-aware)
    "E4 FLAN-T5 LoRA":       "#9C27B0",   # purple— LoRA
    "E5 T5-base LoRA":       "#009688",   # teal  — LoRA
    "E6 RAG CodeT5+":        "#F44336",   # red   — RAG
    "E7 Zero-shot T5":       "#607D8B",   # slate — zero-shot
    "E8 FLAN-T5 Few-shot":   "#795548",   # brown — few-shot
}

df["label"] = df["file"].map(LABELS)

# Saini et al. 2025 best reported numbers (reference paper)
SAINI_R1  = 28.03
SAINI_R2  = 16.85
SAINI_RL  = 25.49

# ── Figure 1: Grouped bar chart — ROUGE-1 / ROUGE-2 / ROUGE-L ──────────────
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(df))
w = 0.25

bars1 = ax.bar(x - w,   df["rouge1"],  w, label="ROUGE-1", color="#1976D2", alpha=0.85)
bars2 = ax.bar(x,       df["rouge2"],  w, label="ROUGE-2", color="#388E3C", alpha=0.85)
bars3 = ax.bar(x + w,   df["rougeL"],  w, label="ROUGE-L", color="#F57C00", alpha=0.85)

ax.axhline(SAINI_R1, color="#1976D2", linestyle="--", linewidth=1.2, alpha=0.6, label=f"Saini R1 ({SAINI_R1})")
ax.axhline(SAINI_RL, color="#F57C00", linestyle="--", linewidth=1.2, alpha=0.6, label=f"Saini RL ({SAINI_RL})")

ax.set_xticks(x)
ax.set_xticklabels(df["label"], rotation=28, ha="right", fontsize=9)
ax.set_ylabel("Score (×100)")
ax.set_title("ROUGE-1 / ROUGE-2 / ROUGE-L across all experiments\nvs. Saini et al. (2025) baseline (dashed lines)")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, 40)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig_rouge_grouped.png", dpi=150)
plt.close()
print("Saved fig_rouge_grouped.png")

# ── Figure 2: BLEU / METEOR / BERTScore bar chart ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = [("bleu4", "BLEU-4", "#5C6BC0"),
           ("meteor", "METEOR", "#26A69A"),
           ("bertscore_f1", "BERTScore-F1", "#EF5350")]

for ax, (col, title, color) in zip(axes, metrics):
    colors = [COLORS[l] for l in df["label"]]
    bars = ax.bar(df["label"], df[col], color=colors, alpha=0.85, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Score (×100)" if col != "bertscore_f1" else "Score (×100)")
    ax.set_xticklabels(df["label"], rotation=32, ha="right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    # Annotate top bar
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}",
                ha="center", va="bottom", fontsize=6.5)

plt.suptitle("BLEU-4, METEOR, BERTScore-F1 across all experiments", y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig_other_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_other_metrics.png")

# ── Figure 3: ROUGE-L rank comparison vs Saini et al. ───────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
sorted_df = df.sort_values("rougeL", ascending=True)
colors = [COLORS[l] for l in sorted_df["label"]]
bars = ax.barh(sorted_df["label"], sorted_df["rougeL"], color=colors, alpha=0.85, edgecolor="white")
ax.axvline(SAINI_RL, color="red", linestyle="--", linewidth=1.5, label=f"Saini et al. RL = {SAINI_RL}")
ax.set_xlabel("ROUGE-L (×100)")
ax.set_title("ROUGE-L Ranking — Our experiments vs. Saini et al. (2025)")
ax.legend(fontsize=9)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{w:.2f}",
            va="center", fontsize=8)
ax.set_xlim(0, 36)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig_rougeL_rank.png", dpi=150)
plt.close()
print("Saved fig_rougeL_rank.png")

# ── Figure 4: LoRA vs Full FT ROUGE-L scatter ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
PARAM_COUNTS = {   # approx trainable params (M)
    "E1 BART Full FT":  139.0,
    "E2 BART LoRA":       2.4,
    "E3 CodeT5+ LoRA":    2.8,
    "E4 FLAN-T5 LoRA":    2.1,
    "E5 T5-base LoRA":    2.1,
}
sub = df[df["label"].isin(PARAM_COUNTS)].copy()
sub["params"] = sub["label"].map(PARAM_COUNTS)
for _, row in sub.iterrows():
    ax.scatter(row["params"], row["rougeL"], s=120,
               color=COLORS[row["label"]], zorder=3, label=row["label"])
    ax.annotate(row["label"], (row["params"], row["rougeL"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Trainable parameters (M)")
ax.set_ylabel("ROUGE-L (×100)")
ax.set_title("LoRA efficiency: ROUGE-L vs trainable parameter count")
ax.grid(alpha=0.3)
ax.set_xlim(-5, 155)
plt.tight_layout()
plt.savefig(OUT / "fig_lora_efficiency.png", dpi=150)
plt.close()
print("Saved fig_lora_efficiency.png")

# ── Figure 5: Length ratio ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
colors = [COLORS[l] for l in df["label"]]
bars = ax.bar(df["label"], df["len_ratio"], color=colors, alpha=0.85, edgecolor="white")
ax.axhline(1.0, color="black", linestyle="-", linewidth=1.2, label="Perfect length (ratio=1)")
ax.set_ylabel("Length ratio (pred / ref)")
ax.set_title("Prediction length ratio — over-generation (>1) vs under-generation (<1)")
ax.set_xticklabels(df["label"], rotation=28, ha="right", fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "fig_length_ratio.png", dpi=150)
plt.close()
print("Saved fig_length_ratio.png")

print(f"\nAll figures saved to {OUT.resolve()}")

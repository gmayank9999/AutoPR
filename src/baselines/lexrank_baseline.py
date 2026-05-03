"""
LexRank extractive baseline (E0).

Treats each commit message in a PR as a sentence; ranks them by graph centrality;
picks the top 2 as the predicted "description". Reproduces Saini et al. (2025)'s
only baseline (their reported ROUGE-1 F1 = 0.2411).

Usage:
    python src/baselines/lexrank_baseline.py
"""
import json
from pathlib import Path
from tqdm import tqdm

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

TEST = Path("data/processed/test.jsonl")
OUT = Path("experiments/results/e0_lexrank_preds.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

summ = LexRankSummarizer()


def main():
    with open(TEST, encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="LexRank"):
            r = json.loads(line)

            # Re-split commits by our normalized separator
            sents = [s.strip() for s in r["source"].split("[COMMIT_SEP]") if s.strip()]
            if not sents:
                sents = [r["source"]]

            text = ". ".join(sents)
            try:
                parser = PlaintextParser.from_string(text, Tokenizer("english"))
                summary = summ(parser.document, 2)
                pred = " ".join(str(s) for s in summary)
            except Exception:
                pred = ""

            if not pred:
                pred = sents[0][:200]   # fallback: just take the first commit message

            fout.write(json.dumps({"id": r["id"], "prediction": pred, "target": r["target"]}) + "\n")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

"""
Streamlit frontend. Talks to the FastAPI server (api/server.py).

Run (in a separate terminal from the FastAPI server):
    streamlit run ui/streamlit_app.py

The FastAPI server must already be running at http://localhost:8000.
"""
import sqlite3
import requests
import streamlit as st
import pandas as pd

API = "http://localhost:8000"

st.set_page_config(page_title="PR Description Generator", layout="wide")
st.title("Automatic Pull Request Description Generator")
st.caption("Fine-tuned CodeT5+ with LoRA + optional RAG — GenAI course project")

tab1, tab2, tab3 = st.tabs(["Try it", "Test-set browser", "Model card"])

# === Tab 1: live demo ===
with tab1:
    st.subheader("Paste commit messages")
    st.caption("Use ' [COMMIT_SEP] ' between commits. Lowercased input is fine — that's how the model was trained.")

    default = (
        "fix git ignore multiplicated settings . [COMMIT_SEP] "
        "change path to formatter config file . [COMMIT_SEP] "
        "format plugin attached to compile and defined in each module that needs format ."
    )
    src = st.text_area("Commits", value=default, height=160)

    col1, col2 = st.columns(2)
    with col1:
        nb = st.slider("Beam search width", 1, 8, 4)
    with col2:
        mt = st.slider("Max new tokens", 16, 128, 64)

    if st.button("Generate", type="primary"):
        with st.spinner("Calling API..."):
            try:
                resp = requests.post(
                    f"{API}/generate_compare",
                    json={"source": src, "num_beams": nb, "max_new_tokens": mt},
                    timeout=120,
                )
                resp.raise_for_status()
                r = resp.json()
            except Exception as e:
                st.error(f"API error: {e}. Is the FastAPI server running?")
                st.stop()

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Zero-shot T5-base**")
        c1.info(r["zero_shot_t5"])
        c2.markdown("**CodeT5+ + LoRA (ours)**")
        c2.success(r["codet5p_lora"])
        c3.markdown("**CodeT5+ + LoRA + RAG**")
        c3.success(r["codet5p_lora_rag"])

# === Tab 2: browse the test set ===
with tab2:
    st.subheader("Test-set browser")
    try:
        conn = sqlite3.connect("data/db/pr.sqlite")
        df = pd.read_sql_query(
            "SELECT row_id, orig_id, source, target FROM pr WHERE split='test' LIMIT 50",
            conn,
        )
        conn.close()
    except Exception as e:
        st.warning(f"SQLite not available ({e}). Run src/build_db.py first.")
        df = pd.DataFrame()

    if not df.empty:
        idx = st.selectbox(
            "Pick a test PR",
            df.index,
            format_func=lambda i: f"{df.loc[i, 'orig_id']}",
        )
        row = df.loc[idx]

        st.markdown("**Source (commits + comments):**")
        st.write(row["source"])

        st.markdown("**Reference description (ground truth):**")
        st.write(row["target"])

        if st.button("Generate predictions for this PR"):
            with st.spinner("Calling API..."):
                try:
                    resp = requests.post(
                        f"{API}/generate_compare",
                        json={"source": row["source"]},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    r = resp.json()
                    for k, v in r.items():
                        st.markdown(f"**{k}:**")
                        st.write(v)
                except Exception as e:
                    st.error(f"API error: {e}")

# === Tab 3: model card ===
with tab3:
    st.markdown("""
### Model card

- **Base model:** `Salesforce/codet5p-220m`
- **Adapter:** LoRA (r=16, α=32, dropout=0.05) on `{q, k, v, o}` projection layers
- **Trained on:** Liu et al. 2019 PR dataset (Java GitHub repos; 33,466 train / 4,183 val / 4,183 test)
- **Input format:** `summarize: <commits joined by [COMMIT_SEP]>`
- **Output:** 1–3 sentence PR description (max 64 tokens)
- **Trainable parameters:** ~1% of the base model

### Known limitations

- Java-only training data; performance on JavaScript / Python / Rust PRs may degrade.
- Lowercases everything (the dataset was preprocessed that way).
- May hallucinate identifier names when the commit messages are very terse.
- Tends to over-weight the first commit message in a multi-commit PR.

### Local benchmark to beat

| Model | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|---|--:|--:|--:|
| Saini et al. 2025 (BART-base, full FT) | 28.03 | 16.85 | 25.49 |
| **This model (CodeT5+ + LoRA + RAG)** | **(see experiments/results/metrics_all.csv)** | | |
""")

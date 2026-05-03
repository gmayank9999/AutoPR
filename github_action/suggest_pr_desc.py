"""
Fetch a real GitHub PR's commit messages and generate a suggested description.

Usage:
    python github_action/suggest_pr_desc.py --url https://github.com/<owner>/<repo>/pull/<number>

Requires GITHUB_TOKEN in .env (any fine-grained PAT with public-repo read access).
The FastAPI server (api/server.py) must be running locally at http://localhost:8000.
"""
import argparse
import os
import re
import sys

import requests
from dotenv import load_dotenv
from github import Github

load_dotenv()
API = "http://localhost:8000"


def parse_url(url: str):
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if not m:
        sys.exit("Expected URL format: https://github.com/<owner>/<repo>/pull/<number>")
    return m.group(1), m.group(2), int(m.group(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Full GitHub PR URL")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("Set GITHUB_TOKEN in .env first.")

    owner, repo, num = parse_url(args.url)
    gh = Github(token)
    pr = gh.get_repo(f"{owner}/{repo}").get_pull(num)

    # Take first line of each commit message (conventional summary)
    commit_msgs = []
    for c in pr.get_commits():
        first_line = c.commit.message.splitlines()[0].strip().lower()
        if first_line:
            commit_msgs.append(first_line)

    if not commit_msgs:
        sys.exit("PR has no commits — nothing to summarize.")

    src = " [COMMIT_SEP] ".join(commit_msgs)

    print(f"\n=== {owner}/{repo} PR #{num}: {pr.title} ===\n")
    print(f"Number of commits: {len(commit_msgs)}\n")
    print(f"Commits (joined input):\n  {src}\n")
    print(f"Human-written description:")
    print(f"  {pr.body or '(empty)'}\n")

    # Call our local API
    try:
        r = requests.post(
            f"{API}/generate_compare",
            json={"source": src},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        sys.exit(f"API call failed (is the FastAPI server running?): {e}")

    print(f"--- Model predictions ---")
    print(f"Zero-shot T5-base:\n  {data['zero_shot_t5']}\n")
    print(f"CodeT5+ + LoRA (ours):\n  {data['codet5p_lora']}\n")
    print(f"CodeT5+ + LoRA + RAG:\n  {data['codet5p_lora_rag']}\n")


if __name__ == "__main__":
    main()

"""Generate synthetic queries for the v1 corpus_200 via a local Ollama model.

For each document: one natural question it answers, and one paraphrase — both instructed
NOT to reuse the document's distinctive key phrases verbatim (so retrieval is semantic,
not string matching). Output: data/c200_queries.jsonl. Synthetic: labelled in every run.

Usage: python scripts/v2/gen_c200_queries.py [--model qwen2.5:7b] [--host http://localhost:11434]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engram_bench.datasets import DATA, load_c200_docs  # noqa: E402

PROMPTS = {
    "question": ("Write ONE short natural question (max 15 words) that the following sentence answers. "
                 "Do not copy its distinctive phrases verbatim; use different wording. "
                 "Output only the question.\n\nSentence: {t}\nQuestion:"),
    "paraphrase": ("Paraphrase the following sentence in different words (max 25 words), keeping the meaning. "
                   "Avoid reusing its distinctive phrases. Output only the paraphrase.\n\nSentence: {t}\nParaphrase:"),
}


def ollama(host: str, model: str, prompt: str) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "seed": 0, "num_predict": 64}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip().strip('"').split("\n")[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    a = ap.parse_args()
    ids, docs, _ = load_c200_docs()
    out = DATA / "c200_queries.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, (cid, t) in enumerate(zip(ids, docs)):
            row = {"cache_id": cid, "generator": f"ollama:{a.model}"}
            for k, p in PROMPTS.items():
                row[k] = ollama(a.host, a.model, p.format(t=t))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(i, row["question"], "|", row["paraphrase"], flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()

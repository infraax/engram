"""Datasets for engram-bench.

Each loader returns a RetrievalSet: docs (id, text), queries (id, text), qrels {qid: set(doc_idx)}.

  c200        v1 corpus (200 single-sentence docs, 10 domains). Queries are LLM-generated
              questions (data/c200_queries.jsonl, made by scripts/v2/gen_c200_queries.py) —
              synthetic, labelled as such in every run.
  c200-para   same, paraphrase queries.
  scifact     BEIR SciFact (public): 5183 abstracts, 300 test queries, human qrels.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@dataclass
class RetrievalSet:
    name: str
    doc_ids: list[str]
    docs: list[str]
    query_ids: list[str]
    queries: list[str]
    qrels: dict[int, set[int]]                 # query index -> set of doc indices
    meta: dict = field(default_factory=dict)


def load_c200_docs(path: Path | None = None) -> tuple[list[str], list[str], list[str]]:
    path = path or DATA / "corpus_200.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r["cache_id"] for r in rows], [r["text"].strip() for r in rows], [r["domain"] for r in rows]


def load_c200(kind: str = "question", path: Path | None = None) -> RetrievalSet:
    ids, docs, domains = load_c200_docs()
    qpath = path or DATA / "c200_queries.jsonl"
    if not qpath.exists():
        raise FileNotFoundError(f"{qpath} missing — run scripts/gen_c200_queries.py first")
    qrows = [json.loads(l) for l in open(qpath, encoding="utf-8") if l.strip()]
    idx = {d: i for i, d in enumerate(ids)}
    qids, qs, qrels = [], [], {}
    for r in qrows:
        text = r.get(kind)
        if not text:
            continue
        qrels[len(qs)] = {idx[r["cache_id"]]}
        qids.append(f"{r['cache_id']}:{kind}")
        qs.append(text.strip())
    return RetrievalSet(f"c200-{kind}", ids, docs, qids, qs, qrels,
                        meta={"synthetic_queries": True, "generator": qrows[0].get("generator", "?"),
                              "domains": domains})


def load_scifact(max_docs: int | None = None, seed: int = 0) -> RetrievalSet:
    """BEIR SciFact via the mteb parquet mirror (corpus / queries / qrels-test)."""
    from datasets import load_dataset

    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", "default", split="test")
    rel: dict[str, set[str]] = {}
    for r in qrels_ds:
        if int(r.get("score", 1)) > 0:
            rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    doc_ids = [str(r["_id"]) for r in corpus]
    docs = [((r.get("title") or "") + ". " + (r.get("text") or "")).strip() for r in corpus]
    if max_docs and max_docs < len(doc_ids):
        needed = set().union(*rel.values())
        keep = [i for i, d in enumerate(doc_ids) if d in needed]
        rest = [i for i, d in enumerate(doc_ids) if d not in needed]
        random.Random(seed).shuffle(rest)
        keep = sorted(keep + rest[: max(0, max_docs - len(keep))])
        doc_ids, docs = [doc_ids[i] for i in keep], [docs[i] for i in keep]
    didx = {d: i for i, d in enumerate(doc_ids)}
    qtext = {str(r["_id"]): r["text"] for r in queries}
    qids, qs, qrels = [], [], {}
    for qid, ds in rel.items():
        golds = {didx[d] for d in ds if d in didx}
        if qid in qtext and golds:
            qrels[len(qs)] = golds
            qids.append(qid)
            qs.append(qtext[qid].strip())
    return RetrievalSet("scifact", doc_ids, docs, qids, qs, qrels,
                        meta={"synthetic_queries": False, "n_docs": len(docs)})


# ── perturbation suite (identity task, H1) ───────────────────────────────────
def _typo(text: str, rng: random.Random) -> str:
    words = text.split()
    cands = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not cands:
        return text
    i = rng.choice(cands)
    w = list(words[i]); j = rng.randrange(1, len(w) - 2)
    w[j], w[j + 1] = w[j + 1], w[j]
    words[i] = "".join(w)
    return " ".join(words)


PERTURBATIONS = {
    "lead_newline": lambda t, r: "\n" + t,
    "lead_space":   lambda t, r: " " + t,
    "trail_newline": lambda t, r: t + "\n",
    "prefix_note":  lambda t, r: "Note: " + t,
    "prefix_long":  lambda t, r: "The following sentence is from an encyclopedia entry. " + t,
    "typo":         _typo,
    "lowercase":    lambda t, r: t.lower(),
}


def perturb_all(texts: list[str], seed: int = 0) -> dict[str, list[str]]:
    rng = random.Random(seed)
    return {name: [fn(t, rng) for t in texts] for name, fn in PERTURBATIONS.items()}

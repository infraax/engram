"""Retrieval + stability metrics. Vectors are assumed L2-normalized (cosine = dot)."""

from __future__ import annotations

import numpy as np


def retrieval_metrics(D: np.ndarray, Q: np.ndarray, qrels: dict[int, set[int]], ks=(1, 5, 10)) -> dict:
    D = np.asarray(D, np.float32)
    Q = np.asarray(Q, np.float32)
    S = Q @ D.T                                           # [nq, nd]
    order = np.argsort(-S, axis=1)
    nq = len(qrels)
    rec = {k: 0.0 for k in ks}
    mrr = ndcg = 0.0
    margins = []
    for qi, golds in qrels.items():
        ranked = order[qi]
        hits = np.isin(ranked[:max(ks + (10,))], list(golds))
        for k in ks:
            rec[k] += float(hits[:k].any())
        first = np.where(hits[:10])[0]
        mrr += 1.0 / (first[0] + 1) if len(first) else 0.0
        gains = hits[:10] / np.log2(np.arange(2, 12))
        ideal = (1 / np.log2(np.arange(2, 2 + min(len(golds), 10)))).sum()
        ndcg += gains.sum() / ideal
        s = S[qi]
        gold_best = max(s[list(golds)])
        mask = np.ones(len(s), bool); mask[list(golds)] = False
        margins.append(float(gold_best - s[mask].max()))
    m = np.array(margins)
    out = {f"R@{k}": rec[k] / nq for k in ks}
    out.update({"MRR@10": mrr / nq, "nDCG@10": ndcg / nq,
                "margin_mean": float(m.mean()), "margin_median": float(np.median(m)),
                "margin_p10": float(np.percentile(m, 10)), "n_queries": nq, "n_docs": len(D)})
    return out


def offset_stats(D: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> dict:
    """Mean |cos| and mean cos between random distinct docs (E-02 diagnostic)."""
    rng = np.random.default_rng(seed)
    n = len(D)
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    keep = i != j
    c = (D[i[keep]] * D[j[keep]]).sum(1)
    return {"pair_cos_mean": float(c.mean()), "pair_abs_cos_mean": float(np.abs(c).mean()),
            "pair_cos_p99": float(np.percentile(c, 99))}


def stability_metrics(D_clean: np.ndarray, D_pert: np.ndarray) -> dict:
    """Identity task: perturbed doc i should map to clean doc i."""
    self_cos = (D_clean * D_pert).sum(1)
    S = D_pert @ D_clean.T
    r1 = float((S.argmax(1) == np.arange(len(D_clean))).mean())
    np.fill_diagonal(S, -np.inf)
    margin = self_cos - S.max(1)
    return {"self_cos_mean": float(self_cos.mean()), "self_cos_min": float(self_cos.min()),
            "id_R@1": r1, "id_margin_mean": float(margin.mean())}

"""M0 runner — first P0 measurements (H1 position fragility, H2 sinks/centering, H5 null hypothesis).

Tasks:
  --task baselines   text-embedding models (MEANING sense) on the datasets + perturbation suite
  --task kv          KV-key fingerprint pipelines (REFLEX sense) for one HF model, all catalog pipelines
                     computed from a single capture per text

Calibration (for centering / PC removal) is always OUT-OF-SAMPLE:
  c200-*   → 256 SciFact documents        scifact → c200 docs + c200 questions
Every run writes results/v2/<date>/<run_id>.json via engram_bench.registry.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engram_bench.catalog import PIPELINES  # noqa: E402
from engram_bench.datasets import load_c200, load_scifact, perturb_all  # noqa: E402
from engram_bench.metrics import offset_stats, retrieval_metrics, stability_metrics  # noqa: E402
from engram_bench.registry import write_run  # noqa: E402


def load_sets(names: list[str], scifact_max_docs: int | None):
    sets = {}
    for n in names:
        if n == "c200":
            sets["c200-question"] = load_c200("question")
            sets["c200-paraphrase"] = load_c200("paraphrase")
        elif n == "scifact":
            sets["scifact"] = load_scifact(max_docs=scifact_max_docs)
    return sets


def calibration_texts(set_name: str, sets_cache: dict, scifact_max_docs) -> list[str]:
    if set_name.startswith("c200"):
        sf = sets_cache.get("scifact") or load_scifact(max_docs=scifact_max_docs)
        return sf.docs[:256]
    c = load_c200("question")
    return c.docs + c.queries


# ── baselines ────────────────────────────────────────────────────────────────
def run_baselines(sets: dict, embedders: list[str], device: str | None) -> None:
    from engram_bench.baselines import Embedder

    for key in embedders:
        emb = Embedder(key, device=device)
        for sname, rs in sets.items():
            t0 = time.time()
            D = emb.docs(rs.docs); Q = emb.queries(rs.queries)
            m = {"retrieval": retrieval_metrics(D, Q, rs.qrels), "offset": offset_stats(D),
                 "encode_s": round(time.time() - t0, 2), "dims": int(D.shape[1])}
            if sname == "c200-question":
                m["stability"] = {p: stability_metrics(D, emb.docs(txts))
                                  for p, txts in perturb_all(rs.docs).items()}
            cfg = {"task": "baseline", "sense": "meaning_text", "embedder": emb.spec["id"], "dataset": sname,
                   "synthetic_queries": rs.meta.get("synthetic_queries")}
            p = write_run(f"base-{key}-{sname}", cfg, m)
            r = m["retrieval"]
            print(f"[{key:9s}] {sname:16s} R@1={r['R@1']:.3f} R@10={r['R@10']:.3f} nDCG@10={r['nDCG@10']:.3f} "
                  f"margin={r['margin_mean']:+.3f} |cos|={m['offset']['pair_abs_cos_mean']:.3f}  → {p.name}",
                  flush=True)


# ── KV pipelines ─────────────────────────────────────────────────────────────
def run_kv(sets: dict, model_id: str, device: str | None, dtype: str, pipelines: list[str],
           max_tokens: int, scifact_max_docs) -> None:
    import torch

    from engram_core.transforms import HFCapturer, Pipeline

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "auto": "auto"}[dtype]
    cap = HFCapturer(model_id, device=device, dtype=torch_dtype, want_pre=True, want_hidden=True,
                     max_tokens=max_tokens)
    pipes = {n: Pipeline(n, PIPELINES[n]) for n in pipelines}

    def raws(texts: list[str], label: str) -> dict[str, np.ndarray]:
        out = {n: [] for n in pipes}
        rope_err = []
        t0 = time.time()
        for i, t in enumerate(texts):
            c = cap.capture(t)
            if c.keys_pre is not None and c.cos is not None and i < 32:      # S1 correctness check
                from engram_core.transforms.rope import inverse_rope
                l = c.n_layers // 2
                dr = inverse_rope(c.keys_post[l], c.cos, c.sin).reshape(-1)
                pr = c.keys_pre[l].reshape(-1)
                rope_err.append(float(torch.nn.functional.cosine_similarity(dr, pr, dim=0)))
            for n, p in pipes.items():
                out[n].append(p.raw(c))
        print(f"  captured {len(texts):5d} {label:14s} in {time.time()-t0:6.1f}s", flush=True)
        res = {n: np.stack(v) for n, v in out.items()}
        res["_rope_check_cos"] = np.array(rope_err) if rope_err else np.array([np.nan])
        return res

    for sname, rs in sets.items():
        print(f"== {model_id} :: {sname}", flush=True)
        cal = raws(calibration_texts(sname, sets, scifact_max_docs), "calibration")
        Rd = raws(rs.docs, "docs")
        Rq = raws(rs.queries, "queries")
        Rp = {}
        if sname == "c200-question":
            for pname, txts in perturb_all(rs.docs).items():
                Rp[pname] = raws(txts, pname)
        results = {}
        for n, p in pipes.items():
            p.fit(cal[n]) if p.needs_fit else None
            D = np.stack([p.transform(x) for x in Rd[n]])
            Q = np.stack([p.transform(x) for x in Rq[n]])
            m = {"retrieval": retrieval_metrics(D, Q, rs.qrels), "offset": offset_stats(D),
                 "dims": int(D.shape[1]), "space_key": p.space(model_id, D.shape[1]).key,
                 "pcs_removed": getattr(p.frozen, "n_removed", 0) if p.frozen else 0}
            if Rp:
                m["stability"] = {pn: stability_metrics(D, np.stack([p.transform(x) for x in R[n]]))
                                  for pn, R in Rp.items()}
            results[n] = m
            r = m["retrieval"]
            stab = ""
            if Rp:
                stab = " | " + " ".join(f"{k[:6]}={v['self_cos_mean']:.3f}/{v['id_R@1']:.2f}"
                                        for k, v in m["stability"].items())
            print(f"  {n:30s} R@1={r['R@1']:.3f} R@10={r['R@10']:.3f} nDCG={r['nDCG@10']:.3f} "
                  f"margin={r['margin_mean']:+.4f} |cos|={m['offset']['pair_abs_cos_mean']:.3f}{stab}", flush=True)
        cfg = {"task": "kv", "sense": "reflex_kv", "model": model_id, "dtype": dtype, "dataset": sname,
               "max_tokens": max_tokens, "pipelines": {n: PIPELINES[n] for n in pipes},
               "synthetic_queries": rs.meta.get("synthetic_queries"),
               "calibration": "scifact docs[:256]" if sname.startswith("c200") else "c200 docs + questions"}
        extra = {"rope_check_cos_mean": float(np.nanmean(Rd["_rope_check_cos"])),
                 "rope_check_cos_min": float(np.nanmin(Rd["_rope_check_cos"]))}
        path = write_run(f"kv-{model_id.split('/')[-1]}-{sname}", cfg, results, extra)
        print(f"  rope check (derope(post) vs pre): mean cos={extra['rope_check_cos_mean']:.6f} "
              f"min={extra['rope_check_cos_min']:.6f}  → {path.name}", flush=True)
    cap.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["baselines", "kv"], required=True)
    ap.add_argument("--datasets", default="c200,scifact")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--device", default=None)
    ap.add_argument("--embedders", default="minilm,bge-small")
    ap.add_argument("--pipelines", default=",".join(PIPELINES))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--scifact-max-docs", type=int, default=None)
    a = ap.parse_args()
    sets = load_sets(a.datasets.split(","), a.scifact_max_docs)
    if a.task == "baselines":
        run_baselines(sets, a.embedders.split(","), a.device)
    else:
        run_kv(sets, a.model, a.device, a.dtype, a.pipelines.split(","), a.max_tokens, a.scifact_max_docs)


if __name__ == "__main__":
    main()

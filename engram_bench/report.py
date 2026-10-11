"""Build comparison tables from results/v2 run files.

Usage: python -m engram_bench.report [--date YYYY-MM-DD] [--md out.md]
Latest run per (model|embedder, dataset) wins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import RESULTS

DATASETS = ["c200-question", "c200-paraphrase", "scifact"]


def load(date: str | None):
    files = sorted((RESULTS / date).glob("*.json") if date else RESULTS.glob("*/*.json"))
    latest = {}
    for f in files:
        r = json.loads(f.read_text())
        c = r["config"]
        who = c.get("embedder") or c.get("model")
        latest[(c["task"], who, c["dataset"])] = r
    return latest


def rows(latest):
    out = []
    for (task, who, ds), r in sorted(latest.items()):
        if task == "baseline":
            m = r["metrics"]
            out.append({"system": who.split("/")[-1], "pipeline": "(embedder)", "dataset": ds,
                        **_pick(m), "run_id": r["run_id"]})
        else:
            for pname, m in r["metrics"].items():
                out.append({"system": who.split("/")[-1], "pipeline": pname, "dataset": ds,
                            **_pick(m), "run_id": r["run_id"]})
    return out


def _pick(m):
    ret = m["retrieval"]
    d = {"R@1": ret["R@1"], "R@10": ret["R@10"], "nDCG@10": ret["nDCG@10"],
         "margin": ret["margin_mean"], "abs_cos": m["offset"]["pair_abs_cos_mean"], "dims": m.get("dims")}
    st = m.get("stability")
    if st:
        d["stab_min_selfcos"] = min(v["self_cos_mean"] for v in st.values())
        d["stab_min_idR1"] = min(v["id_R@1"] for v in st.values())
    return d


def markdown(rs, dataset: str, metric: str = "nDCG@10") -> str:
    sub = sorted([r for r in rs if r["dataset"] == dataset], key=lambda r: -r[metric])
    hdr = "| system | pipeline | R@1 | R@10 | nDCG@10 | margin | mean\\|cos\\| | stab self-cos (min) | id R@1 (min) |\n|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['system']} | {r['pipeline']} | {r['R@1']:.3f} | {r['R@10']:.3f} | {r['nDCG@10']:.3f} | "
        f"{r['margin']:+.4f} | {r['abs_cos']:.3f} | {r.get('stab_min_selfcos', float('nan')):.3f} | "
        f"{r.get('stab_min_idR1', float('nan')):.2f} |\n" for r in sub)
    return f"### {dataset}\n\n" + hdr + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--md")
    a = ap.parse_args()
    rs = rows(load(a.date))
    text = "\n".join(markdown(rs, d) for d in DATASETS if any(r["dataset"] == d for r in rs))
    print(text)
    if a.md:
        Path(a.md).write_text(text)


if __name__ == "__main__":
    main()

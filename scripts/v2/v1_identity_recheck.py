"""E-09 recheck: what does v1's "Recall@1 at N=200" measure?

Recomputes v1 Fourier fingerprints (|rFFT| over the layer axis, per-frequency L2) from the saved
v1 artefacts and reports Recall@1 / margin when the query is the SAME document text encoded by
the other Llama model (3B <-> 8B), plus the trivial same-model case.

Inputs (v1 artefacts, gitignored — present only in a local v1 checkout):
  results/absolute/layer_keys_3b.pt   {"keys": [200, 28, 1024]}   Llama 3.2 3B, mean keys per layer
  results/absolute/layer_keys_8b.pt   {"keys": [200, 32, 1024]}   Llama 3.1 8B

Usage: python scripts/v2/v1_identity_recheck.py [--dir results/absolute]
See docs/v2/FINDINGS.md §2.1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def fingerprint(keys: torch.Tensor, freqs: list[int]) -> torch.Tensor:
    amp = torch.fft.rfft(keys.float(), dim=1).abs()            # [N, L//2+1, D]
    fp = torch.cat([F.normalize(amp[:, f], dim=-1) for f in freqs], dim=-1)
    return F.normalize(fp, dim=-1)


def evaluate(q: torch.Tensor, d: torch.Tensor) -> tuple[float, float, float]:
    s = q @ d.T
    n = s.shape[0]
    r1 = (s.argmax(1) == torch.arange(n)).float().mean().item()
    imp = s.clone()
    imp.fill_diagonal_(-9.0)
    best_impostor = imp.max(1).values
    return r1, (s.diag() - best_impostor).mean().item(), best_impostor.mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/absolute")
    a = ap.parse_args()
    root = Path(a.dir)
    k3 = torch.load(root / "layer_keys_3b.pt", weights_only=False)["keys"]
    k8 = torch.load(root / "layer_keys_8b.pt", weights_only=False)["keys"]
    print(f"3B keys {tuple(k3.shape)}  8B keys {tuple(k8.shape)}")
    print(f"{'freqs':8s} {'query->index':14s} {'R@1':>6s} {'margin':>9s} {'impostor cos':>13s}")
    for freqs in ([1], [0, 1]):
        f3, f8 = fingerprint(k3, freqs), fingerprint(k8, freqs)
        for name, (q, d) in {"3B -> 8B": (f3, f8), "8B -> 3B": (f8, f3), "3B -> 3B": (f3, f3)}.items():
            r1, m, imp = evaluate(q, d)
            label = "+".join(f"f{f}" for f in freqs)
            print(f"{label:8s} {name:14s} {r1:6.3f} {m:+9.5f} {imp:13.4f}")


if __name__ == "__main__":
    main()

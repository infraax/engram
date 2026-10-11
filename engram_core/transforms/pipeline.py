"""Fingerprint pipeline: S1 derope … S8 normalize, driven by a config dict.

Config keys (all optional except source/axis):
  source      "post" | "pre" | "derope" | "hidden"     (S0/S1)
  layers      "all" | [a, b] fractions (0..1) | [i, j] ints   (layer band)
  drop_first  int   — drop the first k tokens before pooling (S2, attention-sink control)
  pool        "mean"                                    (S3)
  axis        {"type": "rfft_amp", "freqs": [0,1], "normalize_layers": bool}
              {"type": "concat", "normalize_layers": bool}
              {"type": "layer_mean"}
              {"type": "svd_perdoc", "rank": 128, "gate": 6, "max_rows": 8192}   (v1 MAR)
  center      bool  — subtract a frozen mean fitted on a calibration set (S5)
  remove_pcs  int | "bbp" — remove top principal directions fitted on calibration (S6, "all-but-the-top")
  norm        "l2" | "none"                             (S8)

`raw(cap)` → vector before fitted stages; `fit(raw_vectors)` learns FrozenMean;
`transform(raw)` → final fingerprint. `space(model_id)` → SpaceDescriptor.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..space import SpaceDescriptor, digest_hex
from .capture import KVCapture
from .rope import inverse_rope


# ── helpers ──────────────────────────────────────────────────────────────────
def _layer_indices(spec, n_layers: int) -> list[int]:
    if spec in (None, "all"):
        return list(range(n_layers))
    a, b = spec
    if isinstance(a, float) or isinstance(b, float):
        a, b = int(round(a * n_layers)), int(round(b * n_layers))
    a, b = max(0, a), min(n_layers, b)
    return list(range(a, b)) or list(range(n_layers))


def _keys(cap: KVCapture, source: str) -> list[torch.Tensor]:
    if source == "post":
        return cap.keys_post
    if source == "pre":
        if cap.keys_pre is None:
            raise ValueError("capture has no pre-RoPE keys (want_pre=False)")
        return cap.keys_pre
    if source == "derope":
        if cap.cos is None:
            raise ValueError("capture has no rope cos/sin")
        return [inverse_rope(k, cap.cos, cap.sin) for k in cap.keys_post]
    raise ValueError(f"unknown key source {source!r}")


def _svd_perdoc(keys: torch.Tensor, rank: int, gate: int, max_rows: int) -> torch.Tensor:
    """Faithful port of v1 MARStateExtractor._svd_project. keys: [L, H, T, D]."""
    L, H, T, D = keys.shape
    target = max(1, min(rank, D - min(gate, D - 1)))   # fixed output length per space
    v = _svd_perdoc_raw(keys, rank, gate, max_rows)
    out = torch.zeros(target)
    out[: min(target, v.shape[0])] = v[:target]
    return out


def _svd_perdoc_raw(keys: torch.Tensor, rank: int, gate: int, max_rows: int) -> torch.Tensor:
    L, H, T, D = keys.shape
    rows = keys.reshape(L * H * T, D).float()
    if rows.shape[0] > max_rows:
        g = torch.Generator().manual_seed(42)
        rows = rows[torch.randperm(rows.shape[0], generator=g)[:max_rows]]
    max_rank = min(D, rows.shape[0])
    gate = min(gate, max_rank - 1)          # tiny models: never return an empty vector
    eff = min(gate + rank, max_rank)
    _, _, Vh = torch.linalg.svd(rows, full_matrices=False)
    W = Vh[gate:eff, :].T
    return (rows @ W).mean(0)


def _fourier(mat: torch.Tensor, freqs, normalize_layers: bool) -> torch.Tensor:
    """v1 compute_fourier_fingerprint(_v2): rFFT over the layer axis. mat: [L, dim]."""
    if normalize_layers:
        mat = F.normalize(mat, dim=-1)
    amp = torch.fft.rfft(mat, dim=0).abs()
    return torch.cat([F.normalize(amp[f], dim=-1) for f in freqs], dim=-1)


# ── frozen statistics (S5/S6) ────────────────────────────────────────────────
class FrozenMean:
    """Mean (+ optional top principal directions) fitted once on a calibration set."""

    def __init__(self, remove_pcs=0):
        self.remove_pcs = remove_pcs
        self.mean: np.ndarray | None = None
        self.pcs: np.ndarray | None = None
        self.n_removed = 0
        self.eigs: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FrozenMean":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(0)
        Xc = X - self.mean
        if self.remove_pcs:
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            eig = s ** 2 / max(1, X.shape[0] - 1)
            self.eigs = eig
            if self.remove_pcs == "bbp":
                # crude Marchenko–Pastur edge: noise var ≈ median eigenvalue (documented as crude, H3)
                n, p = X.shape
                gamma = min(p, n) / max(p, n)
                edge = float(np.median(eig)) * (1 + np.sqrt(gamma)) ** 2
                k = int((eig > edge).sum())
            else:
                k = int(self.remove_pcs)
            self.n_removed = k
            self.pcs = Vt[:k] if k > 0 else None
        return self

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64) - self.mean
        if self.pcs is not None:
            x = x - (x @ self.pcs.T) @ self.pcs
        return x

    @property
    def hash(self) -> str:
        if self.mean is None:
            return ""
        payload = {"mean": np.round(self.mean, 6).tolist(),
                   "pcs": None if self.pcs is None else np.round(self.pcs, 6).tolist()}
        return digest_hex(payload, "engram.center.v2", n_bytes=8)


# ── pipeline ─────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = copy.deepcopy(config)
        self.frozen: FrozenMean | None = None
        if config.get("center") or config.get("remove_pcs"):
            self.frozen = FrozenMean(remove_pcs=config.get("remove_pcs", 0))

    @property
    def needs_fit(self) -> bool:
        return self.frozen is not None

    def raw(self, cap: KVCapture) -> np.ndarray:
        c = self.config
        axis = c["axis"]
        if c["source"] == "hidden":
            if cap.hidden_mean is None:
                raise ValueError("capture has no hidden states (want_hidden=False)")
            idx = _layer_indices(c.get("layers"), len(cap.hidden_mean))
            mat = torch.stack([cap.hidden_mean[i] for i in idx])          # [L, d_model]
        else:
            keys = _keys(cap, c["source"])
            idx = _layer_indices(c.get("layers"), len(keys))
            k = c.get("drop_first", 0)
            sel = [keys[i][:, k:, :] if keys[i].shape[1] > k else keys[i] for i in idx]
            if axis["type"] == "svd_perdoc":
                v = _svd_perdoc(torch.stack(sel), axis.get("rank", 128), axis.get("gate", 6),
                                axis.get("max_rows", 8192))
                return v.numpy().astype(np.float64)
            mat = torch.stack([s.mean(dim=1).reshape(-1) for s in sel])  # [L, H_kv*D]
        t = axis["type"]
        if t == "rfft_amp":
            v = _fourier(mat, axis.get("freqs", [0, 1]), axis.get("normalize_layers", False))
        elif t == "concat":
            m = F.normalize(mat, dim=-1) if axis.get("normalize_layers", True) else mat
            v = m.reshape(-1)
        elif t == "layer_mean":
            v = mat.mean(0)
        else:
            raise ValueError(f"unknown axis type {t!r}")
        return v.numpy().astype(np.float64)

    def fit(self, raw_vectors) -> "Pipeline":
        if self.frozen is not None:
            self.frozen.fit(np.stack(list(raw_vectors)))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        x = self.frozen.apply(raw) if self.frozen is not None else np.asarray(raw, np.float64)
        if self.config.get("norm", "l2") == "l2":
            n = np.linalg.norm(x)
            x = x / n if n > 0 else x
        return x.astype(np.float32)

    def space(self, model_id: str, dims: int) -> SpaceDescriptor:
        sense = "reflex_kv" if self.config["source"] != "hidden" else "reflex_kv"
        return SpaceDescriptor(sense=sense, model_id=model_id, pipeline=self.config, dims=dims,
                               center_hash=self.frozen.hash if self.frozen else "",
                               extra={"name": self.name})

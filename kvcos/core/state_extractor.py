"""
Engrammatic Geometry Retrieval — State Extraction Layer


Extracts a retrieval state vector from a KV cache tensor for MIPS-based
retrieval in EGR (Engrammatic Geometry Retrieval). The state vector is
a compact geometric fingerprint of a cognitive state — positioned in the
model's own pre-RoPE key manifold for geometrically consistent retrieval.

Three extraction modes:

  mean_pool:   Fast baseline. Mean over heads + context of key matrices
               across extraction layers. Output: [head_dim]. No learned
               parameters. Use for bootstrapping and smoke tests.

  svd_project: Truncated SVD on pre-RoPE keys, extraction layers (D3: 8-31),
               rank-160 for 8B models. Validated by ShadowKV (ICML 2025,
               ByteDance) on Llama-3.1-8B and Phi-3-Mini-128K.
               Output: [rank]. Projection is prompt-dependent — W computed
               per cache via online SVD, not precomputed globally.
               Reference: github.com/ByteDance-Seed/ShadowKV

  xkv_project: Grouped cross-layer SVD. Groups 4 adjacent extraction layers,
               extracts shared basis vectors across the group. Achieves
               6.8x compression vs 2.5x single-layer SVD. K:V rank ratio
               1:1.5 is optimal per xKV paper.
               Reference: github.com/abdelfattah-lab/xKV
               arXiv:2503.18893

REMOVED: sals_project — last-layer-only extraction invalidated by
Layer-Condensed KV Cache (ACL 2024). See D3.

D4: No L2 normalization. True MIPS. L2 norm stored as metadata for
    optional downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from einops import rearrange

from kvcos.core.types import (
    DEFAULT_SVD_RANK,
    ModelCacheSpec,
    StateExtractionMode,
)


@dataclass
class ExtractionResult:
    """Result of state vector extraction from a KV cache."""

    state_vec: torch.Tensor  # [d_out] — the retrieval vector
    l2_norm: float  # stored as metadata per D4
    mode: StateExtractionMode
    n_layers_used: int
    n_tokens: int


@dataclass
class SVDProjection:
    """Learned SVD projection matrix for a specific cache.

    ShadowKV finding: pre-RoPE keys share low-rank subspaces WITHIN
    sequences but differ ACROSS sequences. Projection must be computed
    online per cache, not precomputed globally.
    """

    W: torch.Tensor  # [head_dim, rank] — right singular vectors
    singular_values: torch.Tensor  # [rank] — for diagnostics
    explained_variance_ratio: float  # fraction of variance captured
    source_shape: tuple[int, ...]  # shape of the keys used to compute this


class MARStateExtractor:
    """Extracts retrieval state vectors from KV cache tensors for EGR.

    Usage:
        extractor = MARStateExtractor(mode="svd_project", rank=160)
        result = extractor.extract(keys, spec)
        # result.state_vec is the retrieval vector for FAISS IndexFlatIP
        # result.l2_norm goes into .eng metadata (D4)
    """

    # Max rows fed to SVD. 8192 rows on a 128-dim matrix runs in ~15ms
    # vs ~2000ms for the full 786K-row matrix. Subspace quality is
    # preserved because SVD only needs O(head_dim²) samples to recover
    # the top singular vectors of a low-rank matrix.
    MAX_SVD_ROWS: int = 8192

    def __init__(
        self,
        mode: StateExtractionMode = StateExtractionMode.SVD_PROJECT,
        rank: int = DEFAULT_SVD_RANK,
        xkv_group_size: int = 4,
        xkv_kv_rank_ratio: float = 1.5,
        max_svd_rows: int | None = None,
        layer_range: tuple[int, int] | None = None,
        gate_start: int = 0,
    ):
        self.mode = mode
        self.rank = rank
        self.xkv_group_size = xkv_group_size
        self.xkv_kv_rank_ratio = xkv_kv_rank_ratio
        self.max_svd_rows = max_svd_rows or self.MAX_SVD_ROWS
        # Override spec extraction_layers when set. (8, 24) uses middle
        # layers which encode semantic content (Tenney 2019, Huh 2024).
        self.layer_range = layer_range
        # Skip top gate_start singular values in SVD projection.
        # Top SVs encode shared positional/syntactic structure;
        # skipping them isolates semantic content (gate_start=6 optimal).
        self.gate_start = gate_start

        # Cached projection from last extract call (for inspection/reuse)
        self._last_projection: SVDProjection | None = None

    def extract(
        self,
        keys: torch.Tensor,
        spec: ModelCacheSpec,
    ) -> ExtractionResult:
        """Extract a state vector from KV cache key tensors.

        Args:
            keys: [n_layers, n_kv_heads, ctx_len, head_dim] — the K cache.
                  Must be pre-RoPE if available. Post-RoPE works but with
                  reduced retrieval quality due to position-dependent distortion.
            spec: Model architecture spec (provides extraction_layers).

        Returns:
            ExtractionResult with state vector and metadata.
        """
        n_layers, n_kv_heads, ctx_len, head_dim = keys.shape

        # Layer selection: layer_range overrides spec extraction_layers
        if self.layer_range is not None:
            start, end = self.layer_range
            start = max(0, min(start, n_layers))
            end = max(start, min(end, n_layers))
            layer_indices = list(range(start, end))
        else:
            extraction_layers = spec["extraction_layers"]
            layer_indices = [l for l in extraction_layers if l < n_layers]

        if not layer_indices:
            layer_indices = list(range(n_layers))

        selected_keys = keys[layer_indices]  # [n_selected, n_kv_heads, ctx_len, head_dim]

        match self.mode:
            case StateExtractionMode.MEAN_POOL:
                state_vec = self._mean_pool(selected_keys)
            case StateExtractionMode.SVD_PROJECT:
                state_vec = self._svd_project(selected_keys)
            case StateExtractionMode.XKV_PROJECT:
                state_vec = self._xkv_project(selected_keys)
            case _:
                raise ValueError(f"Unknown extraction mode: {self.mode}")

        # D4: No normalization. True MIPS. Store norm as metadata.
        l2_norm = float(torch.linalg.vector_norm(state_vec).item())

        return ExtractionResult(
            state_vec=state_vec,
            l2_norm=l2_norm,
            mode=self.mode,
            n_layers_used=len(layer_indices),
            n_tokens=ctx_len,
        )

    def _mean_pool(self, keys: torch.Tensor) -> torch.Tensor:
        """Fast baseline: mean over layers, heads, and context positions.

        Input:  [n_layers, n_kv_heads, ctx_len, head_dim]
        Output: [head_dim]
        """
        return keys.float().mean(dim=(0, 1, 2))

    def _svd_project(self, keys: torch.Tensor) -> torch.Tensor:
        """Truncated SVD projection on pre-RoPE keys.

        ShadowKV approach: flatten all extraction layers' keys into a 2D matrix
        [N, head_dim], compute truncated SVD, project onto top-rank singular vectors,
        then mean-pool the projected vectors.

        For large contexts (N > max_svd_rows), we subsample rows before SVD.
        SVD only needs O(head_dim²) samples to recover the top singular vectors
        of a low-rank matrix, so subsampling to 8K rows preserves subspace quality
        while reducing SVD from ~2000ms to ~15ms at 4K context.

        Input:  [n_layers, n_kv_heads, ctx_len, head_dim]
        Output: [rank]
        """
        n_layers, n_kv_heads, ctx_len, head_dim = keys.shape

        # Total rows in the flattened matrix
        n_rows = n_layers * n_kv_heads * ctx_len

        if n_rows > self.max_svd_rows:
            # Subsample BEFORE flatten+cast to avoid allocating the full
            # float32 matrix (saves ~30ms rearrange + 100MB at 4K context).
            gen = torch.Generator()
            gen.manual_seed(42)
            indices = torch.randperm(n_rows, generator=gen)[:self.max_svd_rows]
            flat_keys = keys.reshape(n_rows, head_dim)[indices].float()
            svd_input = flat_keys
        else:
            flat_keys = rearrange(keys.float(), 'l h t d -> (l h t) d')
            svd_input = flat_keys

        # Clamp rank to not exceed matrix dimensions
        max_rank = min(head_dim, svd_input.shape[0])
        effective_rank = min(self.gate_start + self.rank, max_rank)

        # Truncated SVD on (subsampled) matrix
        U, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)

        # W = right singular vectors with gating: skip top gate_start SVs
        # to remove shared positional/syntactic structure
        W = Vh[self.gate_start:effective_rank, :].T

        # Store projection for inspection
        total_var = (S ** 2).sum()
        explained_var = (S[:effective_rank] ** 2).sum()
        self._last_projection = SVDProjection(
            W=W,
            singular_values=S[:effective_rank],
            explained_variance_ratio=float((explained_var / total_var).item()) if total_var > 0 else 0.0,
            source_shape=tuple(keys.shape),
        )

        # Project subsampled rows and mean-pool → [rank]
        # Using the subsample for projection too avoids the expensive
        # 786K × 128 matmul + mean that dominates at large contexts.
        projected = svd_input @ W
        state_vec = projected.mean(dim=0)

        return state_vec

    def _xkv_project(self, keys: torch.Tensor) -> torch.Tensor:
        """Grouped cross-layer SVD (xKV approach).

        Groups adjacent layers (default 4), computes shared SVD basis
        per group, projects keys onto that basis, then concatenates
        group state vectors.

        This captures cross-layer structure that single-layer SVD misses.
        Achieves 6.8x vs 2.5x for single-layer SVD on Llama-3.1-8B.

        K:V rank ratio 1:1.5 is optimal per xKV paper, but since we
        only index keys (D2: K→K retrieval), we use the K rank only.

        Input:  [n_layers, n_kv_heads, ctx_len, head_dim]
        Output: [n_groups * rank_per_group]
        """
        n_layers, n_kv_heads, ctx_len, head_dim = keys.shape

        # Compute rank per group
        # xKV finding: K rank is lower than V rank by factor 1:1.5
        # For 160 total rank budget across groups, allocate per group
        n_groups = max(1, n_layers // self.xkv_group_size)
        rank_per_group = max(1, self.rank // n_groups)
        rank_per_group = min(rank_per_group, head_dim)

        group_vecs: list[torch.Tensor] = []

        for g in range(n_groups):
            start = g * self.xkv_group_size
            end = min(start + self.xkv_group_size, n_layers)
            group_keys = keys[start:end]  # [group_size, n_kv_heads, ctx_len, head_dim]

            # Flatten group
            n_group_rows = group_keys.shape[0] * n_kv_heads * ctx_len

            if n_group_rows > self.max_svd_rows:
                gen = torch.Generator()
                gen.manual_seed(42 + g)
                indices = torch.randperm(n_group_rows, generator=gen)[:self.max_svd_rows]
                svd_input = group_keys.reshape(n_group_rows, head_dim)[indices].float()
            else:
                svd_input = rearrange(group_keys.float(), 'l h t d -> (l h t) d')

            effective_rank = min(rank_per_group, svd_input.shape[0], head_dim)

            # Truncated SVD for this group (on subsampled data)
            U, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)
            W_group = Vh[:effective_rank, :].T  # [head_dim, rank_per_group]

            # Project subsampled rows and mean-pool → [rank_per_group]
            projected = svd_input @ W_group
            group_vec = projected.mean(dim=0)
            group_vecs.append(group_vec)

        # Handle remainder layers (if n_layers not divisible by group_size)
        remainder_start = n_groups * self.xkv_group_size
        if remainder_start < n_layers:
            remainder_keys = keys[remainder_start:]
            n_rem_rows = remainder_keys.shape[0] * n_kv_heads * ctx_len

            if n_rem_rows > self.max_svd_rows:
                gen = torch.Generator()
                gen.manual_seed(42 + n_groups)
                indices = torch.randperm(n_rem_rows, generator=gen)[:self.max_svd_rows]
                svd_input = remainder_keys.reshape(n_rem_rows, head_dim)[indices].float()
            else:
                svd_input = rearrange(remainder_keys.float(), 'l h t d -> (l h t) d')

            effective_rank = min(rank_per_group, svd_input.shape[0], head_dim)
            U, S, Vh = torch.linalg.svd(svd_input, full_matrices=False)
            W_rem = Vh[:effective_rank, :].T
            projected = svd_input @ W_rem
            group_vecs.append(projected.mean(dim=0))

        # Concatenate all group vectors → [n_groups * rank_per_group + remainder]
        state_vec = torch.cat(group_vecs, dim=0)

        return state_vec

    # ── Fixed Corpus Basis (FCB) ────────────────────────────────────────────

    @classmethod
    def compute_corpus_basis(
        cls,
        key_tensors: list[torch.Tensor],
        layer_range: tuple[int, int],
        gate_start: int,
        rank: int,
        max_rows: int = 32768,
        seed: int = 42,
    ) -> torch.Tensor:
        """Compute a fixed projection matrix from a corpus of key tensors.

        Returns P: [rank, head_dim] — the global semantic basis.
        Unlike per-document SVD, this basis is document-independent.
        All documents projected with P exist in the same coordinate system,
        enabling stable cross-document and cross-model comparison.
        """
        l_start, l_end = layer_range
        gen = torch.Generator()
        gen.manual_seed(seed)

        all_rows: list[torch.Tensor] = []
        per_doc_max = max(1, max_rows // len(key_tensors))

        for keys in key_tensors:
            k = keys[l_start:l_end].float()
            n_rows = k.shape[0] * k.shape[1] * k.shape[2]
            flat = k.reshape(n_rows, k.shape[3])
            if flat.shape[0] > per_doc_max:
                idx = torch.randperm(flat.shape[0], generator=gen)[:per_doc_max]
                flat = flat[idx]
            all_rows.append(flat)

        corpus = torch.cat(all_rows, dim=0)
        if corpus.shape[0] > max_rows:
            idx = torch.randperm(corpus.shape[0], generator=gen)[:max_rows]
            corpus = corpus[idx]

        _, S, Vh = torch.linalg.svd(corpus, full_matrices=False)
        P = Vh[gate_start : gate_start + rank]  # [rank, head_dim]
        return P

    def extract_with_basis(
        self,
        keys: torch.Tensor,
        spec: ModelCacheSpec,
        basis: torch.Tensor,
    ) -> ExtractionResult:
        """Extract state vector using a pre-computed fixed corpus basis.

        All vectors computed with the same basis share a coordinate system,
        which is required for cross-model transfer via adapter.

        Args:
            keys: [n_layers, n_kv_heads, n_cells, head_dim]
            spec: Model spec (used for layer_range fallback)
            basis: [rank, head_dim] from compute_corpus_basis()

        Returns:
            ExtractionResult with L2-normalized state vector
        """
        if self.layer_range is not None:
            l_start, l_end = self.layer_range
        else:
            l_start, l_end = 0, keys.shape[0]
        l_start = max(0, min(l_start, keys.shape[0]))
        l_end = max(l_start, min(l_end, keys.shape[0]))

        k = keys[l_start:l_end].float()
        n_rows = k.shape[0] * k.shape[1] * k.shape[2]
        flat = k.reshape(n_rows, k.shape[3])

        proj = flat @ basis.T  # [N_rows, rank]
        vec = proj.mean(dim=0)  # [rank]

        norm = float(torch.linalg.vector_norm(vec).item())
        vec_normed = vec / (norm + 1e-8)

        return ExtractionResult(
            state_vec=vec_normed.to(torch.float32),
            l2_norm=norm,
            mode=self.mode,
            n_layers_used=l_end - l_start,
            n_tokens=k.shape[2],
        )

    # ── Fourier Fingerprint (Engram Absolute) ────────────────────────

    @staticmethod
    def compute_fourier_fingerprint(
        keys: torch.Tensor,
        freqs: tuple[int, ...] = (0, 1),
    ) -> torch.Tensor:
        """Compute the Fourier Absolute fingerprint from KV cache keys.

        Takes the real DFT over the layer dimension, extracts the
        amplitude at the specified frequencies, normalizes each, and
        concatenates them into a single fingerprint vector.

        This fingerprint is:
          - Cross-model invariant (cos ~0.90 between 3B and 8B)
          - Corpus-independent (no basis, no center, no training)
          - Scale-stable (98% recall@1 at N=1000, decay N^-0.207)

        Args:
            keys: [n_layers, n_kv_heads, n_cells, head_dim] — full KV keys.
                  All layers are used (not sliced by layer_range).
            freqs: Frequency indices to extract. Default (0, 1) = DC + 1st harmonic.
                   f=0 captures overall key magnitude profile.
                   f=1 captures dominant oscillation across depth.

        Returns:
            Fingerprint vector [dim * len(freqs)], L2-normalized.
        """
        # Mean over cells (tokens) per layer: [n_layers, n_kv_heads * head_dim]
        n_layers = keys.shape[0]
        layer_means = keys.float().mean(dim=2).reshape(n_layers, -1)

        # DFT over layer dimension
        F_complex = torch.fft.rfft(layer_means, dim=0)  # [n_freq, dim]
        F_amp = F_complex.abs()  # amplitude spectrum

        # Extract and normalize each frequency component
        parts = []
        for f in freqs:
            if f >= F_amp.shape[0]:
                # Frequency out of range — use zeros
                parts.append(torch.zeros(F_amp.shape[1]))
            else:
                v = F_amp[f]
                parts.append(v / (v.norm() + 1e-8))

        fingerprint = torch.cat(parts, dim=0)
        return fingerprint / (fingerprint.norm() + 1e-8)

    @property
    def last_projection(self) -> SVDProjection | None:
        """Access the SVD projection from the last svd_project call.

        Useful for diagnostics: check explained_variance_ratio to validate
        that the rank is sufficient for this particular cache.
        """
        return self._last_projection

    def output_dim(self, spec: ModelCacheSpec) -> int:
        """Compute the output dimension of the state vector for a given spec.

        This is needed to initialize the FAISS index with the correct dimension.
        """
        match self.mode:
            case StateExtractionMode.MEAN_POOL:
                return spec["head_dim"]
            case StateExtractionMode.SVD_PROJECT:
                max_rank = min(self.gate_start + self.rank, spec["head_dim"])
                return max_rank - self.gate_start
            case StateExtractionMode.XKV_PROJECT:
                extraction_layers = spec["extraction_layers"]
                n_layers = len(extraction_layers)
                n_groups = max(1, n_layers // self.xkv_group_size)
                rank_per_group = max(1, self.rank // n_groups)
                rank_per_group = min(rank_per_group, spec["head_dim"])
                # Groups + possible remainder group
                has_remainder = (n_layers % self.xkv_group_size) != 0
                total_groups = n_groups + (1 if has_remainder else 0)
                return total_groups * rank_per_group
            case _:
                raise ValueError(f"Unknown mode: {self.mode}")

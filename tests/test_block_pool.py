"""
ENGRAM Protocol — Block Pool Tests
Tests for 256-token block segmentation/assembly/extend.
"""

from __future__ import annotations

import pytest
import torch

from kvcos.core.block_pool import BlockPool, KVBlock
from kvcos.core.types import BLOCK_SIZE_TOKENS


def _kv(n_layers: int, n_heads: int, ctx: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(n_layers, n_heads, ctx, dim, dtype=torch.float16)
    return k, k.clone()


class TestSegment:
    """Segment full KV cache into 256-token blocks."""

    def test_exact_blocks(self) -> None:
        keys, vals = _kv(32, 8, 512, 128)
        pool = BlockPool(agent_id="a", model_id="m")
        blocks = pool.segment(keys, vals)
        assert len(blocks) == 2
        assert all(b.is_full for b in blocks)

    def test_partial_last_block(self) -> None:
        keys, vals = _kv(32, 8, 300, 128)
        pool = BlockPool(agent_id="a", model_id="m")
        blocks = pool.segment(keys, vals)
        assert len(blocks) == 2
        assert blocks[0].is_full
        assert not blocks[1].is_full
        assert blocks[1].block_len == 44

    def test_total_tokens(self) -> None:
        keys, vals = _kv(32, 8, 700, 128)
        pool = BlockPool(agent_id="a", model_id="m")
        pool.segment(keys, vals)
        assert pool.total_tokens == 700


class TestAssemble:
    """Assemble blocks back into full KV cache."""

    def test_round_trip(self) -> None:
        keys, vals = _kv(4, 2, 512, 64)
        pool = BlockPool(agent_id="a", model_id="m")
        pool.segment(keys, vals)
        k_out, v_out = pool.assemble()
        assert torch.equal(k_out, keys)

    def test_subset_assembly(self) -> None:
        keys, vals = _kv(4, 2, 768, 64)
        pool = BlockPool(agent_id="a", model_id="m")
        pool.segment(keys, vals)
        k_out, _ = pool.assemble(block_indices=[0, 2])
        assert k_out.shape[2] == BLOCK_SIZE_TOKENS * 2

    def test_empty_raises(self) -> None:
        pool = BlockPool(agent_id="a", model_id="m")
        with pytest.raises(ValueError, match="No blocks"):
            pool.assemble()


class TestExtend:
    """Extend pool with new tokens."""

    def test_fills_partial_block(self) -> None:
        keys, vals = _kv(4, 2, 200, 64)
        pool = BlockPool(agent_id="a", model_id="m")
        pool.segment(keys, vals)
        assert not pool.blocks[-1].is_full

        new_k, new_v = _kv(4, 2, 56, 64)
        pool.extend(new_k, new_v)
        assert pool.blocks[-1].is_full
        assert pool.total_tokens == 256

    def test_extend_creates_new_blocks(self) -> None:
        keys, vals = _kv(4, 2, 256, 64)
        pool = BlockPool(agent_id="a", model_id="m")
        pool.segment(keys, vals)
        assert pool.n_blocks == 1

        new_k, new_v = _kv(4, 2, 300, 64)
        pool.extend(new_k, new_v)
        assert pool.n_blocks == 3
        assert pool.total_tokens == 556

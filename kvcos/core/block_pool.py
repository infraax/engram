"""
ENGRAM Protocol — 256-Token Block Pool Manager


Segments a full KV cache into fixed-size blocks (256 tokens each) that can be:
  - Stored independently (one .eng file per block — D7)
  - Retrieved individually via EGR (fine-grained cache hits)
  - Composed (assemble a context from multiple blocks)
  - Evicted independently (LRU per block, not per session)

Design from arXiv:2603.04428 (Persistent Q4 KV Cache, agent-memory paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from kvcos.core.types import BLOCK_SIZE_TOKENS


@dataclass
class KVBlock:
    """A single 256-token block of KV cache data."""

    block_index: int
    token_start: int
    token_end: int  # exclusive

    keys: torch.Tensor  # [n_layers, n_kv_heads, block_len, head_dim]
    values: torch.Tensor  # [n_layers, n_kv_heads, block_len, head_dim]

    @property
    def block_len(self) -> int:
        return self.token_end - self.token_start

    @property
    def is_full(self) -> bool:
        return self.block_len == BLOCK_SIZE_TOKENS

    @property
    def n_layers(self) -> int:
        return self.keys.shape[0]

    @property
    def n_kv_heads(self) -> int:
        return self.keys.shape[1]

    @property
    def head_dim(self) -> int:
        return self.keys.shape[3]


@dataclass
class BlockPool:
    """Manages a collection of KV blocks for an agent session."""

    agent_id: str
    model_id: str
    blocks: list[KVBlock] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(b.block_len for b in self.blocks)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def segment(
        self, keys: torch.Tensor, values: torch.Tensor,
    ) -> list[KVBlock]:
        """Segment a full KV cache into 256-token blocks.

        Args:
            keys:   [n_layers, n_kv_heads, ctx_len, head_dim]
            values: [n_layers, n_kv_heads, ctx_len, head_dim]
        """
        if keys.shape != values.shape:
            raise ValueError(f"Shape mismatch: keys {keys.shape} vs values {values.shape}")

        ctx_len = keys.shape[2]
        blocks: list[KVBlock] = []

        for i in range(0, ctx_len, BLOCK_SIZE_TOKENS):
            end = min(i + BLOCK_SIZE_TOKENS, ctx_len)
            block = KVBlock(
                block_index=len(blocks),
                token_start=i,
                token_end=end,
                keys=keys[:, :, i:end, :].contiguous(),
                values=values[:, :, i:end, :].contiguous(),
            )
            blocks.append(block)

        self.blocks = blocks
        return blocks

    def assemble(
        self, block_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble KV cache from blocks (concatenate along ctx_len dim)."""
        if not self.blocks:
            raise ValueError("No blocks to assemble")

        selected = self.blocks if block_indices is None else [self.blocks[i] for i in block_indices]
        if not selected:
            raise ValueError("No blocks selected for assembly")

        keys = torch.cat([b.keys for b in selected], dim=2)
        values = torch.cat([b.values for b in selected], dim=2)
        return keys, values

    def append_block(self, block: KVBlock) -> None:
        block.block_index = len(self.blocks)
        self.blocks.append(block)

    def get_block(self, index: int) -> KVBlock:
        if index < 0 or index >= len(self.blocks):
            raise IndexError(f"Block index {index} out of range [0, {len(self.blocks)})")
        return self.blocks[index]

    def extend(
        self, new_keys: torch.Tensor, new_values: torch.Tensor,
    ) -> list[KVBlock]:
        """Extend the pool with additional tokens, filling last block first."""
        new_ctx_len = new_keys.shape[2]
        modified_blocks: list[KVBlock] = []
        offset = 0

        if self.blocks and not self.blocks[-1].is_full:
            last = self.blocks[-1]
            space = BLOCK_SIZE_TOKENS - last.block_len
            fill = min(space, new_ctx_len)

            merged_k = torch.cat([last.keys, new_keys[:, :, :fill, :]], dim=2).contiguous()
            merged_v = torch.cat([last.values, new_values[:, :, :fill, :]], dim=2).contiguous()

            self.blocks[-1] = KVBlock(
                block_index=last.block_index,
                token_start=last.token_start,
                token_end=last.token_start + merged_k.shape[2],
                keys=merged_k,
                values=merged_v,
            )
            modified_blocks.append(self.blocks[-1])
            offset = fill

        remaining = new_ctx_len - offset
        if remaining > 0:
            token_base = self.blocks[-1].token_end if self.blocks else 0
            sub_pool = BlockPool(agent_id=self.agent_id, model_id=self.model_id)
            new_blocks = sub_pool.segment(
                new_keys[:, :, offset:, :], new_values[:, :, offset:, :],
            )
            for b in new_blocks:
                b.block_index = len(self.blocks)
                b.token_start += token_base
                b.token_end += token_base
                self.blocks.append(b)
                modified_blocks.append(b)

        return modified_blocks

    def clear(self) -> None:
        self.blocks.clear()

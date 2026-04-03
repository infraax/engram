"""
ENGRAM Protocol — Serializer Tests
Tests for .eng safetensors serialize/deserialize round-trip (D7).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from kvcos.core.serializer import EngramSerializer, SerializationError
from kvcos.core.types import CompressionMethod
from tests.conftest import make_synthetic_kv
from kvcos.core.cache_spec import LLAMA_3_1_8B


class TestSerializeRoundTrip:
    """Serialize → deserialize preserves shape, dtype, metadata."""

    def test_round_trip_shape(self, tmp_data_dir: Path) -> None:
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=256)
        s = EngramSerializer()
        eng = tmp_data_dir / "test.eng"

        s.serialize(
            keys=keys, values=values,
            agent_id="test-agent", task_description="unit test",
            model_id=LLAMA_3_1_8B["model_id"], output_path=eng,
            compression=CompressionMethod.FP16,
        )
        k_out, v_out, meta = s.deserialize(eng)

        assert k_out.shape == keys.shape
        assert v_out.shape == values.shape

    def test_metadata_fields(self, tmp_data_dir: Path) -> None:
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        s = EngramSerializer()
        eng = tmp_data_dir / "meta.eng"

        s.serialize(
            keys=keys, values=values,
            agent_id="agent-42", task_description="metadata check",
            model_id=LLAMA_3_1_8B["model_id"], output_path=eng,
            compression=CompressionMethod.Q8_0,
        )
        _, _, meta = s.deserialize(eng)

        assert meta["agent_id"] == "agent-42"
        assert meta["task_description"] == "metadata check"
        assert meta["compression"] == "q8_0"
        assert meta["n_layers"] == "32"
        assert meta["model_family"] == "llama"

    def test_safetensors_loadable(self, tmp_data_dir: Path) -> None:
        """D7: File must be valid safetensors."""
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        s = EngramSerializer()
        eng = tmp_data_dir / "valid.eng"

        s.serialize(
            keys=keys, values=values,
            agent_id="test", task_description="safetensors check",
            model_id=LLAMA_3_1_8B["model_id"], output_path=eng,
            compression=CompressionMethod.FP16,
        )
        tensors = load_file(str(eng))
        assert "layer_0_keys" in tensors
        assert "layer_0_values" in tensors

    def test_result_dict(self, tmp_data_dir: Path) -> None:
        keys, values = make_synthetic_kv(LLAMA_3_1_8B, ctx_len=64)
        s = EngramSerializer()
        eng = tmp_data_dir / "result.eng"

        result = s.serialize(
            keys=keys, values=values,
            agent_id="test", task_description="result check",
            model_id=LLAMA_3_1_8B["model_id"], output_path=eng,
        )
        assert "cache_id" in result
        assert result["size_bytes"] > 0
        assert result["n_layers"] == 32


class TestSerializerErrors:
    """Edge cases and error handling."""

    def test_shape_mismatch_raises(self, tmp_data_dir: Path) -> None:
        keys = torch.randn(32, 8, 64, 128, dtype=torch.float16)
        values = torch.randn(32, 8, 32, 128, dtype=torch.float16)
        s = EngramSerializer()

        with pytest.raises(SerializationError, match="mismatch"):
            s.serialize(
                keys=keys, values=values,
                agent_id="t", task_description="t",
                model_id="test", output_path=tmp_data_dir / "bad.eng",
            )

    def test_3d_tensor_raises(self, tmp_data_dir: Path) -> None:
        keys = torch.randn(8, 64, 128, dtype=torch.float16)
        s = EngramSerializer()

        with pytest.raises(SerializationError, match="4D"):
            s.serialize(
                keys=keys, values=keys,
                agent_id="t", task_description="t",
                model_id="test", output_path=tmp_data_dir / "bad.eng",
            )

    def test_missing_file_raises(self, tmp_data_dir: Path) -> None:
        s = EngramSerializer()
        with pytest.raises(SerializationError, match="not found"):
            s.deserialize(tmp_data_dir / "nonexistent.eng")

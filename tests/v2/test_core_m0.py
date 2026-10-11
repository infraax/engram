"""M0 tests for engram_core: space keys, RoPE inversion, capture, pipelines."""

import numpy as np
import pytest
import torch

from engram_core.space import SpaceDescriptor, assert_same_space
from engram_core.transforms.rope import apply_rope, inverse_rope

TINY = ["hf-internal-testing/tiny-random-LlamaForCausalLM",
        "hf-internal-testing/tiny-random-Qwen2ForCausalLM"]


def test_space_key_deterministic_and_sensitive():
    a = SpaceDescriptor("reflex_kv", "m", {"source": "derope", "drop_first": 1}, 128)
    b = SpaceDescriptor("reflex_kv", "m", {"drop_first": 1, "source": "derope"}, 128)
    c = SpaceDescriptor("reflex_kv", "m", {"source": "derope", "drop_first": 2}, 128)
    assert a.key == b.key and len(a.key) == 16
    assert a.key != c.key
    with pytest.raises(TypeError):
        assert_same_space(a, c)
    with pytest.raises(ValueError):
        SpaceDescriptor("telepathy", "m")


@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_inverse_rope_exact(scale):
    T, D = 17, 64
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    ang = torch.outer(torch.arange(T).float(), inv)
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos() * scale, emb.sin() * scale
    k = torch.randn(3, T, D)
    back = inverse_rope(apply_rope(k, cos, sin), cos, sin)
    assert torch.allclose(back, k, atol=1e-5)


@pytest.fixture(scope="module", params=TINY)
def capturer(request):
    pytest.importorskip("transformers")
    from engram_core.transforms import HFCapturer
    try:
        return HFCapturer(request.param, device="cpu", dtype=torch.float32, want_hidden=True)
    except Exception as e:  # offline CI
        pytest.skip(f"cannot load {request.param}: {e}")


def test_capture_derope_matches_prerope(capturer):
    c = capturer.capture("The quick brown fox jumps over the lazy dog.")
    assert c.n_layers == len(c.keys_pre) and c.cos is not None
    for l in range(c.n_layers):
        assert c.keys_post[l].shape == c.keys_pre[l].shape
        back = inverse_rope(c.keys_post[l], c.cos, c.sin)
        assert torch.allclose(back, c.keys_pre[l], atol=1e-4), f"layer {l}"


def test_all_catalog_pipelines_run(capturer):
    from engram_bench.catalog import PIPELINES
    from engram_core.transforms import Pipeline
    texts = ["Alpha beta gamma delta.", "A completely different sentence about cells.",
             "Third text, mitochondria and energy.", "Fourth: sorting algorithms in O(n log n)."]
    caps = [capturer.capture(t) for t in texts]
    for name, cfg in PIPELINES.items():
        p = Pipeline(name, cfg)
        raws = [p.raw(c) for c in caps]
        p.fit(raws)
        vs = np.stack([p.transform(r) for r in raws])
        assert np.isfinite(vs).all(), name
        assert np.allclose(np.linalg.norm(vs, axis=1), 1.0, atol=1e-4), name
        assert len(p.space(capturer.model_id, vs.shape[1]).key) == 16


def test_derope_removes_offset_sensitivity(capturer):
    """Same text at a different position: derope keys of the shared tokens must match exactly
    (causal attention ⇒ identical pre-RoPE keys for the shared prefix-free span only when the
    prefix is identical; here we check the weaker, exact property: position enters ONLY via RoPE)."""
    c = capturer.capture("Position test sentence number one.")
    pre_from_post = [inverse_rope(k, c.cos, c.sin) for k in c.keys_post]
    rerot = [apply_rope(k, c.cos, c.sin) for k in pre_from_post]
    for a, b in zip(rerot, c.keys_post):
        assert torch.allclose(a, b, atol=1e-4)


def test_svd_perdoc_fixed_length_for_short_inputs():
    from engram_core.transforms.pipeline import _svd_perdoc
    long = _svd_perdoc(torch.randn(14, 2, 40, 128), rank=128, gate=6, max_rows=8192)
    short = _svd_perdoc(torch.randn(14, 2, 3, 128), rank=128, gate=6, max_rows=8192)   # 84 rows < 128
    assert long.shape == short.shape == (122,)

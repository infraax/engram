"""S0 — capture per-layer key tensors from a HuggingFace causal LM.

Two key sources are captured in the same forward pass:
  * keys_post : what the KV cache actually stores (post-RoPE)  — what v1 fingerprinted
  * keys_pre  : the k_proj (or k_norm, if the model has one) output, i.e. true pre-RoPE keys
Plus the RoPE cos/sin used for this sequence, so S1 can invert the rotation exactly
and we can verify derope(keys_post) == keys_pre.

All tensors are returned on CPU in float32, per layer shaped [n_kv_heads, T, head_dim].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCapture:
    model_id: str
    token_ids: list[int]
    keys_post: list[torch.Tensor]                 # per layer [H_kv, T, D]
    keys_pre: list[torch.Tensor] | None           # per layer [H_kv, T, D]
    cos: torch.Tensor | None                      # [T, D]
    sin: torch.Tensor | None                      # [T, D]
    hidden_mean: list[torch.Tensor] | None = None  # per layer [d_model] (mean over tokens), optional
    meta: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.keys_post)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


def _cache_keys(pkv, layer: int) -> torch.Tensor:
    """Return cached keys [B, H_kv, T, D] across transformers cache API versions."""
    if hasattr(pkv, "layers"):                      # transformers >= 4.56 / 5.x
        return pkv.layers[layer].keys
    if hasattr(pkv, "key_cache"):                   # 4.3x–4.5x DynamicCache
        return pkv.key_cache[layer]
    return pkv[layer][0]                            # legacy tuple


class HFCapturer:
    """Capture KV keys (post- and pre-RoPE) for arbitrary text."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: torch.dtype | str = "auto",
        want_pre: bool = True,
        want_hidden: bool = False,
        add_special_tokens: bool = True,
        max_tokens: int = 4096,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        if dtype == "auto":
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_id)
        full = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.base = full.model.to(self.device).eval()   # skip lm_head: we only need the cache
        del full
        self.cfg = self.base.config
        self.want_pre = want_pre
        self.want_hidden = want_hidden
        self.add_special_tokens = add_special_tokens
        self.max_tokens = max_tokens
        self.n_kv = getattr(self.cfg, "num_key_value_heads", self.cfg.num_attention_heads)
        self.head_dim = getattr(self.cfg, "head_dim", None) or self.cfg.hidden_size // self.cfg.num_attention_heads
        self._pre: dict[int, torch.Tensor] = {}
        self._hooks = []
        if want_pre:
            self._install_hooks()

    # -- hooks ------------------------------------------------------------------
    def _install_hooks(self) -> None:
        for i, layer in enumerate(self.base.layers):
            attn = layer.self_attn
            mod = getattr(attn, "k_norm", None) or attn.k_proj   # pre-RoPE key = last op before rotation
            self._hooks.append(mod.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, i: int):
        def hook(_m, _inp, out):
            self._pre[i] = out.detach()
        return hook

    # -- capture ----------------------------------------------------------------
    @torch.inference_mode()
    def capture(self, text: str) -> KVCapture:
        enc = self.tok(text, return_tensors="pt", add_special_tokens=self.add_special_tokens,
                       truncation=True, max_length=self.max_tokens)
        ids = enc["input_ids"].to(self.device)
        T = ids.shape[1]
        pos = torch.arange(T, device=self.device).unsqueeze(0)
        self._pre.clear()
        out = self.base(input_ids=ids, position_ids=pos, use_cache=True,
                        output_hidden_states=self.want_hidden)
        pkv = out.past_key_values
        L = len(self.base.layers)
        keys_post = [_cache_keys(pkv, l)[0].float().cpu() for l in range(L)]
        keys_pre = None
        if self.want_pre:
            keys_pre = []
            for l in range(L):
                k = self._pre[l][0]                     # [T, H_kv*D] or [T, H_kv, D] (k_norm)
                k = k.reshape(T, self.n_kv, self.head_dim).permute(1, 0, 2)
                keys_pre.append(k.float().cpu())
        cos = sin = None
        if hasattr(self.base, "rotary_emb"):
            probe = torch.zeros(1, T, self.head_dim, device=self.device, dtype=self.dtype)
            c, s = self.base.rotary_emb(probe, pos)
            cos, sin = c[0].float().cpu(), s[0].float().cpu()
        hidden = None
        if self.want_hidden and out.hidden_states is not None:
            hidden = [h[0].float().mean(0).cpu() for h in out.hidden_states[1:]]
        return KVCapture(self.model_id, ids[0].tolist(), keys_post, keys_pre, cos, sin, hidden,
                         meta={"dtype": str(self.dtype), "device": self.device})

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

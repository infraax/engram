"""Named pipeline configs for M0/P0 (each name = one candidate space)."""

FOURIER = {"type": "rfft_amp", "freqs": [0, 1], "normalize_layers": True}
MID = [0.25, 0.75]          # v1 MAR used layers 8–24 of 32

PIPELINES: dict[str, dict] = {
    # ── v1 reproductions ──────────────────────────────────────────────────────
    "v1_fourier":            {"source": "post", "layers": "all", "axis": FOURIER},
    "v1_fourier_rawlayers":  {"source": "post", "layers": "all",
                              "axis": {"type": "rfft_amp", "freqs": [0, 1], "normalize_layers": False}},
    "v1_svd_gate6":          {"source": "post", "layers": MID,
                              "axis": {"type": "svd_perdoc", "rank": 128, "gate": 6}},
    "v1_svd_gate0":          {"source": "post", "layers": MID,
                              "axis": {"type": "svd_perdoc", "rank": 128, "gate": 0}},
    # ── H1: position (RoPE) ──────────────────────────────────────────────────
    "fourier_derope":        {"source": "derope", "layers": "all", "axis": FOURIER},
    "fourier_pre":           {"source": "pre", "layers": "all", "axis": FOURIER},
    # ── H2: sinks + centering vs gating ──────────────────────────────────────
    "fourier_derope_sink1":  {"source": "derope", "layers": "all", "drop_first": 1, "axis": FOURIER},
    "fourier_derope_sink1_c": {"source": "derope", "layers": "all", "drop_first": 1, "axis": FOURIER,
                               "center": True},
    "fourier_derope_sink1_abtt": {"source": "derope", "layers": "all", "drop_first": 1, "axis": FOURIER,
                                  "center": True, "remove_pcs": "bbp"},
    "svd_derope_sink1_gate0": {"source": "derope", "layers": MID, "drop_first": 1,
                               "axis": {"type": "svd_perdoc", "rank": 128, "gate": 0}},
    "concat_mid_derope_sink1_c": {"source": "derope", "layers": MID, "drop_first": 1,
                                  "axis": {"type": "concat", "normalize_layers": True}, "center": True},
    "concat_mid_derope_sink1_abtt": {"source": "derope", "layers": MID, "drop_first": 1,
                                     "axis": {"type": "concat", "normalize_layers": True},
                                     "center": True, "remove_pcs": "bbp"},
    # ── control: the model's own hidden states (is K-geometry special?) ──────
    "hidden_mid_c":          {"source": "hidden", "layers": [0.5, 0.75], "axis": {"type": "layer_mean"},
                              "center": True},
    "hidden_mid_abtt":       {"source": "hidden", "layers": [0.5, 0.75], "axis": {"type": "layer_mean"},
                              "center": True, "remove_pcs": "bbp"},
}

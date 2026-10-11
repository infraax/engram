### c200-question

| system | pipeline | R@1 | R@10 | nDCG@10 | margin | mean\|cos\| | stab self-cos (min) | id R@1 (min) |
|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | (embedder) | 0.975 | 1.000 | 0.990 | +0.1612 | 0.494 | 0.945 | 1.00 |
| all-MiniLM-L6-v2 | (embedder) | 0.970 | 1.000 | 0.985 | +0.3405 | 0.081 | 0.880 | 1.00 |
| SmolLM2-1.7B-Instruct | concat_mid_derope_sink1_abtt | 0.815 | 0.970 | 0.897 | +0.0692 | 0.252 | 0.736 | 1.00 |
| Qwen2.5-1.5B-Instruct | concat_mid_derope_sink1_abtt | 0.740 | 0.915 | 0.832 | +0.0584 | 0.281 | 0.759 | 1.00 |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1_abtt | 0.680 | 0.950 | 0.815 | +0.0478 | 0.258 | 0.720 | 1.00 |
| Qwen2.5-0.5B-Instruct | concat_mid_derope_sink1_abtt | 0.450 | 0.835 | 0.650 | +0.0089 | 0.271 | 0.754 | 1.00 |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1_abtt | 0.430 | 0.785 | 0.588 | -0.0218 | 0.264 | 0.747 | 1.00 |
| Qwen2.5-1.5B-Instruct | concat_mid_derope_sink1_c | 0.360 | 0.720 | 0.535 | -0.0331 | 0.480 | 0.839 | 1.00 |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1_abtt | 0.355 | 0.645 | 0.486 | -0.0421 | 0.318 | 0.775 | 0.99 |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1_c | 0.310 | 0.655 | 0.462 | -0.0499 | 0.429 | 0.821 | 0.99 |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1 | 0.310 | 0.655 | 0.462 | -0.0064 | 0.945 | 0.984 | 0.99 |
| Qwen2.5-1.5B-Instruct | fourier_derope | 0.310 | 0.650 | 0.460 | -0.0065 | 0.945 | 0.984 | 0.99 |
| Qwen2.5-1.5B-Instruct | fourier_pre | 0.310 | 0.650 | 0.460 | -0.0065 | 0.945 | 0.984 | 0.99 |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1_c | 0.325 | 0.605 | 0.454 | -0.0536 | 0.415 | 0.781 | 1.00 |
| SmolLM2-1.7B-Instruct | concat_mid_derope_sink1_c | 0.325 | 0.600 | 0.449 | -0.0625 | 0.424 | 0.805 | 1.00 |
| Qwen2.5-1.5B-Instruct | v1_fourier | 0.290 | 0.640 | 0.448 | -0.0070 | 0.941 | 0.978 | 0.98 |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1 | 0.305 | 0.605 | 0.442 | -0.0030 | 0.973 | 0.991 | 0.99 |
| Qwen2.5-0.5B-Instruct | concat_mid_derope_sink1_c | 0.280 | 0.610 | 0.430 | -0.0528 | 0.538 | 0.857 | 0.94 |
| Qwen2.5-1.5B-Instruct | v1_fourier_rawlayers | 0.270 | 0.615 | 0.426 | -0.0001 | 0.999 | 1.000 | 0.98 |
| SmolLM2-1.7B-Instruct | fourier_derope | 0.285 | 0.590 | 0.426 | -0.0033 | 0.973 | 0.991 | 0.98 |
| SmolLM2-1.7B-Instruct | fourier_pre | 0.285 | 0.590 | 0.426 | -0.0033 | 0.973 | 0.991 | 0.98 |
| Qwen2.5-0.5B-Instruct | hidden_mid_abtt | 0.270 | 0.480 | 0.374 | -0.0227 | 0.773 | 0.917 | 1.00 |
| Qwen2.5-1.5B-Instruct | hidden_mid_abtt | 0.260 | 0.490 | 0.366 | -0.0265 | 0.742 | 0.887 | 1.00 |
| SmolLM2-1.7B-Instruct | v1_fourier | 0.235 | 0.520 | 0.361 | -0.0044 | 0.972 | 0.987 | 0.90 |
| SmolLM2-1.7B-Instruct | v1_fourier_rawlayers | 0.220 | 0.525 | 0.356 | -0.0045 | 0.973 | 0.987 | 0.91 |
| SmolLM2-1.7B-Instruct | hidden_mid_abtt | 0.220 | 0.455 | 0.325 | -0.0367 | 0.704 | 0.895 | 1.00 |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1 | 0.200 | 0.480 | 0.321 | -0.0093 | 0.956 | 0.986 | 0.90 |
| Qwen2.5-1.5B-Instruct | hidden_mid_c | 0.215 | 0.440 | 0.316 | -0.0003 | 0.998 | 0.999 | 0.99 |
| Qwen2.5-0.5B-Instruct | hidden_mid_c | 0.200 | 0.470 | 0.315 | -0.0018 | 0.990 | 0.996 | 1.00 |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1_c | 0.185 | 0.490 | 0.314 | -0.0803 | 0.537 | 0.843 | 0.90 |
| Qwen2.5-0.5B-Instruct | fourier_derope | 0.190 | 0.475 | 0.314 | -0.0092 | 0.957 | 0.986 | 0.92 |
| Qwen2.5-0.5B-Instruct | fourier_pre | 0.190 | 0.475 | 0.314 | -0.0092 | 0.957 | 0.986 | 0.92 |
| Qwen2.5-0.5B-Instruct | v1_fourier | 0.190 | 0.470 | 0.309 | -0.0096 | 0.954 | 0.982 | 0.81 |
| SmolLM2-1.7B-Instruct | hidden_mid_c | 0.165 | 0.390 | 0.267 | -0.0011 | 0.994 | 0.998 | 1.00 |
| Qwen2.5-0.5B-Instruct | v1_fourier_rawlayers | 0.140 | 0.380 | 0.248 | -0.0006 | 0.998 | 0.999 | 0.67 |
| Qwen2.5-1.5B-Instruct | svd_derope_sink1_gate0 | 0.030 | 0.085 | 0.054 | -0.2891 | 0.617 | 0.626 | 0.01 |
| Qwen2.5-1.5B-Instruct | v1_svd_gate6 | 0.010 | 0.085 | 0.040 | -0.5653 | 0.196 | -0.029 | 0.00 |
| SmolLM2-1.7B-Instruct | v1_svd_gate0 | 0.005 | 0.075 | 0.034 | -0.5574 | 0.198 | 0.013 | 0.01 |
| Qwen2.5-0.5B-Instruct | svd_derope_sink1_gate0 | 0.005 | 0.080 | 0.033 | -0.0519 | 0.943 | 0.925 | 0.05 |
| Qwen2.5-1.5B-Instruct | v1_svd_gate0 | 0.015 | 0.050 | 0.031 | -0.7732 | 0.593 | -0.047 | 0.01 |
| SmolLM2-1.7B-Instruct | v1_svd_gate6 | 0.005 | 0.055 | 0.029 | -0.4980 | 0.191 | 0.006 | 0.01 |
| Qwen2.5-0.5B-Instruct | v1_svd_gate6 | 0.015 | 0.045 | 0.027 | -0.6647 | 0.227 | -0.030 | 0.01 |
| SmolLM2-1.7B-Instruct | svd_derope_sink1_gate0 | 0.005 | 0.055 | 0.026 | -0.5623 | 0.219 | -0.022 | 0.02 |
| Qwen2.5-0.5B-Instruct | v1_svd_gate0 | 0.000 | 0.050 | 0.022 | -0.8491 | 0.950 | -0.429 | 0.00 |

### c200-paraphrase

| system | pipeline | R@1 | R@10 | nDCG@10 | margin | mean\|cos\| | stab self-cos (min) | id R@1 (min) |
|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | (embedder) | 1.000 | 1.000 | 1.000 | +0.1989 | 0.494 | nan | nan |
| SmolLM2-1.7B-Instruct | concat_mid_derope_sink1_abtt | 1.000 | 1.000 | 1.000 | +0.2819 | 0.252 | nan | nan |
| Qwen2.5-1.5B-Instruct | concat_mid_derope_sink1_abtt | 1.000 | 1.000 | 1.000 | +0.2794 | 0.281 | nan | nan |
| all-MiniLM-L6-v2 | (embedder) | 0.990 | 1.000 | 0.996 | +0.4098 | 0.081 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1_abtt | 0.985 | 1.000 | 0.994 | +0.2637 | 0.258 | nan | nan |
| Qwen2.5-0.5B-Instruct | concat_mid_derope_sink1_abtt | 0.985 | 1.000 | 0.994 | +0.2472 | 0.271 | nan | nan |
| Qwen2.5-1.5B-Instruct | concat_mid_derope_sink1_c | 0.985 | 1.000 | 0.994 | +0.1467 | 0.480 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1_abtt | 0.980 | 1.000 | 0.993 | +0.2435 | 0.264 | nan | nan |
| Qwen2.5-0.5B-Instruct | concat_mid_derope_sink1_c | 0.975 | 1.000 | 0.990 | +0.1150 | 0.538 | nan | nan |
| SmolLM2-1.7B-Instruct | hidden_mid_abtt | 0.975 | 1.000 | 0.989 | +0.0900 | 0.704 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1_c | 0.965 | 1.000 | 0.987 | +0.1569 | 0.429 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1 | 0.970 | 1.000 | 0.987 | +0.0073 | 0.973 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1_c | 0.970 | 1.000 | 0.986 | +0.1631 | 0.415 | nan | nan |
| SmolLM2-1.7B-Instruct | concat_mid_derope_sink1_c | 0.965 | 1.000 | 0.986 | +0.1592 | 0.424 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_fourier_rawlayers | 0.965 | 1.000 | 0.986 | +0.0001 | 0.999 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_fourier | 0.970 | 1.000 | 0.986 | +0.0072 | 0.972 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope | 0.965 | 1.000 | 0.985 | +0.0073 | 0.973 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_pre | 0.965 | 1.000 | 0.985 | +0.0073 | 0.973 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_fourier_rawlayers | 0.965 | 1.000 | 0.984 | +0.0071 | 0.973 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_fourier | 0.960 | 1.000 | 0.983 | +0.0156 | 0.941 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope | 0.955 | 1.000 | 0.981 | +0.0147 | 0.945 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_pre | 0.955 | 1.000 | 0.981 | +0.0147 | 0.945 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1 | 0.955 | 1.000 | 0.981 | +0.0147 | 0.945 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope | 0.960 | 0.995 | 0.980 | +0.0092 | 0.957 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_pre | 0.960 | 0.995 | 0.980 | +0.0092 | 0.957 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1 | 0.960 | 0.995 | 0.980 | +0.0093 | 0.956 | nan | nan |
| Qwen2.5-1.5B-Instruct | hidden_mid_abtt | 0.960 | 1.000 | 0.980 | +0.0742 | 0.742 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1_abtt | 0.965 | 0.995 | 0.979 | +0.1716 | 0.318 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_fourier | 0.955 | 0.995 | 0.978 | +0.0095 | 0.954 | nan | nan |
| Qwen2.5-0.5B-Instruct | hidden_mid_abtt | 0.950 | 0.995 | 0.975 | +0.0621 | 0.773 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1_c | 0.950 | 0.995 | 0.973 | +0.0974 | 0.537 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_fourier_rawlayers | 0.935 | 0.990 | 0.963 | +0.0004 | 0.998 | nan | nan |
| Qwen2.5-1.5B-Instruct | hidden_mid_c | 0.935 | 0.985 | 0.961 | +0.0005 | 0.998 | nan | nan |
| SmolLM2-1.7B-Instruct | hidden_mid_c | 0.920 | 0.995 | 0.960 | +0.0015 | 0.994 | nan | nan |
| Qwen2.5-0.5B-Instruct | hidden_mid_c | 0.935 | 0.985 | 0.958 | +0.0020 | 0.990 | nan | nan |
| Qwen2.5-0.5B-Instruct | svd_derope_sink1_gate0 | 0.055 | 0.185 | 0.121 | -0.0440 | 0.943 | nan | nan |
| Qwen2.5-1.5B-Instruct | svd_derope_sink1_gate0 | 0.030 | 0.135 | 0.076 | -0.2763 | 0.617 | nan | nan |
| SmolLM2-1.7B-Instruct | svd_derope_sink1_gate0 | 0.050 | 0.105 | 0.075 | -0.5793 | 0.219 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_svd_gate6 | 0.025 | 0.105 | 0.058 | -0.5487 | 0.191 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_svd_gate0 | 0.015 | 0.100 | 0.052 | -0.5756 | 0.198 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_svd_gate6 | 0.010 | 0.105 | 0.046 | -0.5882 | 0.196 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_svd_gate6 | 0.020 | 0.080 | 0.045 | -0.6817 | 0.227 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_svd_gate0 | 0.020 | 0.070 | 0.044 | -0.8972 | 0.593 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_svd_gate0 | 0.015 | 0.085 | 0.039 | -0.8769 | 0.950 | nan | nan |

### scifact

| system | pipeline | R@1 | R@10 | nDCG@10 | margin | mean\|cos\| | stab self-cos (min) | id R@1 (min) |
|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | (embedder) | 0.590 | 0.840 | 0.706 | +0.0145 | 0.627 | nan | nan |
| all-MiniLM-L6-v2 | (embedder) | 0.503 | 0.800 | 0.648 | +0.0007 | 0.148 | nan | nan |
| SmolLM2-1.7B-Instruct | hidden_mid_abtt | 0.217 | 0.480 | 0.324 | -0.0532 | 0.617 | nan | nan |
| Qwen2.5-1.5B-Instruct | hidden_mid_abtt | 0.160 | 0.443 | 0.286 | -0.0555 | 0.691 | nan | nan |
| SmolLM2-1.7B-Instruct | concat_mid_derope_sink1_abtt | 0.143 | 0.417 | 0.252 | -0.0562 | 0.693 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1_abtt | 0.117 | 0.403 | 0.244 | -0.0606 | 0.626 | nan | nan |
| Qwen2.5-1.5B-Instruct | concat_mid_derope_sink1_abtt | 0.097 | 0.403 | 0.227 | -0.0633 | 0.744 | nan | nan |
| Qwen2.5-0.5B-Instruct | hidden_mid_abtt | 0.093 | 0.307 | 0.182 | -0.0785 | 0.724 | nan | nan |
| SmolLM2-1.7B-Instruct | hidden_mid_c | 0.077 | 0.260 | 0.154 | -0.0015 | 0.999 | nan | nan |
| Qwen2.5-1.5B-Instruct | hidden_mid_c | 0.057 | 0.240 | 0.136 | -0.0006 | 1.000 | nan | nan |
| Qwen2.5-0.5B-Instruct | concat_mid_derope_sink1_abtt | 0.063 | 0.207 | 0.122 | -0.0843 | 0.805 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1_abtt | 0.060 | 0.197 | 0.112 | -0.0997 | 0.712 | nan | nan |
| Qwen2.5-0.5B-Instruct | hidden_mid_c | 0.043 | 0.170 | 0.096 | -0.0025 | 0.999 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1_c | 0.053 | 0.143 | 0.089 | -0.1388 | 0.695 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope_sink1 | 0.047 | 0.140 | 0.084 | -0.0062 | 0.989 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_derope | 0.043 | 0.140 | 0.082 | -0.0063 | 0.989 | nan | nan |
| SmolLM2-1.7B-Instruct | fourier_pre | 0.043 | 0.140 | 0.082 | -0.0063 | 0.989 | nan | nan |
| SmolLM2-1.7B-Instruct | concat_mid_derope_sink1_c | 0.030 | 0.100 | 0.060 | -0.1591 | 0.738 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1_c | 0.020 | 0.100 | 0.056 | -0.1749 | 0.682 | nan | nan |
| Qwen2.5-1.5B-Instruct | concat_mid_derope_sink1_c | 0.030 | 0.093 | 0.053 | -0.1527 | 0.743 | nan | nan |
| Qwen2.5-0.5B-Instruct | concat_mid_derope_sink1_c | 0.020 | 0.077 | 0.042 | -0.1893 | 0.770 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1_abtt | 0.023 | 0.063 | 0.040 | -0.1694 | 0.744 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope_sink1 | 0.023 | 0.047 | 0.033 | -0.0193 | 0.972 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_derope | 0.023 | 0.043 | 0.032 | -0.0195 | 0.972 | nan | nan |
| Qwen2.5-1.5B-Instruct | fourier_pre | 0.023 | 0.043 | 0.032 | -0.0195 | 0.972 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1_c | 0.020 | 0.040 | 0.027 | -0.1884 | 0.785 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope_sink1 | 0.017 | 0.043 | 0.025 | -0.0199 | 0.981 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_derope | 0.017 | 0.040 | 0.024 | -0.0203 | 0.981 | nan | nan |
| Qwen2.5-0.5B-Instruct | fourier_pre | 0.017 | 0.040 | 0.024 | -0.0203 | 0.981 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_fourier | 0.007 | 0.020 | 0.011 | -0.0449 | 0.960 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_fourier_rawlayers | 0.003 | 0.020 | 0.010 | -0.0004 | 1.000 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_fourier | 0.000 | 0.017 | 0.006 | -0.0460 | 0.967 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_fourier_rawlayers | 0.003 | 0.010 | 0.006 | -0.0269 | 0.983 | nan | nan |
| SmolLM2-1.7B-Instruct | svd_derope_sink1_gate0 | 0.003 | 0.010 | 0.005 | -0.6457 | 0.243 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_fourier | 0.003 | 0.007 | 0.005 | -0.0278 | 0.983 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_fourier_rawlayers | 0.000 | 0.013 | 0.005 | -0.0029 | 0.998 | nan | nan |
| Qwen2.5-1.5B-Instruct | svd_derope_sink1_gate0 | 0.000 | 0.010 | 0.004 | -0.4942 | 0.585 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_svd_gate0 | 0.003 | 0.003 | 0.003 | -0.8130 | 0.597 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_svd_gate0 | 0.000 | 0.007 | 0.003 | -0.9625 | 0.955 | nan | nan |
| Qwen2.5-0.5B-Instruct | v1_svd_gate6 | 0.000 | 0.003 | 0.000 | -0.7506 | 0.215 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_svd_gate6 | 0.000 | 0.000 | 0.000 | -0.6703 | 0.222 | nan | nan |
| SmolLM2-1.7B-Instruct | v1_svd_gate0 | 0.000 | 0.000 | 0.000 | -0.6856 | 0.244 | nan | nan |
| Qwen2.5-0.5B-Instruct | svd_derope_sink1_gate0 | 0.000 | 0.000 | 0.000 | -0.5398 | 0.941 | nan | nan |
| Qwen2.5-1.5B-Instruct | v1_svd_gate6 | 0.000 | 0.000 | 0.000 | -0.6967 | 0.217 | nan | nan |

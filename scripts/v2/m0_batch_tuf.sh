#!/bin/bash
# M0 batch on TUF — sequential so the 6 GB GPU is never shared
set -x
cd ~/dev/engram && source .venv/bin/activate
export PYTHONPATH=. TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=0
[ -s data/c200_queries.jsonl ] || python scripts/v2/gen_c200_queries.py --model qwen2.5:7b
ollama stop qwen2.5:7b || true
python scripts/v2/run_m0.py --task baselines --datasets c200,scifact
python scripts/v2/run_m0.py --task kv --model Qwen/Qwen2.5-0.5B-Instruct --dtype fp32 --datasets c200,scifact
python scripts/v2/run_m0.py --task kv --model Qwen/Qwen2.5-1.5B-Instruct --dtype fp16 --datasets c200,scifact
python scripts/v2/run_m0.py --task kv --model HuggingFaceTB/SmolLM2-1.7B-Instruct --dtype fp16 --datasets c200,scifact
echo M0_BATCH_DONE

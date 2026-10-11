#!/bin/bash
cd ~/dev/engram && source .venv/bin/activate; export PYTHONPATH=. TOKENIZERS_PARALLELISM=false
while pgrep -f m0_batch_tuf.sh >/dev/null; do sleep 30; done
python scripts/v2/run_m0.py --task kv --model Qwen/Qwen2.5-1.5B-Instruct --dtype fp16 --datasets c200,scifact
echo M0_FOLLOWUP_DONE

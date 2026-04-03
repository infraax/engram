# ENGRAM Protocol

KV cache fingerprinting for persistent cross-session semantic retrieval.

ENGRAM extracts Fourier fingerprints from LLM KV caches, stores them as compact binary certificates (`.eng` files), and retrieves them via HNSW approximate nearest neighbor search. This enables cross-session memory, semantic deduplication, and KV cache restoration for large language models.

## Key Features

- **Fourier fingerprinting** of LLM KV caches (f0+f1 DFT decomposition)
- **4-stage geodesic retrieval** pipeline with confidence scoring
- **HNSW index** (faiss) for sub-millisecond search at scale
- **Multi-architecture support**: Llama, Gemma, Gemma 4 (ISWA), Phi, Qwen, Mistral
- **EIGENGRAM binary format** (v1.2) - portable, versioned `.eng` certificates
- **MCP server** for Claude Code session memory integration
- **Knowledge index** for semantic search over markdown documentation

## Metrics

| Metric | Value |
|---|---|
| Recall@1 (N=200) | 100.0% |
| HNSW speedup | 5.7x vs brute-force |
| Tests | 220 passing |
| Architectures | llama, gemma, gemma4/ISWA, phi, qwen, mistral |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/engram-protocol/engram.git
cd engram
./scripts/setup.sh

# Run tests
source .venv/bin/activate
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. pytest tests/ -x -q

# Start the API server
cp .env.template .env
# Edit .env with your model path
engram-server
```

## Installation

### From source (recommended)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Optional: sentence-transformers embedder (recommended)
pip install -e ".[sbert]"

# Optional: MCP server for Claude Code
pip install -e ".[mcp]"

# Optional: development tools
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.11
- PyTorch >= 2.3.0
- faiss-cpu >= 1.8.0 (pip install, not conda-forge)

## Project Structure

```
kvcos/                     Core library
  core/                    Foundation: types, parsing, fingerprinting, extraction
    types.py               Type system (ModelCacheSpec, CacheSection, AttentionType)
    blob_parser.py         llama.cpp state blob parser (standard + ISWA)
    fingerprint.py         Fourier fingerprint computation (v1, v2, ISWA)
    cache_spec.py          Model registry (Llama, Gemma, Phi, Qwen, Mistral)
    state_extractor.py     MAR state extraction (SVD, mean_pool, xKV)
    manifold_index.py      FAISS IndexFlatIP wrapper
    retriever.py           EGR retrieval pipeline orchestrator
    serializer.py          EIGENGRAM compression codec
    config.py              Centralized pydantic-settings config
  engram/                  High-level retrieval and session memory
    retrieval.py           4-stage geodesic retrieval pipeline
    hnsw_index.py          HNSW index wrapper (EngramIndex)
    index_c.py             SQLite confidence history (IndexC)
    knowledge_index.py     HNSW over knowledge .eng files
    embedder.py            Unified fingerprint: llama_cpp > sbert > hash
    format.py              EIGENGRAM binary format v1.2 codec
    chunker.py             Markdown-aware semantic chunker
    manifest.py            Knowledge index manifest registry
    session_propagator.py  Session lifecycle manager
    metadata_disambiguate.py  Stage 4 metadata tiebreaker
  api/                     FastAPI REST API
    server.py              Application factory + lifespan
    routes.py              API route handlers
    schemas.py             Pydantic request/response models
  client/                  Python client library
    python_client.py       Sync HTTP client (EngramClient)
  storage/                 Storage backends
    local.py               Local filesystem backend
integrations/              External LLM runtime bridges
  llama_cpp_bridge.py      llama-cpp-python bridge (KV extraction + injection)
mcp/                       MCP server for Claude Code
  engram_memory.py         7 tools: session + knowledge memory
scripts/                   CLI utilities
  setup.sh                 One-command environment setup
  index_knowledge.py       Batch markdown indexer
  demo_agent_session.py    End-to-end demo
tests/                     220 tests (pytest)
```

## Retrieval Pipeline

ENGRAM uses a 4-stage geodesic retrieval pipeline with confidence scoring:

```
Stage 0: Prior preemption     IndexC chronic failure -> skip HNSW
Stage 1: HNSW search          -> HIGH / MEDIUM confidence
Stage 2: Trajectory correction -> MEDIUM (interpolation w=0.3)
Stage 3: Negative constraints  -> LOW (apophatic layer)
Stage 4: Metadata disambig     -> LOW + stage4_used=True
```

Entry point: `geodesic_retrieve_stage4()` in `kvcos/engram/retrieval.py`

## Configuration

Copy `.env.template` to `.env` and configure:

```bash
ENGRAM_PORT=8080              # API server port
ENGRAM_DATA_DIR=~/.engram/data  # Storage directory
ENGRAM_MODEL_PATH=            # Path to GGUF model file
ENGRAM_N_CTX=16384            # Context window size
```

All configuration uses the `ENGRAM_` prefix via pydantic-settings.

## MCP Integration (Claude Code)

ENGRAM includes an MCP server for persistent session memory:

```bash
# Register globally
claude mcp add --global engram-memory \
  -e ENGRAM_SESSIONS_DIR=~/.engram/sessions \
  -- python3 mcp/engram_memory.py
```

**Tools**: `write_session_engram`, `get_last_session`, `retrieve_relevant_sessions`, `get_relevant_context`, `list_indexed`, `index_knowledge`

## Multi-Architecture Support

ENGRAM supports standard and ISWA (Interleaved Sliding Window Attention) models:

| Architecture | Type | Fingerprint Dim |
|---|---|---|
| Llama 3.x | Standard | 2048 |
| Gemma 2 | Standard | 2048 |
| Gemma 4 26B | ISWA (dual-cache) | 6144 |
| Phi 3 Mini | Standard | 768 |
| Qwen 2.5 | Standard | 256 |
| Mistral 7B | Standard | 2048 |

## License

Apache 2.0

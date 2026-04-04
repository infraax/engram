# engram-mcp

TypeScript client for the [ENGRAM](https://github.com/infraax/engram) session memory MCP server.

KV cache fingerprinting for persistent cross-session LLM memory. Fourier decomposition achieves 98% Recall@1 at 51us.

## Install

```bash
npm install engram-mcp
```

**Prerequisite:** The Python ENGRAM server must be installed:

```bash
pip install engram-kv
```

## Quick Start

```typescript
import { EngramClient } from "engram-mcp";

const client = new EngramClient();
await client.connect();

// Load last session state
const last = await client.getLastSession();
console.log(last.terminal_state);

// Search past sessions
const matches = await client.retrieveRelevantSessions({
  query: "KV cache fingerprinting experiments",
  k: 3,
});

// Save session state
await client.writeSessionEngram({
  session_summary: `
    VALIDATED: 220 tests passing, 100% recall
    CURRENT: kvcos/engram/retrieval.py, 4-stage pipeline
    NEXT: cross-model transfer experiments
    OPEN: FCDB scaling at N>50
  `,
  session_id: "s7_2026-04-03",
  domain: "engram",
});

// Search knowledge index
const docs = await client.getRelevantContext({
  query: "Fourier fingerprint frequency ablation",
  k: 5,
});

await client.disconnect();
```

## API

### Session Tools

| Method | Description |
|--------|-------------|
| `writeSessionEngram(params)` | Store terminal session state as .eng binary |
| `getLastSession()` | Load most recent session (fast path, no search) |
| `retrieveRelevantSessions(params)` | Semantic search over session memories |

### Knowledge Tools

| Method | Description |
|--------|-------------|
| `getRelevantContext(params)` | HNSW semantic search over knowledge index |
| `listIndexed(project?)` | List indexed files and chunk counts |
| `indexKnowledge(params)` | Index markdown files into knowledge .eng |

### Configuration

```typescript
const client = new EngramClient({
  pythonPath: "/path/to/python3",     // default: "python3"
  serverPath: "/path/to/engram_memory.py",
  sessionsDir: "~/.engram/sessions",
  knowledgeDir: "~/.engram/knowledge",
});
```

## How It Works

ENGRAM fingerprints text using Fourier decomposition of embedding vectors, producing compact ~800-byte binary certificates (.eng files). These are indexed in an HNSW graph for sub-millisecond semantic retrieval.

The MCP server exposes this as a [Model Context Protocol](https://modelcontextprotocol.io/) service, and this npm package provides a typed TypeScript client.

## Links

- [PyPI: engram-kv](https://pypi.org/project/engram-kv/)
- [GitHub](https://github.com/infraax/engram)
- [Paper: "You Don't Need Adapters"](https://github.com/infraax/engram#paper)

## License

Apache-2.0

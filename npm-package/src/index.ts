/**
 * engram-mcp — TypeScript client for the ENGRAM Session Memory MCP Server
 *
 * Provides typed wrappers around the 7 ENGRAM MCP tools:
 *   Session:   writeSessionEngram, getLastSession, retrieveRelevantSessions
 *   Knowledge: getRelevantContext, listIndexed, indexKnowledge
 *
 * Usage:
 *   import { EngramClient } from "engram-mcp";
 *
 *   const client = new EngramClient();
 *   await client.connect();
 *
 *   const last = await client.getLastSession();
 *   console.log(last.terminal_state);
 *
 *   await client.writeSessionEngram({
 *     session_summary: "VALIDATED: 220 tests passing\\nCURRENT: ...",
 *     session_id: "s7_2026-04-03",
 *     domain: "engram",
 *   });
 *
 * @module engram-mcp
 * @author ENIGMA
 * @license Apache-2.0
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// ── Response Types ──────────────────────────────────────────────────

export interface WriteSessionResult {
  readonly stored: string;
  readonly session_id: string;
  readonly fp_source: string;
  readonly chars: number;
}

export interface LastSessionResult {
  readonly session_id: string | null;
  readonly terminal_state: string | null;
  readonly stored_at: number | null;
  readonly fp_source: string | null;
  readonly status?: string;
}

export interface SessionMatch {
  readonly session_id: string;
  readonly terminal_state: string;
  readonly similarity: number;
  readonly fp_source: string;
}

export interface KnowledgeMatch {
  readonly content: string;
  readonly source_path: string;
  readonly project: string;
  readonly chunk: string;
  readonly headers: readonly string[];
  readonly similarity: number;
  readonly fp_source: string;
}

export interface IndexedFile {
  readonly path: string;
  readonly project: string;
  readonly chunks: number;
  readonly size: number;
}

export interface IndexSummary {
  readonly total_sources: number;
  readonly total_chunks: number;
  readonly projects: readonly string[];
  readonly files: readonly IndexedFile[];
}

export interface IndexResult {
  readonly indexed: number;
  readonly skipped: number;
  readonly errors: number;
  readonly total_chunks: number;
}

// ── Input Types ─────────────────────────────────────────────────────

export interface WriteSessionParams {
  readonly session_summary: string;
  readonly session_id?: string;
  readonly domain?: string;
}

export interface RetrieveSessionsParams {
  readonly query: string;
  readonly k?: number;
}

export interface GetContextParams {
  readonly query: string;
  readonly k?: number;
  readonly project?: string;
}

export interface IndexKnowledgeParams {
  readonly source_path: string;
  readonly project?: string;
  readonly force?: boolean;
}

// ── Client Configuration ────────────────────────────────────────────

export interface EngramClientConfig {
  /** Path to Python executable (default: "python3") */
  readonly pythonPath?: string;
  /** Path to the MCP server script (default: auto-detect) */
  readonly serverPath?: string;
  /** Sessions directory override */
  readonly sessionsDir?: string;
  /** Knowledge directory override */
  readonly knowledgeDir?: string;
  /** Project directory for ENGRAM (default: auto-detect) */
  readonly projectDir?: string;
}

// ── Client ──────────────────────────────────────────────────────────

export class EngramClient {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private readonly config: EngramClientConfig;

  constructor(config: EngramClientConfig = {}) {
    this.config = Object.freeze({ ...config });
  }

  /**
   * Connect to the ENGRAM MCP server.
   * Spawns the Python server process via stdio transport.
   */
  async connect(): Promise<void> {
    const pythonPath = this.config.pythonPath ?? "python3";
    const serverPath = this.config.serverPath ?? this.detectServerPath();

    const env: Record<string, string> = {
      ...process.env as Record<string, string>,
    };

    if (this.config.sessionsDir) {
      env["ENGRAM_SESSIONS_DIR"] = this.config.sessionsDir;
    }
    if (this.config.knowledgeDir) {
      env["ENGRAM_KNOWLEDGE_DIR"] = this.config.knowledgeDir;
    }
    if (this.config.projectDir) {
      env["ENGRAM_PROJECT_DIR"] = this.config.projectDir;
    }

    this.transport = new StdioClientTransport({
      command: pythonPath,
      args: [serverPath],
      env,
    });

    this.client = new Client(
      { name: "engram-mcp-client", version: "1.0.0" },
      { capabilities: {} },
    );

    await this.client.connect(this.transport);
  }

  /**
   * Disconnect from the ENGRAM MCP server.
   */
  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
      this.transport = null;
    }
  }

  // ── Session Tools ───────────────────────────────────────────────

  /**
   * Encode the terminal session state and store as a .eng binary.
   *
   * Call at the END of every session. Use this format:
   *   VALIDATED: <confirmed results, accuracy metrics>
   *   CURRENT:   <current file locations, system state>
   *   NEXT:      <prioritised next steps>
   *   OPEN:      <unresolved items, known failures>
   */
  async writeSessionEngram(params: WriteSessionParams): Promise<WriteSessionResult> {
    return this.callTool<WriteSessionResult>("write_session_engram", {
      session_summary: params.session_summary,
      session_id: params.session_id ?? "",
      domain: params.domain ?? "engram",
    });
  }

  /**
   * Get the most recent session's terminal state.
   * Call at the START of every session.
   */
  async getLastSession(): Promise<LastSessionResult> {
    return this.callTool<LastSessionResult>("get_last_session", {});
  }

  /**
   * Semantic search over all stored session memories.
   * Returns the k most similar prior sessions to the query.
   */
  async retrieveRelevantSessions(
    params: RetrieveSessionsParams,
  ): Promise<readonly SessionMatch[]> {
    return this.callTool<readonly SessionMatch[]>(
      "retrieve_relevant_sessions",
      {
        query: params.query,
        k: params.k ?? 3,
      },
    );
  }

  // ── Knowledge Tools ─────────────────────────────────────────────

  /**
   * Semantic search over the ENGRAM knowledge index.
   * Searches indexed markdown files for relevant chunks using HNSW.
   */
  async getRelevantContext(
    params: GetContextParams,
  ): Promise<readonly KnowledgeMatch[]> {
    return this.callTool<readonly KnowledgeMatch[]>(
      "get_relevant_context",
      {
        query: params.query,
        k: params.k ?? 5,
        project: params.project ?? "",
      },
    );
  }

  /**
   * List all indexed knowledge files and their chunk counts.
   */
  async listIndexed(project?: string): Promise<IndexSummary> {
    return this.callTool<IndexSummary>("list_indexed", {
      project: project ?? "",
    });
  }

  /**
   * Index a markdown file or directory into the knowledge index.
   */
  async indexKnowledge(params: IndexKnowledgeParams): Promise<IndexResult> {
    return this.callTool<IndexResult>("index_knowledge", {
      source_path: params.source_path,
      project: params.project ?? "engram",
      force: params.force ?? false,
    });
  }

  // ── Internal ────────────────────────────────────────────────────

  private ensureConnected(): Client {
    if (!this.client) {
      throw new Error(
        "EngramClient not connected. Call await client.connect() first.",
      );
    }
    return this.client;
  }

  private async callTool<T>(name: string, args: Record<string, unknown>): Promise<T> {
    const client = this.ensureConnected();
    const result = await client.callTool({ name, arguments: args });

    const content = result.content;
    if (!Array.isArray(content) || content.length === 0) {
      throw new Error(`Empty response from tool: ${name}`);
    }

    const first = content[0];
    if (first.type !== "text" || typeof first.text !== "string") {
      throw new Error(`Unexpected response type from tool: ${name}`);
    }

    return JSON.parse(first.text) as T;
  }

  private detectServerPath(): string {
    const candidates = [
      // Relative to this package (if installed alongside)
      require.resolve?.("../../mcp/engram_memory.py") ?? "",
      // Global install location
      `${process.env["HOME"]}/.engram/mcp/engram_memory.py`,
      // pip install location
      "engram_memory",
    ].filter(Boolean);

    for (const p of candidates) {
      try {
        require("fs").accessSync(p);
        return p;
      } catch {
        continue;
      }
    }

    // Default — let Python resolve it
    return "mcp/engram_memory.py";
  }
}

// Re-export everything
export default EngramClient;

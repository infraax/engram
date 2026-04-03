
## Session Memory Protocol (ENGRAM MCP)

### Session Start (ALWAYS)
1. Call get_last_session() — inject terminal state as context
2. If task is non-trivial: call retrieve_relevant_sessions(task, k=3)

### Session End (ALWAYS)
Call write_session_engram() with this format:
  VALIDATED: <what was confirmed this session, with metrics>
  CURRENT:   <exact file paths, system state, test count>
  NEXT:      <next priorities, in order>
  OPEN:      <unresolved items, known failures>

### Why This Matters
Each session_summary stored in ~/.engram/sessions/ is fingerprinted
using the ENGRAM f0+f1 protocol and becomes retrievable by semantic
similarity. This is ENGRAM using itself for its own memory.
The session_summary is the terminal state coordinate — the geodesic
destination, not the path. Keep it compressed and precise.

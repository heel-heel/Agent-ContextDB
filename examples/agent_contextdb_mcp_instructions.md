# ContextDB MCP Agent Instructions

Use ContextDB as an advisory context and skill service. Do not execute a
ContextDB recommendation automatically.

1. At the start of a new turn, call `contextdb_prepare_context` with the stable
   Agent `source` and `session_id`. Use its returned `prompt_context` and
   `agent_context` only when relevant to the current task.
2. After each tool execution that actually occurred, call
   `contextdb_record_tool_result` with the observed tool name, command, status,
   output preview, and exit code. A failure can return a matching skill in the
   same MCP response.
3. If ContextDB delivers a recommendation, inspect it. Before using it, call
   `contextdb_record_skill_decision` with `accepted`, `rejected`, or `deferred`.
   Explain a rejection briefly.
4. Execute an accepted action only through the Agent's ordinary tool mechanism
   and approval policy. Afterwards call `contextdb_record_skill_application`
   with the real outcome and the supplied skill references.
5. Never report a tool result or skill application that did not actually happen.

# AI agents for developers: code

This repository contains the runnable `opsworld` incident simulator, its on-call tools and loop, an MCP server, the evaluation suite, and framework versions of the agent. Install dependencies with `uv sync --extra mcp` (or the relevant framework extra), start Ollama with `qwen3:8b`, and run a framework with `uv run --extra pydantic-ai python -m frameworks.pydantic_ai_oncall downstream-tax` or `uv run --extra langgraph python -m frameworks.langgraph_oncall downstream-tax`. The MCP server runs with `uv run --extra mcp python mcp_server.py`; the checks use local fakes and do not call Ollama.

- `opsworld.py` — chapter 1.
- `test_opsworld.py` — chapter 1.
- `oncall_tools.py` — chapter 3.
- `mcp_wire.py` — chapter 4.
- `mcp_server.py` — chapter 4.
- `test_mcp_server.py` — chapter 4.
- `loop_parts.py` — chapter 5.
- `oncall_loop.py` — chapter 5.
- `plan_parts.py` — chapter 6.
- `planning.py` — chapter 6.
- `context_parts.py` — chapter 8.
- `memory_parts.py` — chapter 9.
- `team.py` — chapter 12.
- `evals.py` — chapter 13.
- `suite.py` — chapter 13.
- `cost_parts.py` — chapter 14.
- `frameworks/__init__.py` — chapter 15.
- `frameworks/pydantic_ai_oncall.py` — chapter 15.
- `frameworks/langgraph_oncall.py` — chapter 15.
- `test_frameworks.py` — chapter 15.

---
title: MCP Server | Syda
description: Use Syda directly from Claude Desktop, Cursor, Windsurf, and other AI assistants via the Model Context Protocol.
keywords:
  - MCP
  - Model Context Protocol
  - Claude Desktop
  - Cursor
  - synthetic data MCP
  - AI agent synthetic data
---

# MCP Server

Syda ships a built-in [Model Context Protocol](https://modelcontextprotocol.io) server that lets you generate synthetic data directly from Claude Desktop, Cursor, Windsurf, or any MCP-compatible AI assistant — no Python code required.

**Division of labour:** the AI agent designs the schema; Syda handles LLM generation, FK integrity, codegen optimisation, and cost tracking.

---

## Install

Choose one of three install methods depending on your setup:

### Option A: `uvx` — recommended, no install needed

[`uvx`](https://docs.astral.sh/uv/) runs syda-mcp on demand in an isolated environment. Nothing to install globally, always uses the latest version.

```bash
# Install uv if you don't have it
brew install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then use `uvx` directly in your MCP config — no `pip install` needed.

### Option B: `pipx` — install once, available everywhere

```bash
pipx install "syda[mcp]"
```

`syda-mcp` is then available globally. Verify:

```bash
syda-mcp --help
```

### Option C: pip into a virtual environment

```bash
pip install "syda[mcp]"
```

Use the full path to the venv binary in your MCP config (see below).

---

## Configure your AI assistant

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

**With `uvx` (recommended):**
```json
{
  "mcpServers": {
    "syda": {
      "command": "uvx",
      "args": ["--from", "syda[mcp]", "syda-mcp"]
    }
  }
}
```

**With `pipx`:**
```json
{
  "mcpServers": {
    "syda": {
      "command": "syda-mcp"
    }
  }
}
```

**With a project venv (full path):**
```json
{
  "mcpServers": {
    "syda": {
      "command": "/path/to/your/.venv/bin/syda-mcp",
      "cwd": "/path/to/your/project"
    }
  }
}
```

!!! tip "API keys"
    If your API keys are in a `.env` file in your project, set `"cwd"` to that directory — Syda loads `.env` automatically. Otherwise pass keys explicitly in `"env"`.

Restart Claude Desktop. Syda tools will appear automatically.

### Cursor

Create or edit `.cursor/mcp.json` in your project (or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "syda": {
      "command": "uvx",
      "args": ["--from", "syda[mcp]", "syda-mcp"]
    }
  }
}
```

### Windsurf

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "syda": {
      "command": "uvx",
      "args": ["--from", "syda[mcp]", "syda-mcp"]
    }
  }
}
```

---

## Available tools

The MCP server exposes five tools:

### `generate_from_schema` (primary)

Generate synthetic data from a schema dict. The agent designs the schema; Syda generates the data with full FK integrity, codegen optimisation for large tables, and a cost report.

**Example prompt to Claude:**

> *"Generate a fintech dataset with 500 customers and 2,000 transactions. Customers have name, email, country, and account_balance. Transactions have amount, type (credit/debit), and a status (pending/completed/failed)."*

Claude calls `generate_from_schema` automatically and returns a preview of the generated data plus cost breakdown.

**Schema format:**

```python
{
  "customers": {
    "customer_id": {"type": "integer", "primary_key": True},
    "email":       {"type": "email",   "unique": True},
    "name":        {"type": "string"},
    "country":     {"type": "string",  "enum": ["US","UK","DE","FR","IN"]},
    "balance":     {"type": "float",   "min": 0.0, "max": 100000.0},
    "created_at":  {"type": "date"},
  },
  "transactions": {
    "tx_id":       {"type": "integer", "primary_key": True},
    "customer_id": {
      "type": "foreign_key",
      "references": {"schema": "customers", "field": "customer_id"}
    },
    "amount":      {"type": "float",   "min": 1.0, "max": 5000.0},
    "type":        {"type": "string",  "enum": ["credit", "debit"]},
    "status":      {"type": "string",  "enum": ["pending", "completed", "failed"],
                    "probabilities": [0.1, 0.8, 0.1]},
    "tx_date":     {"type": "date"},
  }
}
```

**Supported column types:** `integer`, `float`, `string`, `text`, `email`, `date`, `boolean`, `foreign_key`

**Key parameters:**
- `sample_sizes` — rows per table e.g. `{"customers": 500, "transactions": 2000}`
- `prompts` — per-table context e.g. `{"customers": "US and EU customers only, realistic names"}`
- `output_dir` — save CSVs to disk (recommended for large tables)
- `provider` — `anthropic`, `openai`, `gemini`, `grok`, `openai_compatible`
- `generation_mode` — `auto` (default), `direct`, `codegen`
- `max_workers` — parallel table generation (default 1)

---

### `validate_schema`

Validate a schema dict without generating data or making any LLM calls. Returns a list of errors and warnings — useful before a large generation run.

**Example prompt:**

> *"Validate this schema before generating: [paste schema]"*

---

### `infer_schema_from_db`

Connect to a live database and infer schemas from its structure. Returns a schema dict ready to pass to `generate_from_schema`.

**Example prompt:**

> *"Infer the schema from my PostgreSQL database at postgresql://localhost/myapp and generate 1,000 rows per table"*

Supported databases: PostgreSQL, MySQL, SQLite (any SQLAlchemy dialect).

---

### `get_providers`

List available LLM providers and which ones have API keys configured in the environment.

**Example prompt:**

> *"Which LLM providers does Syda have configured?"*

---

### `get_run_report`

Cost and token usage are included directly in each `generate_from_schema` response in the `report` field. The `get_run_report` tool explains how to access them.

---

## Example conversation

```
User: Generate a healthcare dataset with patients, encounters, and diagnoses.
      500 patients, 2000 encounters, 3000 diagnoses. Use Claude Haiku.

Claude: I'll design the schema and generate this with Syda.

[calls generate_from_schema with healthcare schema, sample_sizes, Haiku model]

Syda returns:
  ✓ patients:    500 rows
  ✓ encounters: 2,000 rows
  ✓ diagnoses:  3,000 rows
  FK integrity: all checks passed
  Cost: $0.08  |  Time: 42s

Here are 3 sample rows from each table:
[previews]
```

---

## How it works

The `syda-mcp` command runs as a **subprocess** started by your AI assistant — no port, no HTTP server, nothing to keep running. The assistant communicates with it over stdin/stdout using the MCP JSON-RPC protocol and kills the process when done.

```
Claude Desktop
    │ starts subprocess
    ▼
syda-mcp (stdio)
    │ calls syda internally
    ▼
LLM API → generated data → returned to Claude
```

---

## Environment variables

Syda auto-detects your provider from whichever key is set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # → uses Claude (default)
export OPENAI_API_KEY=sk-proj-...     # → uses GPT
export GEMINI_API_KEY=...             # → uses Gemini
export GROK_API_KEY=xai-...           # → uses Grok
```

Or put them in a `.env` file in your working directory — Syda loads it automatically.

---

## See also

- [Generation modes](deep_dive/large_dataset.md) — direct vs codegen for large tables
- [Schema reference](schema_reference/field_types.md) — all column types and constraints
- [Model configuration](deep_dive/model_configuration.md) — provider and model options
- [CLI reference](deep_dive/cli.md) — command-line alternative to MCP

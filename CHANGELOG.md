# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.4.0] - 2026-08-08

### Added
- **MCP server** (`syda-mcp`) — exposes Syda's synthetic data generation to AI agents (Claude Desktop, Cursor, Windsurf) over the Model Context Protocol via stdio. Five tools: `generate_from_schema`, `validate_schema`, `infer_schema_from_db`, `get_providers`, `get_run_report`. Install with `pip install "syda[mcp]"`.
- `examples/mcp/test_provider_matrix.py` — MCP example that forces `provider=` explicitly per run across all supported providers and verifies real FK referential integrity for each; useful as a smoke test after touching provider routing.

### Fixed
- **FK schema format mismatch** — `generate_from_schema`/`validate_schema`/`infer_schema_from_db` documented and accepted a foreign-key shape (`{"foreign_key": {"table", "column"}}`) that the core `SchemaLoader` never recognized, so FK columns silently generated as unrelated integers instead of being sampled from the parent table's real primary keys. Standardized on the core engine's existing `{"type": "foreign_key", "references": {"schema", "field"}}` convention everywhere.
- **`api_key` override silently dropped for `gemini`/`azureopenai`** in `generate_from_schema` — `gemini` now forwards `gemini_api_key`; `azureopenai` has no such constructor kwarg, so its override is routed into `extra_kwargs["api_key"]`, the only place pydantic-ai's `AzureProvider` reads it from.
- **stdout corruption of the MCP stdio transport** — `syda/generate.py` and `db_schema_loader.py` have plain `print()` calls meant for CLI usage; over stdio, stdout IS the JSON-RPC wire, so a stray print() during generation could interleave garbage into the protocol stream and hang the client. Generation and DB-introspection calls are now wrapped with `contextlib.redirect_stdout` inside the MCP server.
- **Gemini generation broken end-to-end** — pydantic-ai's deprecated `GeminiModel` expects an httpx-style client with `.stream()`, but the current `GoogleProvider` returns a `google.genai.Client` without that method (`AttributeError: 'Client' object has no attribute 'stream'`). Switched to `GoogleModel`, pydantic-ai's maintained replacement.
- **Unescaped DB credentials** in `infer_schema_from_db`'s `DB_USER`/`DB_PASSWORD` connection-string building — now URL-encoded via `urllib.parse.quote_plus` so special characters (`@`, `:`, `/`, `%`) don't corrupt the connection URL.
- **Credential leakage in error messages** — `_tool_error` now redacts `user:password@` from exception text, since DB drivers often embed the full DSN (including cleartext password) in connection-failure errors.

### Changed
- Version bumped to `0.4.0`
- Updated stale/retired model defaults across docs and examples: `gemini-1.5-*`/`gemini-2.0-*`/`gemini-2.5-*` (confirmed retired by Google's API) → `gemini-flash-latest`/`gemini-pro-latest`/`gemini-flash-lite-latest`; older Claude snapshots → the current Claude 5 family (`claude-haiku-4-5-20251001`, `claude-sonnet-5`, `claude-opus-5`); `gpt-4o`/`gpt-4-turbo`/`o3-*` → the current `gpt-5` family.

## [0.3.0] - 2026-06-27

### Added
- **Parallel table generation** — `ModelConfig(max_workers=N)` runs tables within the same DAG level concurrently using `ThreadPoolExecutor`. Default `max_workers=1` preserves the original sequential behaviour. CLI flag: `--workers N` on `syda generate` and `syda db generate`.
- **Shared global backoff signal** — when any thread hits a 429 rate-limit, `_set_global_backoff()` sets a shared deadline and all threads pause before the next LLM call, preventing thundering-herd retries in parallel mode.
- `DependencyHandler.compute_parallel_levels()` — groups independent tables into BFS levels so tables within the same level can run concurrently.
- FK registration is now protected by `_fk_lock` to eliminate race conditions in concurrent table generation.

### Fixed
- `'RunUsage' object has no attribute 'request_tokens'` crash with **pydantic-ai ≥ 2.0**. Token attributes `request_tokens` / `response_tokens` were renamed to `input_tokens` / `output_tokens` in pydantic-ai 2.0; syda now uses `getattr` with the new names and works correctly with both 1.x and 2.x.

### Changed
- Version bumped to `0.3.0`
- Pre-release test script updated with parallel-generation checks (`DependencyHandler`, `compute_parallel_levels`, `ModelConfig.max_workers`, `--workers` CLI flags).

## [0.2.0] - 2026-06-08

### Added
- **Code-gen cache** (`CodegenCache`, `compute_schema_hash`) — generated Python functions for simple columns are persisted as human-editable `.py` files under `output_dir/.syda_cache/`. Cache key = `hash(schema_content + prompt)`. On cache HIT the LLM analysis call is skipped entirely.
- **`force_llm` column flag** — set `force_llm: true` on any schema column to always generate it via LLM in code-gen mode, even if a cached function exists. Useful for narrative, taglines, and context-sensitive columns.
- **`RunReport` / `TableReport` / `ColumnReport`** (`syda/run_report.py`) — per-column observability: strategy (`fk_sampler` / `codegen_simple` / `codegen_semantic` / `direct_llm`), token counts, cost, cache hit/miss. Accessible via `generator.last_report`. HTML report auto-saved to `output_dir/run_report_<timestamp>.html`.
- **Memory-bounded multi-table generation** — when `output_dir` is set, each table is flushed to CSV immediately after generation; only FK columns are kept in RAM for child-table generation.
- `force_llm` example in `examples/force_llm/` — 600-row product catalog demonstrating mixed codegen / force_llm columns.

### Changed
- Version bumped to `0.2.0`

## [0.1.0] - 2026-05-26

### Added
- `syda` CLI (`syda/cli.py`) — full command-line interface for synthetic data generation
  - `syda version` — print installed version
  - `syda validate --schema PATH` — validate schema file(s) without making any LLM calls
  - `syda generate --schema PATH` — generate synthetic data from a YAML/JSON schema file or directory
    - `--rows N` — rows per table (default 10)
    - `--output FILE` — single output file; format inferred from `.csv`/`.json` extension
    - `--output-dir DIR` — directory for multi-table output
    - `--format csv|json` — explicit format override
    - `--provider` — LLM provider; auto-detected from env vars when omitted
    - `--model`, `--api-key`, `--base-url`, `--prompt`, `--temperature`
  - `syda db infer --db-url URL --output-dir DIR` — infer schemas from a live database and save as YAML/JSON
    - `--tables TABLE,...` — limit to specific tables
    - `--format yaml|json`
  - `syda db generate --db-url URL` — infer schemas, generate data, optionally write back to the database
    - `--write-back` / `--if-exists append|replace|fail` — insert generated rows into the database
    - All `syda generate` LLM options supported
- CLI example in `examples/cli/` — bash demo script covering all 10 CLI workflows with healthcare schemas
- `click>=8.0.0` added as a runtime dependency
- `syda = syda.cli:main` entry point registered in `pyproject.toml`
- 26 CLI unit tests in `tests/test_cli.py`

### Changed
- Version bumped to `0.1.0` (first proper semantic version — CLI is a minor feature release)

## [0.0.6] - 2026-05-11

### Added
- `openai_compatible` provider in `ModelConfig` — connect to any OpenAI-compatible API (Ollama, Groq, Together AI, Fireworks, DeepSeek, Mistral, LM Studio, vLLM, and more) by passing a `base_url` in `extra_kwargs`
- `response_mode` option in `extra_kwargs` for `openai_compatible` provider — controls how instructor parses the model response: `"markdown"` (default, strips markdown fences), `"tools"` (tool call mode for models that support it), `"json"` (clean JSON content)
- `api_key` in `extra_kwargs` for `openai_compatible` provider — falls back to `OPENAI_API_KEY` env var, then `"none"` for providers that don't require a key (e.g. Ollama)

### Changed
- Default model updated from `claude-3-5-haiku-20241022` to `claude-haiku-4-5-20251001`
- All examples updated to current Claude model names (`claude-haiku-4-5-20251001`, `claude-sonnet-4-5`, `claude-opus-4-5`)
- Version bumped to `0.0.6`

## [0.0.5] - 2026-05-03

### Added
- `DatabaseSchemaLoader` class in `syda/db_schema_loader.py` for connecting directly to relational databases
  - `load_schemas()` — infers table schemas (columns, types, primary keys, foreign keys) as Python dicts
  - `save_schemas()` — writes one YAML or JSON schema file per table to disk
  - `write_to_database()` — inserts generated DataFrames back into the database in FK-safe topological order
- Database integration examples in `examples/database_integration/`:
  - `example_load_schemas.py` — in-memory schema workflow (SQLite)
  - `example_save_schemas.py` — file-based schema workflow (SQLite)
  - `example_postgres.py` — full PostgreSQL end-to-end example
  - `healthcare_demo.db` — SQLite demo database with patient, provider, diagnosis, claim, payment, and adjudication tables
  - Pre-generated schema YAML files and sample output CSVs
- `tests/test_db_schema.py` — comprehensive test suite for `DatabaseSchemaLoader`
- `docs/deep_dive/database_integration.md` — full documentation covering supported databases, Option A/B workflows, type mapping, FK-safe insertion order, and API reference
- `scripts/pre_release_test.sh` — automated pre-release validation script
- `CLAUDE.md` — developer guidance file for Claude Code
- PostgreSQL (`psycopg2-binary`) and MySQL (`pymysql`) driver dependencies in `requirements.txt`
- `DatabaseSchemaLoader` exported from `syda/__init__.py`

### Changed
- Version bumped to `0.0.5`
- `README.md` updated with database integration overview and usage examples
- Database Integration page added to MkDocs navigation (`deep_dive/database_integration`)

## [0.0.3] - 2025-09-21

### Added
- Azure OpenAI provider support for enterprise deployments
- Advanced configuration with `extra_kwargs` parameter for all providers
- AI gateway integration support (LiteLLM, Portkey, Kong, and custom gateways)
- Comprehensive Azure OpenAI documentation and examples
- Enhanced model configuration guide with `extra_kwargs` reference
- Support for custom endpoints, authentication headers, and timeouts
- Enterprise-grade features for production deployments

### Changed
- Development status upgraded from Beta to Production/Stable
- Enhanced documentation with AI gateway integration examples
- Improved error handling and troubleshooting guidance
- Updated model configuration documentation with provider-specific examples

### Fixed
- Enhanced provider-specific parameter handling
- Better error messages for configuration issues


## [0.0.2] - 2025-08-23

### Added
- Support for Google Gemini Models

### Changed
- Documentation Fixes


## [0.0.1] - 2025-08-11

### Added
- Modern packaging with pyproject.toml
- Support for multiple AI providers (OpenAI, Anthropic Claude)
- Comprehensive schema formats (SQLAlchemy, YAML, JSON, Dict)
- Foreign key relationship handling with referential integrity
- Unstructured document generation with templates
- Custom generators for domain-specific data
- Multi-provider AI integration with consistent interface
- Automatic dependency resolution via topological sorting
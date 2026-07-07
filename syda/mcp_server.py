"""
Syda MCP server.

Exposes Syda's synthetic data generation capabilities to AI agents
(Claude Desktop, Cursor, Windsurf, etc.) over the Model Context Protocol.

Runs over stdio. Five tools:
  - generate_from_schema  (primary)
  - validate_schema
  - infer_schema_from_db
  - get_providers
  - get_run_report

Install:
    pip install "syda[mcp]"

Configure in Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "syda": { "command": "syda-mcp" }
      }
    }

Environment variables (or .env in cwd):
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / GROK_API_KEY
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        'The MCP extra is not installed. Run: pip install "syda[mcp]"'
    ) from exc

from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(
    "syda",
    instructions=(
        "Syda generates realistic multi-table synthetic data with full referential integrity "
        "using LLMs. Use it when a user needs test data, dev/staging database seeds, demo "
        "datasets, ML training data, or privacy-safe copies of production schemas.\n\n"
        "Division of labour: YOU design the schema (tables, columns, relationships, prompts); "
        "Syda handles LLM generation, FK integrity, codegen optimisation for large tables, "
        "and cost tracking. The primary tool is generate_from_schema — pass a schema dict "
        "describing each table's columns and the tool returns generated data plus a full "
        "cost/token report.\n\n"
        "Supported providers: anthropic (Claude), openai (GPT), gemini, grok, azureopenai, "
        "openai_compatible (Ollama, Groq, etc.). Provider is auto-detected from env vars if "
        "not specified.\n\n"
        "For large tables (>500 rows), Syda automatically switches to code-gen mode: the LLM "
        "writes Python generator functions for simple columns and only calls the LLM at "
        "runtime for semantic/narrative columns — dramatically fewer API calls at scale."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_error(exc: Exception, suggestion: str = "") -> Dict[str, Any]:
    return {
        "ok": False,
        "error": type(exc).__name__,
        "message": str(exc),
        "suggestion": suggestion,
    }


def _df_preview(df, n: int = 5) -> List[Dict[str, Any]]:
    return json.loads(
        df.head(n).to_json(orient="records", date_format="iso", default_handler=str)
    )


def _build_generator(provider: Optional[str], model: Optional[str],
                     api_key: Optional[str], temperature: float,
                     max_tokens: int, generation_mode: str,
                     batch_size: Optional[int], max_workers: int,
                     extra_kwargs: Optional[Dict[str, Any]] = None):
    """Build a SyntheticDataGenerator from MCP tool parameters."""
    from syda import SyntheticDataGenerator, ModelConfig

    if not provider:
        # Auto-detect from env — same keys as syda/llm.py
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            model = model or "claude-haiku-4-5-20251001"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            model = model or "gpt-4o-mini"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
            model = model or "gemini-1.5-flash"
        elif os.getenv("GROK_API_KEY"):
            provider = "grok"
            model = model or "grok-4.3"
        elif os.getenv("AZURE_OPENAI_API_KEY"):
            provider = "azureopenai"
        else:
            raise ValueError(
                "No API key found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
                "GEMINI_API_KEY, GROK_API_KEY, or AZURE_OPENAI_API_KEY."
            )

    mc_kwargs: Dict[str, Any] = {
        "provider": provider,
        "model_name": model or "claude-haiku-4-5-20251001",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "generation_mode": generation_mode,
        "max_workers": max_workers,
    }
    if batch_size:
        mc_kwargs["batch_size"] = batch_size

    # Build extra_kwargs: start with provider defaults, then merge user overrides.
    # Provider defaults ensure required fields are always present (e.g. Grok base_url,
    # openai_compatible base_url) while letting users override or extend them.
    merged_extra: Dict[str, Any] = {}
    if provider == "grok":
        merged_extra = {"base_url": "https://api.x.ai/v1"}
    if extra_kwargs:
        merged_extra.update(extra_kwargs)
    if merged_extra:
        mc_kwargs["extra_kwargs"] = merged_extra

    # Validate openai_compatible always has base_url
    if provider == "openai_compatible" and "base_url" not in merged_extra:
        raise ValueError(
            "openai_compatible requires 'base_url' in extra_kwargs. "
            "Example: extra_kwargs={'base_url': 'http://localhost:11434/v1', 'api_key': 'ollama'}"
        )

    # Azure requires azure_endpoint
    if provider == "azureopenai" and "azure_endpoint" not in merged_extra:
        raise ValueError(
            "azureopenai requires 'azure_endpoint' in extra_kwargs. "
            "Example: extra_kwargs={'azure_endpoint': 'https://your-resource.openai.azure.com/', "
            "'api_version': '2024-02-01'}"
        )

    if api_key:
        if provider == "anthropic":
            return SyntheticDataGenerator(
                model_config=ModelConfig(**mc_kwargs),
                anthropic_api_key=api_key,
            )
        elif provider in ("openai", "openai_compatible"):
            return SyntheticDataGenerator(
                model_config=ModelConfig(**mc_kwargs),
                openai_api_key=api_key,
            )
        elif provider == "grok":
            return SyntheticDataGenerator(
                model_config=ModelConfig(**mc_kwargs),
                grok_api_key=api_key,
            )

    return SyntheticDataGenerator(model_config=ModelConfig(**mc_kwargs))


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def generate_from_schema(
    schema: Dict[str, Any],
    sample_sizes: Optional[Dict[str, int]] = None,
    default_sample_size: int = 10,
    prompts: Optional[Dict[str, str]] = None,
    output_dir: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    temperature: float = 0.8,
    max_tokens: int = 8192,
    generation_mode: str = "auto",
    batch_size: Optional[int] = None,
    max_workers: int = 1,
    preview_rows: int = 3,
) -> Dict[str, Any]:
    """Generate synthetic data from a schema dict and return previews + a cost report.

    This is the primary Syda tool. You design the schema; Syda handles LLM
    generation, FK ordering, and cost tracking.

    Schema format — each key is a table name, value is a column dict:
    {
      "customers": {
        "customer_id": {"type": "integer", "primary_key": true},
        "email":       {"type": "email",   "unique": true},
        "plan":        {"type": "string",  "enum": ["free","pro","enterprise"]}
      },
      "orders": {
        "order_id":    {"type": "integer", "primary_key": true},
        "customer_id": {"type": "integer", "foreign_key": {"table": "customers", "column": "customer_id"}},
        "amount":      {"type": "float",   "min": 5.0, "max": 500.0}
      }
    }

    Supported column types: integer, float, string, text, email, date, boolean, foreign_key.
    Add "enum": [...] on any string column to declare fixed values.
    Add "primary_key": true, "unique": true, "not_null": true as constraints.

    Args:
        schema:               Table → column definitions dict (see above).
        sample_sizes:         Rows per table e.g. {"customers": 100, "orders": 500}.
                              Omit to use default_sample_size for all tables.
        default_sample_size:  Fallback row count when not in sample_sizes (default 10).
        prompts:              Per-table generation prompts e.g. {"customers": "US customers only"}.
        output_dir:           Save CSVs here. When set, large tables stream to disk (RAM-bounded).
                              Pass null to return data in-memory only.
        provider:             LLM provider: anthropic, openai, gemini, grok, azureopenai,
                              openai_compatible. Auto-detected from env vars if omitted.
        model:                Model name e.g. "claude-haiku-4-5-20251001". Provider default if omitted.
        api_key:              API key override. Uses env var if omitted.
        extra_kwargs:         Provider-specific parameters passed to ModelConfig.extra_kwargs.
                              Required for some providers:
                                azureopenai:       {"azure_endpoint": "https://...", "api_version": "2024-02-01"}
                                openai_compatible: {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}
                              Optional for others:
                                grok:              {"base_url": "https://api.x.ai/v1"}  (already defaulted)
                                any provider:      {"response_mode": "tools"}  (custom response parsing)
        temperature:          Sampling temperature 0.0–1.0 (default 0.8).
        max_tokens:           Max tokens per LLM call (default 8192).
        generation_mode:      "auto" (default), "direct", or "codegen".
                              auto = direct for ≤500 rows, codegen for >500.
        batch_size:           Max rows per LLM call in direct mode. Auto-selected if omitted.
        max_workers:          Tables to generate concurrently (default 1 = sequential).
        preview_rows:         Rows to include in the preview response (default 3).

    Returns:
        ok, tables (name → preview rows), row_counts, fk_integrity, report (cost/tokens/time),
        output_files (when output_dir is set).
    """
    try:
        import threading
        from syda import SyntheticDataGenerator

        generator = _build_generator(
            provider, model, api_key, temperature, max_tokens,
            generation_mode, batch_size, max_workers,
            extra_kwargs=extra_kwargs,
        )

        # syda uses agent.run_sync() which fails when called inside an already-
        # running asyncio event loop (the MCP server's async context). Running in
        # a dedicated thread gives pydantic-ai a clean loop to work with.
        _result: list = [None]
        _exc:    list = [None]

        def _run():
            try:
                _result[0] = generator.generate_for_schemas(
                    schemas=schema,
                    sample_sizes=sample_sizes or {},
                    default_sample_size=default_sample_size,
                    prompts=prompts or {},
                    output_dir=output_dir,
                )
            except Exception as e:
                _exc[0] = e

        t = threading.Thread(target=_run)
        t.start()
        t.join()
        if _exc[0]:
            raise _exc[0]
        results = _result[0]

        report = generator.last_report
        tables_preview = {}
        row_counts = {}

        for name, df in results.items():
            row_counts[name] = report.tables[name].row_count if (
                report and name in report.tables
            ) else len(df)
            # When output_dir is set, df may be slim (FK cols only) — read CSV for preview
            if output_dir and len(df) < row_counts[name]:
                import pandas as pd
                csv = os.path.join(output_dir, f"{name.lower()}.csv")
                if os.path.exists(csv):
                    df = pd.read_csv(csv)
            tables_preview[name] = _df_preview(df, preview_rows)

        report_summary = {}
        if report:
            report_summary = {
                "total_rows":         sum(t.row_count for t in report.tables.values()),
                "total_llm_calls":    sum(t.llm_calls for t in report.tables.values()),
                "total_input_tokens": sum(t.input_tokens for t in report.tables.values()),
                "total_output_tokens":sum(t.output_tokens for t in report.tables.values()),
                "estimated_cost_usd": round(report.estimated_cost_usd, 4),
                "total_time_s":       round(report.total_duration_s, 1),
                "model":              f"{generator.model_config.provider}/{generator.model_config.model_name}",
                "per_table": {
                    name: {
                        "rows":          t.row_count,
                        "mode":          t.mode,
                        "llm_calls":     t.llm_calls,
                        "input_tokens":  t.input_tokens,
                        "output_tokens": t.output_tokens,
                        "cost_usd":      round(t.cost_usd, 4),
                        "duration_s":    round(t.duration_s, 1),
                    }
                    for name, t in report.tables.items()
                },
            }

        output_files = {}
        if output_dir:
            for name in results:
                csv = os.path.join(output_dir, f"{name.lower()}.csv")
                if os.path.exists(csv):
                    output_files[name] = csv

        return {
            "ok": True,
            "tables": tables_preview,
            "row_counts": row_counts,
            "report": report_summary,
            "output_files": output_files,
            "message": (
                f"Generated {sum(row_counts.values()):,} rows across "
                f"{len(row_counts)} table(s). "
                + (f"Estimated cost: ${report_summary.get('estimated_cost_usd', 0):.4f}. " if report_summary else "")
                + (f"Saved to: {output_dir}" if output_dir else "Data returned in-memory.")
            ),
        }

    except Exception as exc:
        return _tool_error(
            exc,
            "Check your schema format and API key. Call get_providers() to see "
            "which providers are configured. For FK columns use: "
            '{"type": "integer", "foreign_key": {"table": "parent", "column": "id"}}',
        )


@mcp.tool()
def validate_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a schema dict without generating any data or making LLM calls.

    Use this before generate_from_schema to catch issues early:
    missing primary keys, invalid FK references, unsupported column types,
    circular dependencies.

    Args:
        schema: Same format as generate_from_schema.

    Returns:
        ok, tables (summary of each table), relationships (detected FKs),
        warnings, errors.
    """
    try:
        from syda.schema_loader import SchemaLoader
        from syda.dependency_handler import DependencyHandler, ForeignKeyHandler
        import networkx as nx

        loader = SchemaLoader()
        errors = []
        warnings_list = []
        tables_summary = {}
        all_fks = {}

        supported_types = {
            "integer", "int", "float", "double", "decimal", "string", "text",
            "varchar", "email", "date", "datetime", "boolean", "bool",
            "foreign_key", "fk", "number",
        }

        for table_name, columns in schema.items():
            if not isinstance(columns, dict):
                errors.append(f"{table_name}: columns must be a dict")
                continue

            col_summary = []
            has_pk = False
            table_fks = {}

            for col_name, col_def in columns.items():
                if col_name.startswith("_"):
                    continue
                if not isinstance(col_def, dict):
                    errors.append(f"{table_name}.{col_name}: column definition must be a dict")
                    continue

                col_type = col_def.get("type", "string").lower()
                if col_type not in supported_types:
                    warnings_list.append(
                        f"{table_name}.{col_name}: unknown type '{col_type}' — will be treated as text"
                    )

                if col_def.get("primary_key"):
                    has_pk = True

                fk = col_def.get("foreign_key")
                if fk or col_type in ("foreign_key", "fk"):
                    if isinstance(fk, dict):
                        parent_table = fk.get("table")
                        parent_col = fk.get("column")
                        if parent_table and parent_col:
                            table_fks[col_name] = (parent_table, parent_col)
                        else:
                            errors.append(
                                f"{table_name}.{col_name}: foreign_key must have 'table' and 'column'"
                            )

                col_summary.append({
                    "name": col_name,
                    "type": col_type,
                    "primary_key": bool(col_def.get("primary_key")),
                    "unique": bool(col_def.get("unique")),
                    "nullable": not bool(col_def.get("not_null")),
                    "has_enum": bool(col_def.get("enum")),
                    "is_fk": bool(fk or col_type in ("foreign_key", "fk")),
                })

            if not has_pk:
                warnings_list.append(f"{table_name}: no primary_key column declared")

            tables_summary[table_name] = {
                "columns": len(col_summary),
                "primary_key": has_pk,
                "foreign_keys": list(table_fks.keys()),
                "column_details": col_summary,
            }
            all_fks[table_name] = table_fks

        # Validate FK references point to existing tables/columns
        relationships = []
        for child_table, fks in all_fks.items():
            for col, (parent_table, parent_col) in fks.items():
                if parent_table not in schema:
                    errors.append(
                        f"{child_table}.{col}: references unknown table '{parent_table}'"
                    )
                else:
                    relationships.append({
                        "child": f"{child_table}.{col}",
                        "parent": f"{parent_table}.{parent_col}",
                    })

        # Check for cycles
        if not errors:
            try:
                g = nx.DiGraph()
                for table in schema:
                    g.add_node(table)
                for child, fks in all_fks.items():
                    for _, (parent, _) in fks.items():
                        if parent in schema:
                            g.add_edge(parent, child)
                if not nx.is_directed_acyclic_graph(g):
                    errors.append("Circular FK dependency detected — schema has a cycle")
            except Exception:
                pass

        return {
            "ok": len(errors) == 0,
            "tables": tables_summary,
            "relationships": relationships,
            "warnings": warnings_list,
            "errors": errors,
            "message": (
                f"Schema valid — {len(schema)} table(s), {len(relationships)} relationship(s)."
                if not errors else
                f"{len(errors)} error(s) found. Fix them before generating."
            ),
        }

    except Exception as exc:
        return _tool_error(exc, "Ensure schema is a valid dict of table definitions.")


@mcp.tool()
def infer_schema_from_db(
    db_url: Optional[str] = None,
    tables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Connect to a database and infer schemas from its structure.

    Returns a schema dict ready to pass directly to generate_from_schema,
    plus a list of detected FK relationships. No data is read — only schema
    metadata (column names, types, PKs, FKs via INFORMATION_SCHEMA queries).

    Credentials can be supplied four ways (in priority order):
      1. db_url parameter — embed directly: "postgresql://user:pass@host/db"
      2. SYDA_DB_URL env var — set once in MCP config, never type in chat
      3. DATABASE_URL env var — standard 12-factor convention
      4. DB_HOST + DB_NAME (+ DB_USER, DB_PASSWORD, DB_PORT) — syda's
         existing per-component convention used in the CLI examples

    Supported databases: PostgreSQL, MySQL, SQLite (any SQLAlchemy dialect).

    Args:
        db_url: SQLAlchemy connection URL. Falls back to SYDA_DB_URL or
                DATABASE_URL env var if not provided.
                Examples:
                  "postgresql://user:pass@localhost:5432/mydb"
                  "mysql+pymysql://user:pass@localhost/mydb"
                  "sqlite:///path/to/database.db"
        tables: Specific table names to infer (default: all tables).

    Returns:
        ok, schema (ready for generate_from_schema), tables (list), relationships.
    """
    try:
        from syda import DatabaseSchemaLoader

        # Credential resolution (priority order):
        #   1. db_url parameter (explicit)
        #   2. SYDA_DB_URL env var (MCP-specific single URL)
        #   3. DATABASE_URL env var (12-factor convention)
        #   4. DB_USER / DB_PASSWORD / DB_HOST / DB_PORT / DB_NAME
        #      (syda's existing per-component convention used in examples)
        resolved_url = db_url or os.getenv("SYDA_DB_URL") or os.getenv("DATABASE_URL")

        if not resolved_url:
            db_user = os.getenv("DB_USER")
            db_pass = os.getenv("DB_PASSWORD")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT", "5432")
            db_name = os.getenv("DB_NAME")
            if db_host and db_name:
                auth = f"{db_user}:{db_pass}@" if db_user else ""
                resolved_url = f"postgresql+psycopg2://{auth}{db_host}:{db_port}/{db_name}"

        if not resolved_url:
            return _tool_error(
                ValueError("No database URL provided"),
                "Pass db_url directly, or set one of: SYDA_DB_URL, DATABASE_URL, "
                "or DB_HOST + DB_NAME (+ DB_USER, DB_PASSWORD, DB_PORT) in the "
                "MCP server environment config.",
            )

        loader = DatabaseSchemaLoader(resolved_url)
        schemas = loader.load_schemas(table_names=tables)

        # Summarise relationships
        relationships = []
        for tbl_name, tbl_schema in schemas.items():
            for col_name, col_def in tbl_schema.items():
                if col_name.startswith("_"):
                    continue
                if isinstance(col_def, dict) and col_def.get("foreign_key"):
                    fk = col_def["foreign_key"]
                    if isinstance(fk, dict):
                        relationships.append({
                            "child":  f"{tbl_name}.{col_name}",
                            "parent": f"{fk.get('table')}.{fk.get('column')}",
                        })

        return {
            "ok": True,
            "schema": schemas,
            "tables": list(schemas.keys()),
            "relationships": relationships,
            "message": (
                f"Inferred {len(schemas)} table(s) from {resolved_url.split('@')[-1]}. "
                f"Pass the 'schema' field directly to generate_from_schema."
            ),
        }

    except Exception as exc:
        return _tool_error(
            exc,
            "Check your db_url format. Examples: "
            "'postgresql://user:pass@localhost/mydb', "
            "'sqlite:///path/to/db.sqlite', "
            "'mysql+pymysql://user:pass@localhost/mydb'",
        )


@mcp.tool()
def get_providers() -> Dict[str, Any]:
    """List available LLM providers and their configuration status.

    Shows which providers have API keys configured in the environment,
    their recommended models, and approximate pricing.

    Returns:
        ok, providers list with configured/model/notes fields.
    """
    providers = [
        {
            "provider":    "anthropic",
            "configured":  bool(os.getenv("ANTHROPIC_API_KEY")),
            "env_var":     "ANTHROPIC_API_KEY",
            "recommended_model": "claude-haiku-4-5-20251001",
            "fast_model":  "claude-haiku-4-5-20251001",
            "quality_model": "claude-sonnet-4-6",
            "notes": "Best semantic quality. Haiku is fast/cheap; Sonnet for narrative columns.",
        },
        {
            "provider":    "openai",
            "configured":  bool(os.getenv("OPENAI_API_KEY")),
            "env_var":     "OPENAI_API_KEY",
            "recommended_model": "gpt-4o-mini",
            "fast_model":  "gpt-4o-mini",
            "quality_model": "gpt-4o",
            "notes": "Reliable. gpt-4o-mini is cost-effective for most schemas.",
        },
        {
            "provider":    "gemini",
            "configured":  bool(os.getenv("GEMINI_API_KEY")),
            "env_var":     "GEMINI_API_KEY",
            "recommended_model": "gemini-1.5-flash",
            "fast_model":  "gemini-1.5-flash",
            "quality_model": "gemini-1.5-pro",
            "notes": "Google Gemini models. Flash is fast and free-tier friendly.",
        },
        {
            "provider":    "grok",
            "configured":  bool(os.getenv("GROK_API_KEY")),
            "env_var":     "GROK_API_KEY",
            "recommended_model": "grok-4.3",
            "fast_model":  "grok-4.3",
            "quality_model": "grok-4.3",
            "notes": "xAI Grok. 2.5x cheaper than Sonnet. Best for cost-sensitive large datasets.",
        },
        {
            "provider":    "azureopenai",
            "configured":  bool(os.getenv("AZURE_OPENAI_API_KEY")),
            "env_var":     "AZURE_OPENAI_API_KEY",
            "recommended_model": "your deployment name",
            "notes": "Azure OpenAI. Also requires azure_endpoint and api_version in extra_kwargs.",
        },
        {
            "provider":    "openai_compatible",
            "configured":  bool(os.getenv("OPENAI_COMPATIBLE_BASE_URL")),
            "env_var":     "OPENAI_COMPATIBLE_BASE_URL + OPENAI_COMPATIBLE_MODEL",
            "recommended_model": "set via OPENAI_COMPATIBLE_MODEL",
            "notes": "Any OpenAI-compatible API: Ollama (local), Groq, Together AI, etc.",
        },
    ]

    configured = [p["provider"] for p in providers if p["configured"]]
    return {
        "ok": True,
        "providers": providers,
        "configured": configured,
        "message": (
            f"{len(configured)} provider(s) configured: {', '.join(configured)}."
            if configured else
            "No providers configured. Set at least one API key env var."
        ),
    }


@mcp.tool()
def get_run_report(format: str = "summary") -> Dict[str, Any]:
    """Return the run report from the most recent generate_from_schema call.

    Includes per-table row counts, LLM call counts, token usage, cost,
    and generation strategy per column (direct_llm / codegen_simple /
    codegen_semantic / fk_sampler).

    Args:
        format: "summary" (default) for totals only, "full" for per-column detail.

    Returns:
        ok, report with cost/token/time breakdown.

    Note: The report is stored per-server-session. Call immediately after
    generate_from_schema while the session is live.
    """
    # The generator instance is not persisted across tool calls in stdio mode.
    # This tool is most useful when called in the same session via the SDK.
    return {
        "ok": False,
        "message": (
            "Run reports are attached to each generate_from_schema response "
            "in the 'report' field. Check the last generate_from_schema "
            "call's response for cost and token details."
        ),
        "hint": (
            "The 'report' field in generate_from_schema responses contains: "
            "total_rows, total_llm_calls, total_input_tokens, total_output_tokens, "
            "estimated_cost_usd, total_time_s, and per_table breakdown."
        ),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

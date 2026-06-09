# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Commands

```bash
# Tests
pytest                          # all tests
pytest --cov=syda              # with coverage (target >70%)
pytest tests/test_generate.py  # single file

# Linting & formatting
black .
isort .
flake8 .
mypy .
pre-commit run --all-files

# Docs — local preview
mkdocs serve

# Docs — publish to production (python.syda.ai)
# gh-pages branch is the source for python.syda.ai (GitHub Pages custom domain)
mkdocs gh-deploy --force
# This builds the site and force-pushes it to the gh-pages branch.
# GitHub Pages then serves it at python.syda.ai automatically.

# Docs — feature branch preview (Railway)
# Railway (project 8773d9b5-c04f-4c80-8f78-dbe92cfe5ecc, service syda) hosts
# a Docker-based preview of the docs for feature branches only.
# Deploy with: railway up --detach --service syda
# URL: https://syda-production.up.railway.app
# DO NOT deploy feature branch docs to gh-pages — that would overwrite production.

# schema_inference tests (separate pytest.ini)
cd schema_inference && python -m pytest test_modules.py -v
```

**DCO required**: all commits must be signed with `git commit -s`.

## Architecture

`SyntheticDataGenerator` (`syda/generate.py`) is the central class that orchestrates the full pipeline:

1. **Schema parsing** — `SchemaLoader` (`schema_loader.py`) accepts Python dicts, JSON/YAML files, or SQLAlchemy ORM models and normalizes them into a common format including foreign keys, metadata, and template fields.

2. **Dependency resolution** — `ForeignKeyHandler` (`dependency_handler.py`) uses networkx to build a DAG of table dependencies and returns a topologically sorted order so parent tables are generated before children, maintaining referential integrity.

3. **LLM generation** — `LLMClient` (`llm.py`) wraps `pydantic-ai` to provide structured (Pydantic model) output from any supported provider: OpenAI, Anthropic, Gemini, Azure OpenAI, or Grok. Provider selection and parameters are configured via `ModelConfig` (`schemas.py`). All calls use `agent.run_sync()` — syda is fully synchronous.

4. **Generation modes** — Two modes selected automatically by `generation_mode` in `ModelConfig`:
   - `direct` (≤500 rows or forced): chunked LLM calls, `batch_size` rows per call, with exponential-backoff retry and short-count top-up retry.
   - `codegen` (>500 rows or forced): LLM makes one analysis call classifying each column as `simple` (writes a Python generator function) or `semantic` (requires LLM at runtime). Simple columns run locally; only semantic columns call the LLM. Dramatically fewer API calls at scale.
   - `auto` (default): selects `direct` for ≤500 rows, `codegen` for >500.

5. **Code-gen cache** — `CodegenCache` (`codegen_cache.py`) persists generated Python functions as human-editable `.py` files under `output_dir/.syda_cache/`. Cache key = `hash(schema_file_content + user_prompt)` — env-agnostic. On cache HIT the LLM analysis call is skipped entirely. Stale files (from schema changes) are deleted on write. Columns with no `generate_*` function in the cache file are treated as semantic automatically — users can promote a column from codegen to semantic by deleting its function.

6. **`force_llm` column flag** — Setting `force_llm: true` on a column in the schema YAML forces that column to always be LLM-generated, even in codegen mode and even if a cached function exists for it. Useful for narrative, creative, or context-sensitive columns.

7. **Custom generators** — `GeneratorManager` (`custom_generators.py`) maintains a registry of user-supplied generator functions keyed by data type or column name. FK columns are automatically registered to sample from parent table values.

8. **Unstructured / template data** — `TemplateProcessor` (`templates.py`) finds `{{ field_name }}` placeholders in documents and fills them with generated values. `UnstructuredDataProcessor` (`unstructured.py`) reads source documents (PDF, DOCX, Excel, HTML, text, images via OCR).

9. **Output** — `save_dataframe(s)` (`output.py`) writes CSV or JSON to disk. When `output_dir` is set on `generate_for_schemas()`, every table is flushed to disk immediately after generation and the in-memory result is slimmed to FK columns only (to keep RAM bounded). **Examples that access `results[schema_name]` after the call must reload from the saved CSV** — the in-memory DataFrame will be empty or FK-only.

10. **Observability** — `RunReport` / `TableReport` / `ColumnReport` (`run_report.py`) capture per-column strategy (`fk_sampler` | `codegen_simple` | `codegen_semantic` | `direct_llm`), token counts, cost, and cache hit/miss. Always accessible via `generator.last_report` after a run. An HTML report is auto-saved to `output_dir/run_report_<timestamp>.html` when `output_dir` is set.

## Output directory layout (dbt-style)

```
output_dir/
  customers.csv
  orders.csv
  ...
  run_report_20260608_120000.html   ← auto-saved after every run
  .syda_cache/
    orders_21a14bdfa18fe8f5.py      ← human-editable codegen artifact
    reviews_8e10a7047151ce40.py
```

## schema_inference Extension

`schema_inference/` is a self-contained extension (not part of the installed `syda` package) that adds database-to-schema inference:

- `schema_modules.py` — connects to SQLite/MySQL/PostgreSQL via SQLAlchemy, extracts table structure, maps SQL types to Syda types, and produces JSON schema files with a validation report.
- `db_schema_inference.py` — higher-level wrapper that produces per-table YAML/JSON files.
- `syda_integration.py` — monkey-patches `SyntheticDataGenerator` with a `create_schemas_from_database()` method via `patch_generator(generator)`.

Typical two-step workflow:
```python
from db_schema_inference import patch_generator
patch_generator(generator)
schema_files = generator.create_schemas_from_database("sqlite:///my.db", output_dir="schemas")
results = generator.generate_for_schemas(schemas=schema_files, ...)
```

## Key Conventions

- All public configuration goes through `ModelConfig` (Pydantic); never pass raw dicts to `LLMClient`.
- `ModelConfig` large-dataset fields: `generation_mode: Literal['auto','direct','codegen'] = 'auto'`, `batch_size: Optional[int] = None` (auto-selected when None), `max_retries: int = 3`.
- Generator functions registered with `GeneratorManager` have signature `fn(row: pd.Series, col_name: str) -> Any`.
- Schema dict format: top-level keys are column names, values are dicts with at minimum a `"type"` key. A `"_meta"` key holds table-level metadata (description, primary keys, etc.) and is stripped before passing to the LLM.
- `force_llm: true` on a column forces LLM generation in codegen mode regardless of cache state.
- `_generate_data_with_llm()` returns a `(df, in_tok, out_tok)` tuple — never unpack as a single value.
- When `output_dir` is set, never rely on the returned `results` dict for full DataFrames — read from the saved CSVs instead.
- Cost estimation is automatic — fallback to 0.0 if the model is unrecognised.
- `result.usage` is a property in pydantic-ai ≥1.0 — never call it as `result.usage()`.
- Conventional commits required: `feat(scope):`, `fix(scope):`, `docs:`, `test:`, `refactor:`.
- DCO: all commits must use `git commit -s`.

## Examples Directory

```
examples/
  cli/
    demo.sh                     # 10-step CLI demo (healthcare + SQLite)
    demo_large_dataset.sh       # large dataset CLI demo (direct/codegen/multi-table)
    schemas/                    # patient.yml, provider.yml, appointment.yml
    schemas_large/              # product.yml, order.yml (for large dataset demo)
  force_llm/
    example_force_llm.py        # 600-row product catalog with force_llm columns
    schema_products.yml
  large_dataset/
    example_large_dataset_postgres.py  # 18k-row e-commerce demo (Grok-3 + PostgreSQL)
  model_selection/
    example_claude_models.py    # Haiku / Sonnet / Opus comparison
  structured_only/
    example_dict_schemas.py
    example_yaml_schemas.py
  health_insurance_claims/      # unstructured PDF + structured CSV demo
  quickstart_output_data/       # output from README quickstart
```

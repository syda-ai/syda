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

# Docs
mkdocs serve

# schema_inference tests (separate pytest.ini)
cd schema_inference && python -m pytest test_modules.py -v
```

**DCO required**: all commits must be signed with `git commit -s`.

## Architecture

`SyntheticDataGenerator` (`syda/generate.py`) is the central class that orchestrates the full pipeline:

1. **Schema parsing** — `SchemaLoader` (`schema_loader.py`) accepts Python dicts, JSON/YAML files, or SQLAlchemy ORM models and normalizes them into a common format including foreign keys, metadata, and template fields.

2. **Dependency resolution** — `ForeignKeyHandler` (`dependency_handler.py`) uses networkx to build a DAG of table dependencies and returns a topologically sorted order so parent tables are generated before children, maintaining referential integrity.

3. **LLM generation** — `LLMClient` (`llm.py`) wraps the `instructor` library to provide structured (Pydantic model) output from any supported provider: OpenAI, Anthropic, Gemini, Azure OpenAI, or Grok. Provider selection and parameters are configured via `ModelConfig` (`schemas.py`).

4. **Custom generators** — `GeneratorManager` (`custom_generators.py`) maintains a registry of user-supplied generator functions keyed by data type or column name. FK columns are automatically registered to sample from parent table values.

5. **Unstructured / template data** — `TemplateProcessor` (`templates.py`) finds `{{ field_name }}` placeholders in documents and fills them with generated values. `UnstructuredDataProcessor` (`unstructured.py`) reads source documents (PDF, DOCX, Excel, HTML, text, images via OCR).

6. **Output** — `save_dataframe(s)` (`output.py`) writes CSV or JSON to disk, creating directories as needed.

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
- Generator functions registered with `GeneratorManager` have signature `fn(row: pd.Series, col_name: str) -> Any`.
- Schema dict format: top-level keys are column names, values are dicts with at minimum a `"type"` key. A `"_meta"` key holds table-level metadata (description, primary keys, etc.) and is stripped before passing to the LLM.
- Conventional commits required: `feat(scope):`, `fix(scope):`, `docs:`, `test:`, `refactor:`.

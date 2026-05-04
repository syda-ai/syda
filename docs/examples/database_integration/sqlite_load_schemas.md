---
title: SQLite — In-Memory Schemas | Database Integration Examples
description: Generate synthetic data from a SQLite database using DatabaseSchemaLoader with in-memory schema dicts — no intermediate files required.
keywords:
  - SQLite synthetic data
  - DatabaseSchemaLoader
  - in-memory schema
  - syda database example
---

# SQLite — In-Memory Schemas (Option A)

Connect to a SQLite database, infer all table schemas automatically, generate synthetic data, and write it back — all without writing any schema files to disk.

## When to use this approach

- You want the simplest possible workflow
- You don't need to inspect or edit schemas before generating
- You're prototyping or running one-off data generation jobs

## Install

```bash
pip install syda sqlalchemy python-dotenv
```

## Full example

```python
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig

load_dotenv()

# --- 1. Connect to your database ---
engine = create_engine("sqlite:///healthcare_demo.db")

# --- 2. Infer schemas directly as dicts (no files written) ---
loader  = DatabaseSchemaLoader(engine)
schemas = loader.load_schemas()
# schemas = {"patient": {...}, "provider": {...}, ...}

# --- 3. Generate synthetic data ---
generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=8192,
    )
)

results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={
        "patient": 10, "provider": 5,
        "diagnosis": 20, "claim": 20,
        "adjudication": 20, "payment": 20,
    },
    prompts={
        "patient":      "Generate realistic patient records with diverse ages (18-85) and genders.",
        "provider":     "Generate realistic healthcare providers with diverse specialties.",
        "diagnosis":    "Generate realistic diagnoses using ICD-10 codes (e.g. I10, E11.9).",
        "claim":        "Generate realistic healthcare claims using CPT codes, amounts $50-$5000.",
        "adjudication": "Generate adjudication records: ~60% Approved, ~25% Partial, ~15% Denied.",
        "payment":      "Generate payment records, mostly Paid status.",
    },
    output_dir="output/load_schemas",
)

# --- 4. Write generated data back to the database ---
loader.write_to_database(results)
```

## What happens

1. `DatabaseSchemaLoader` connects to the database and reads every table's columns, types, primary keys, and foreign keys via SQLAlchemy's `inspect()`.
2. `load_schemas()` returns a `dict[str, dict]` — one schema dict per table — ready for `generate_for_schemas()`.
3. `generate_for_schemas()` resolves the FK dependency graph, generates each table in topological order (parents before children), and saves CSVs to `output_dir`.
4. `write_to_database()` inserts each DataFrame back into the database in the same FK-safe order.

## Set your API key

```bash
# .env
ANTHROPIC_API_KEY=your_key_here
```

## See also

- [SQLite — File-Based Schemas](sqlite_save_schemas.md) — save YAML files first, then generate
- [PostgreSQL Example](postgresql.md) — full cycle against a real PostgreSQL instance
- [Database Integration deep dive](../../deep_dive/database_integration.md)

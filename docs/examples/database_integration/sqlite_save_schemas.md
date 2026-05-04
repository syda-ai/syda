---
title: SQLite — File-Based Schemas | Database Integration Examples
description: Generate synthetic data from a SQLite database by first saving inferred schemas as YAML files, then generating from those files.
keywords:
  - SQLite synthetic data
  - DatabaseSchemaLoader
  - file-based schema
  - YAML schema inference
  - syda database example
---

# SQLite — File-Based Schemas (Option B)

Connect to a SQLite database, save the inferred schemas as YAML files, optionally edit them, then generate synthetic data from those files and write it back to the database.

## When to use this approach

- You want to review or tweak inferred schemas before generating
- You want to version-control schema files alongside your project
- You plan to reuse the same schema files across multiple generation runs

## Install

```bash
pip install syda sqlalchemy python-dotenv
```

## Full example

```python
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig

load_dotenv()

# --- 1. Connect to your database ---
engine = create_engine("sqlite:///healthcare_demo.db")

# --- 2. Save inferred schemas to YAML files ---
loader      = DatabaseSchemaLoader(engine)
schema_files = loader.save_schemas("schema_files/", format="yaml")
# schema_files = {"patient": "/abs/path/schema_files/patient.yaml", ...}

# Optionally edit the YAML files before generating:
#   - add column descriptions
#   - change inferred types
#   - restrict value ranges

# --- 3. Generate synthetic data from schema file paths ---
generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=8192,
    )
)

results = generator.generate_for_schemas(
    schemas=schema_files,          # file paths instead of dicts
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
    output_dir="output/save_schemas",
)

# --- 4. Write generated data back to the database ---
loader.write_to_database(results)
```

## Generated schema file (example)

`schema_files/patient.yaml`:

```yaml
patient_id:
  type: integer
  primary_key: true
patient_name:
  type: string
age:
  type: integer
gender:
  type: string
date_of_birth:
  type: date
```

You can add a `description` or `constraints` key to any column before generating — Syda passes the full schema to the LLM as context.

## Set your API key

```bash
# .env
ANTHROPIC_API_KEY=your_key_here
```

## See also

- [SQLite — In-Memory Schemas](sqlite_load_schemas.md) — simpler workflow, no files written
- [PostgreSQL Example](postgresql.md) — full cycle against a real PostgreSQL instance
- [Database Integration deep dive](../../deep_dive/database_integration.md)

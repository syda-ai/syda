---
title: PostgreSQL Example | Database Integration Examples
description: Generate synthetic data from a PostgreSQL database using DatabaseSchemaLoader, then write the results back in FK-safe order.
keywords:
  - PostgreSQL synthetic data
  - DatabaseSchemaLoader
  - psycopg2
  - syda database example
---

# PostgreSQL — Full Cycle Example

Connect to a PostgreSQL database, infer schemas, generate synthetic healthcare data, and write it back to the database — all in FK-safe topological order.

## Install

```bash
pip install syda sqlalchemy psycopg2-binary python-dotenv
```

## Set credentials

```bash
# .env
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=syda_healthcare_demo
ANTHROPIC_API_KEY=your_key_here
```

## Full example

```python
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig

load_dotenv()

# --- 1. Build connection string from environment ---
conn_str = (
    f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', 'postgres')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'syda_healthcare_demo')}"
)

engine = create_engine(conn_str)

# --- 2. Infer schemas from PostgreSQL ---
loader  = DatabaseSchemaLoader(engine)
schemas = loader.load_schemas()
print(f"Inferred {len(schemas)} tables: {', '.join(schemas.keys())}")

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
    output_dir="output/postgres",
)

# --- 4. Write back to PostgreSQL in FK-safe order ---
loader.write_to_database(results)
# Insertion order: patient → provider → diagnosis → claim → adjudication → payment
```

## FK-safe insertion order

`write_to_database()` resolves foreign key dependencies automatically. For the healthcare schema the resolved insertion order is:

```
patient → provider → diagnosis → claim → adjudication → payment
```

Parent rows always exist before child FK values are inserted, so no constraint violations occur.

## Using `if_exists`

```python
loader.write_to_database(results, if_exists="replace")   # truncate then insert
loader.write_to_database(results, if_exists="append")    # add rows (default)
loader.write_to_database(results, if_exists="fail")      # error if rows exist
```

## See also

- [SQLite — In-Memory Schemas](sqlite_load_schemas.md)
- [SQLite — File-Based Schemas](sqlite_save_schemas.md)
- [Database Integration deep dive](../../deep_dive/database_integration.md)

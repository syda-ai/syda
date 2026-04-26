# Syda Schema Inference Extension

Extends the [Syda](https://github.com/syda-ai/syda) synthetic data framework with automated schema inference from relational databases. Connect to an existing database, extract its structure, and produce synthetic data — no manual schema definition required.

---

## What it does

| Step | Module | Description |
|------|--------|-------------|
| 1 | Database Connector | Connects to SQLite, MySQL, or PostgreSQL via a SQLAlchemy URL |
| 2 | Metadata Extractor | Retrieves table names, column names, types, and nullability |
| 3 | Constraint Extractor | Detects primary keys (including composite) and foreign keys |
| 4 | Type Mapping Engine | Maps SQL types → Syda-compatible types (`integer`, `string`, `date`, etc.) |
| 5 | Schema Builder | Produces a flat Python dictionary compatible with `generate_for_schemas()` |
| 6 | Compatibility Validator | Validates each table schema against Syda; falls back to structural checks |
| 7 | Versioning Module | Saves three timestamped output files for reproducibility |
| 8 | DB Schema Inference | Adds `create_schemas_from_database()` directly to Syda's generator |

---

## Requirements

- Python 3.9+
- SQLAlchemy

```bash
pip install sqlalchemy
```

For MySQL support:
```bash
pip install pymysql
```

For PostgreSQL:
```bash
pip install psycopg2-binary
```

For running tests:
```bash
pip install pytest
```

Install everything at once:
```bash
pip install sqlalchemy psycopg2-binary pymysql pytest syda python-dotenv
```

---

## Quickstart — Syda integration (recommended)

`db_schema_inference.py` adds `create_schemas_from_database()` directly to Syda's
`SyntheticDataGenerator`, enabling a clean two-step workflow:

```python
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv
from db_schema_inference import patch_generator

load_dotenv()

generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001"
    )
)

# Patch the generator with schema inference capability

patch_generator(generator)

# Step 1: Infer schema from database and save as YAML files

schema_files = generator.create_schemas_from_database(
    connection_string_or_engine="sqlite:///healthcare_demo.db",
    output_dir="generated_schemas",
    table_names=None,  # None = all tables
    format="yaml"
)

# Step 2: Generate synthetic data

results = generator.generate_for_schemas(
    schemas=schema_files,
    sample_sizes={"patient": 10, "claim": 20},
    prompts={"patient": "Generate realistic patient records."},
    output_dir="synthetic_output"
)
```

---

## CLI usage

### SQLite

```bash

# Create the healthcare demo database and infer its schema in one command

python schema_modules.py \
  --db-url sqlite:///healthcare_demo.db \
  --create-demo-sqlite \
  --output sqlite_schema.json

# Run on any existing SQLite database

python schema_modules.py \
  --db-url sqlite:///your_database.db \
  --output schema.json
```

### MySQL

```bash
python schema_modules.py \
  --db-url "mysql+pymysql://root:YOUR_PASSWORD@localhost/healthcare_demo" \
  --output mysql_schema.json
```

### PostgreSQL

```bash
python schema_modules.py \
  --db-url "postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost/healthcare_demo" \
  --output postgres_schema.json
```

### Full end-to-end pipeline (CLI)

```bash
python syda_integration.py --db-url sqlite:///healthcare_demo.db --create-demo-sqlite --output-dir synthetic_output

python syda_integration.py --db-url "postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost/healthcare_demo" --output-dir synthetic_output_postgres

python syda_integration.py --db-url "mysql+pymysql://root:YOUR_PASSWORD@localhost/healthcare_demo" --output-dir synthetic_output_mysql
```

---

## Output files

Running `schema_modules.py` produces three files:

| File | Contents |
|------|----------|
| `schema.json` | Full inferred schema with version envelope |
| `schema_syda_dict.json` | Syda-ready flat dictionary |
| `schema_validation_report.json` | Per-table PASS/WARN/FAIL results |

Running `db_schema_inference.py` produces one YAML or JSON file per table in the specified output directory.

---

## Type mapping reference

| SQL type | Syda type |
|----------|-----------|
| INT, INTEGER, BIGINT, SMALLINT, TINYINT | `integer` |
| TEXT, VARCHAR, CHAR, NVARCHAR | `string` |
| DATE, DATETIME, TIMESTAMP, TIME | `date` |
| DECIMAL, NUMERIC, FLOAT, REAL, DOUBLE | `float` |
| BOOLEAN | `boolean` |
| Anything else | `string` (safe default) |

---

## Running the tests

```bash
python -m pytest test_modules.py -v
```

The test suite covers 60 test cases across 11 test classes:

- Type mapping (all variants + edge cases)
- Database connector (valid and invalid connections)
- Metadata extractor (columns, nullability, multiple tables)
- Constraint extractor (single PK, composite PK, FK references)
- Schema builder (`_meta` stripping, flat dict structure)
- Compatibility validator (PASS/WARN/FAIL logic)
- Versioning module (file creation, version envelopes, report summaries)
- Demo database integration (full 6-entity healthcare pipeline)
- Edge cases (empty tables, null-only columns, empty database)
- DB schema inference (YAML/JSON output, FK detection, type mapping)

---

## Project structure

```
schema_modules.py          # Core inference pipeline
db_schema_inference.py     # Syda generator extension (create_schemas_from_database)
syda_integration.py        # CLI end-to-end pipeline
test_modules.py            # Unit and integration tests
pytest.ini                 # Test runner configuration
README.md                  # This file
create_mysql_demo.sql      # MySQL healthcare demo setup script
create_postgres_demo.sql   # PostgreSQL healthcare demo setup script
.env.example               # API key template
```

---

## Demo database schema

```
patient ──┐
          ├──► diagnosis ──► claim ──► adjudication
provider ─┘                      └──► payment
```

Tables: `patient`, `provider`, `diagnosis`, `claim`, `adjudication`, `payment`

---

## License

This project is an extension of the open-source Syda framework.
Base repository: https://github.com/syda-ai/syda
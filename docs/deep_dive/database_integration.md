---
title: Database Integration | Syda Deep Dive
description: Connect Syda to any SQLAlchemy-compatible database to automatically infer schemas, generate synthetic data, and write it back — no manual schema definition needed.
keywords:
  - database integration
  - DatabaseSchemaLoader
  - SQLAlchemy synthetic data
  - PostgreSQL synthetic data
  - MySQL synthetic data
  - schema inference
  - write back to database
---

# Database Integration

`DatabaseSchemaLoader` connects Syda directly to an existing relational database. It infers table schemas automatically (columns, types, primary keys, foreign keys) and returns them in a format that `generate_for_schemas()` accepts — no manual schema definition required.

After generation, `write_to_database()` inserts the synthetic rows back into the database in FK-safe topological order so referential integrity is preserved.

## Supported Databases

Any database with a SQLAlchemy dialect works:

| Database | Driver package | Connection string |
|---|---|---|
| SQLite | built-in | `sqlite:///mydb.db` |
| PostgreSQL | `psycopg2-binary` | `postgresql+psycopg2://user:pass@host/db` |
| MySQL / MariaDB | `pymysql` | `mysql+pymysql://user:pass@host/db` |
| MS SQL Server | `pyodbc` | `mssql+pyodbc://user:pass@host/db?driver=...` |
| Oracle | `oracledb` | `oracle+oracledb://user:pass@host/db` |

Install the driver for your database before connecting:

```bash
pip install sqlalchemy
pip install psycopg2-binary   # PostgreSQL
pip install pymysql           # MySQL / MariaDB
```

## Option A — In-Memory Schemas

Load schemas as Python dicts and pass them directly to `generate_for_schemas()`. No files are written to disk.

```python
from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig
from dotenv import load_dotenv

load_dotenv()

generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=8192,
    )
)

loader  = DatabaseSchemaLoader("sqlite:///mydb.db")
schemas = loader.load_schemas()           # {table_name: schema_dict}

results = generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes={"patient": 10, "claim": 20},
    prompts={"patient": "Generate realistic patient records."},
    output_dir="output",
)

loader.write_to_database(results)
```

Use this approach when you want the simplest possible workflow and don't need to inspect or reuse the schema files.

## Option B — File-Based Schemas

Save one YAML (or JSON) file per table first. Inspect or edit the files, then pass the file paths to `generate_for_schemas()`.

```python
loader = DatabaseSchemaLoader("postgresql+psycopg2://user:pass@localhost/mydb")

# Write schema files — returns {table_name: absolute_file_path}
schema_files = loader.save_schemas("schemas/", format="yaml")

# Optionally edit schemas/ files to add descriptions or constraints

results = generator.generate_for_schemas(schemas=schema_files, output_dir="output")

loader.write_to_database(results)
```

Use this approach when you want to:

- Review or adjust inferred schemas before generating
- Version-control schema files alongside your project
- Reuse saved schema files across multiple generation runs

## Selecting Specific Tables

Both methods accept an optional `table_names` list. Omit it to process all tables.

```python
schemas = loader.load_schemas(table_names=["patient", "provider", "claim"])

schema_files = loader.save_schemas("schemas/", table_names=["patient", "claim"])
```

## Writing Data Back to the Database

`write_to_database()` takes the `results` dict returned by `generate_for_schemas()` and inserts each DataFrame into its table.

```python
loader.write_to_database(results)                    # append rows (default)
loader.write_to_database(results, if_exists="replace")  # truncate then insert
```

| `if_exists` | Behaviour |
|---|---|
| `"append"` (default) | Add new rows to existing data |
| `"replace"` | Drop existing rows, then insert |
| `"fail"` | Raise an error if the table already has rows |

Rows are inserted in topological order — parent tables first, child tables after — so foreign key constraints are never violated.

## How FK-Safe Insertion Order Works

`write_to_database()` calls `_fk_insertion_order()` internally, which:

1. Queries `inspector.get_foreign_keys(table)` for every table in the result set
2. Builds a dependency graph: `diagnosis → {patient, provider}`
3. Runs a depth-first topological sort — recurses into dependencies before appending self

For the healthcare schema the resolved order is:

```
patient → provider → diagnosis → claim → adjudication → payment
```

Parent PKs always exist before child FK values are inserted.

## Type Mapping

SQL types are mapped to Syda types as follows:

| SQL type | Syda type |
|---|---|
| INT, INTEGER, BIGINT, SMALLINT, TINYINT | `integer` |
| TEXT, VARCHAR, CHAR, NVARCHAR, CLOB | `string` |
| DATE, DATETIME, TIMESTAMP, TIME | `date` |
| DECIMAL, NUMERIC, FLOAT, REAL, DOUBLE | `float` |
| BOOLEAN | `boolean` |
| Foreign key column | `foreign_key` |
| Anything else | `string` |

## Full PostgreSQL Example

```python
import os
from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig
from dotenv import load_dotenv

load_dotenv()

conn = (
    f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}:"
    f"{os.getenv('DB_PASSWORD', 'postgres')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'mydb')}"
)

loader    = DatabaseSchemaLoader(conn)
schemas   = loader.load_schemas()

generator = SyntheticDataGenerator(
    model_config=ModelConfig(provider="anthropic", model_name="claude-haiku-4-5-20251001")
)

results = generator.generate_for_schemas(schemas=schemas, output_dir="output")

loader.write_to_database(results)
```

Set credentials in your `.env` file:

```bash
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydb
ANTHROPIC_API_KEY=your_key
```

## API Reference

### `DatabaseSchemaLoader(connection_string_or_engine)`

| Parameter | Type | Description |
|---|---|---|
| `connection_string_or_engine` | `str` or SQLAlchemy `Engine` | SQLAlchemy URL or an existing engine object |

### `load_schemas(table_names=None)`

Returns `Dict[str, Dict]` — schema dicts keyed by table name, ready for `generate_for_schemas(schemas=...)`.

### `save_schemas(output_dir, table_names=None, format="yaml")`

Writes one file per table and returns `Dict[str, str]` — table name to absolute file path.

| Parameter | Default | Description |
|---|---|---|
| `output_dir` | required | Directory to write schema files into |
| `table_names` | `None` (all tables) | Subset of tables to process |
| `format` | `"yaml"` | `"yaml"` or `"json"` |

### `write_to_database(data, if_exists="append")`

Inserts DataFrames back into the database in FK-safe order.

| Parameter | Default | Description |
|---|---|---|
| `data` | required | `Dict[str, DataFrame]` as returned by `generate_for_schemas()` |
| `if_exists` | `"append"` | `"append"`, `"replace"`, or `"fail"` |

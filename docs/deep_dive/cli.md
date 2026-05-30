---
title: CLI Reference | Syda Deep Dive
description: Use the syda command-line interface to generate synthetic data, validate schemas, infer database schemas, and write generated data back to a database — no Python required.
keywords:
  - syda CLI
  - command line
  - synthetic data command line
  - syda generate
  - syda validate
  - syda db infer
  - syda db generate
  - bash synthetic data
---

# CLI Reference

The `syda` CLI lets you generate synthetic data, validate schemas, and work with databases entirely from the terminal — no Python required.

```bash
pip install syda
```

```
syda --help

Commands:
  generate  Generate synthetic data from schema(s).
  validate  Validate schema file(s) without generating data.
  db        Database integration: infer schemas and generate data from a live DB.
  version   Print syda version and exit.
```

---

## Setup

Set your API key in the environment (or a `.env` file in the working directory):

```bash
export ANTHROPIC_API_KEY=your_key
# or
export OPENAI_API_KEY=your_key
# or
export GEMINI_API_KEY=your_key
```

The provider is auto-detected from whichever key is set. You can also pass `--provider` and `--api-key` explicitly on every command.

---

## syda version

Print the installed version.

```bash
syda version
# syda 0.0.7
```

---

## syda validate

Validate one or more schema files without making any LLM calls. Useful as a CI pre-check before generation.

```bash
syda validate --schema patients.yaml
syda validate --schema schemas/
```

```
  [OK] patient (/path/schemas/patient.yml)
  [OK] provider (/path/schemas/provider.yml)
  [OK] appointment (/path/schemas/appointment.yml)

All 3 schema(s) valid.
```

If any schema fails validation, `syda validate` exits with a non-zero status code so CI pipelines catch the error.

### Options

| Option | Short | Description |
|---|---|---|
| `--schema PATH` | `-s` | Schema file (YAML/JSON) or directory of schema files. **Required.** |

---

## syda generate

Generate synthetic data from a schema file or directory of schema files.

### Single schema → CSV

```bash
syda generate --schema patients.yaml --rows 50 --output patients.csv
```

### Single schema → JSON

```bash
syda generate --schema patients.yaml --rows 50 --output patients.json
```

The output format is inferred from the file extension (`.csv` or `.json`). Use `--format` to override.

### Directory of schemas — multi-table with FK integrity

```bash
syda generate --schema schemas/ --rows 100 --output-dir ./data
```

Syda resolves foreign key dependencies automatically and generates tables in the correct topological order (parents before children).

```bash
schemas/
├── department.yml      # generated first (no deps)
├── employee.yml        # generated second (FK → department)
└── performance.yml     # generated last (FK → employee)
```

### Explicit provider and model

```bash
syda generate \
  --schema schemas/ \
  --rows 50 \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --output-dir ./data
```

### Add a context prompt

```bash
syda generate \
  --schema schemas/patient.yml \
  --rows 20 \
  --prompt "US healthcare context, include diverse demographics" \
  --output patients.csv
```

### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--schema PATH` | `-s` | — | Schema file (YAML/JSON) or directory. **Required.** |
| `--rows N` | `-n` | `10` | Rows to generate per table. |
| `--output FILE` | `-o` | — | Output file path. Format inferred from `.csv`/`.json` extension. Single schema only. |
| `--output-dir DIR` | | — | Directory for multi-table output (one file per schema). |
| `--format csv\|json` | `-f` | `csv` | Explicit format override. |
| `--provider` | `-p` | auto | LLM provider: `anthropic`, `openai`, `gemini`, `grok`, `openai_compatible`, `azureopenai`. Auto-detected from env vars. |
| `--model NAME` | `-m` | provider default | Model name. |
| `--api-key KEY` | | env var | API key. Falls back to the provider's standard env var. |
| `--base-url URL` | | — | Base URL for `openai_compatible` providers (e.g. Ollama). |
| `--prompt TEXT` | | — | Context prompt applied to all schemas during generation. |
| `--temperature FLOAT` | | model default | Sampling temperature `0.0`–`1.0`. |

!!! note "Output rules"
    - `--output` is for a single schema file only.
    - `--output-dir` is required when `--schema` points to a directory.
    - You cannot combine `--output` and `--output-dir`.

---

## syda db

The `syda db` subgroup provides database-native workflows — connect to a live database, infer schemas automatically, generate synthetic data, and optionally write the results back.

### syda db infer

Introspect a database and save one schema file per table. No data is generated.

```bash
syda db infer \
  --db-url sqlite:///mydb.db \
  --output-dir schemas/
```

```
Connected to: sqlite:///mydb.db
Inferring schemas → schemas/ ...
  [OK] provider → /path/schemas/provider.yaml
  [OK] patient  → /path/schemas/patient.yaml
  [OK] appointment → /path/schemas/appointment.yaml

Inferred 3 schema(s).
```

You can then edit the schema files to add descriptions or constraints, and pass them to `syda generate`.

#### Specific tables and JSON format

```bash
syda db infer \
  --db-url postgresql://user:pass@localhost/mydb \
  --output-dir schemas/ \
  --tables patients,providers \
  --format json
```

#### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--db-url URL` | `-d` | — | SQLAlchemy connection URL. **Required.** |
| `--output-dir DIR` | `-o` | — | Directory to write schema files. **Required.** |
| `--tables TABLE,...` | `-t` | all tables | Comma-separated list of tables to infer. |
| `--format yaml\|json` | `-f` | `yaml` | Schema file format. |

---

### syda db generate

Infer schemas from a database, generate synthetic data, and optionally write it back.

```bash
syda db generate \
  --db-url sqlite:///mydb.db \
  --rows 50 \
  --output-dir ./data
```

#### Generate and write back in one step

```bash
syda db generate \
  --db-url postgresql://user:pass@localhost/mydb \
  --rows 100 \
  --write-back
```

Rows are inserted in FK-safe topological order (parent tables first), so referential integrity is always preserved.

#### Replace existing data

```bash
syda db generate \
  --db-url sqlite:///mydb.db \
  --rows 50 \
  --write-back \
  --if-exists replace
```

#### Specific tables only

```bash
syda db generate \
  --db-url sqlite:///mydb.db \
  --tables provider,patient \
  --rows 20 \
  --output-dir ./data
```

#### Full example with all options

```bash
syda db generate \
  --db-url postgresql://user:pass@localhost/prod_db \
  --rows 200 \
  --tables patient,provider,appointment \
  --output-dir ./synthetic \
  --format json \
  --write-back \
  --if-exists replace \
  --provider anthropic \
  --model claude-haiku-4-5-20251001 \
  --prompt "US healthcare context, diverse demographics and realistic clinical data" \
  --temperature 0.7
```

#### Options

| Option | Short | Default | Description |
|---|---|---|---|
| `--db-url URL` | `-d` | — | SQLAlchemy connection URL. **Required.** |
| `--rows N` | `-n` | `10` | Rows per table. |
| `--tables TABLE,...` | `-t` | all tables | Comma-separated table subset. |
| `--output-dir DIR` | `-o` | — | Save generated CSV/JSON files here. Optional. |
| `--format csv\|json` | `-f` | `csv` | File format when `--output-dir` is set. |
| `--write-back` | `-w` | off | Insert generated rows into the database. |
| `--if-exists` | | `append` | `append` / `replace` / `fail` — write-back behaviour. |
| `--provider` | `-p` | auto | LLM provider. |
| `--model NAME` | `-m` | provider default | Model name. |
| `--api-key KEY` | | env var | API key. |
| `--base-url URL` | | — | Base URL for `openai_compatible` providers. |
| `--prompt TEXT` | | — | Context prompt applied to all tables. |
| `--temperature FLOAT` | | model default | Sampling temperature `0.0`–`1.0`. |

---

## Supported Databases

Any database with a SQLAlchemy dialect:

| Database | Driver | Connection string |
|---|---|---|
| SQLite | built-in | `sqlite:///mydb.db` |
| PostgreSQL | `psycopg2-binary` | `postgresql+psycopg2://user:pass@host/db` |
| MySQL / MariaDB | `pymysql` | `mysql+pymysql://user:pass@host/db` |
| MS SQL Server | `pyodbc` | `mssql+pyodbc://user:pass@host/db?driver=...` |

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Validation error, generation failure, bad options, or connection failure |

All error messages are written to stdout with a descriptive `Error:` prefix, making them easy to catch in CI pipelines.

---

## CI / Pipeline Usage

{% raw %}
```yaml
# GitHub Actions example
- name: Validate schemas
  run: syda validate --schema schemas/

- name: Generate test fixtures
  run: |
    syda generate \
      --schema schemas/ \
      --rows 20 \
      --output-dir tests/fixtures
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```
{% endraw %}

---

## Example

A complete runnable bash demo is available at [`examples/cli/demo.sh`](https://github.com/syda-ai/syda/blob/main/examples/cli/demo.sh). It covers all 10 workflows end-to-end using healthcare schemas and a SQLite database.

```bash
cd examples/cli
chmod +x demo.sh
./demo.sh
```

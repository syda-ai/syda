# Large Dataset Example — PostgreSQL

Demonstrates syda's large-dataset features against a live PostgreSQL database:

- **Auto mode** — direct generation (≤500 rows, chunked LLM calls) vs code-gen (>500 rows)
- **Code-gen mode** — LLM classifies columns, writes Python generators for simple columns, only calls the LLM for semantic (free-text) columns
- **Codegen cache** — generated Python functions are cached under `output/.syda_cache/` keyed by schema content hash; cache is reused on subsequent runs, saving LLM calls and cost
- **Run report** — HTML report auto-saved to `output/run_report_<timestamp>.html` with per-column strategy breakdown, token counts, cache hit/miss, and estimated cost
- **Multi-table FK integrity** — 5-table e-commerce schema with a full FK chain

## Schema

```
customers (200 rows)  →  orders (1,000 rows)  →  order_items (2,000 rows)
products  (100 rows)  ↗                        ↗
reviews   (600 rows)  ← products + customers
```

Tables ≤500 rows use **direct mode** (chunked LLM calls, `batch_size=50`).  
Tables >500 rows use **code-gen mode** (1 analysis call + semantic columns only).

## Setup

```bash
pip install syda sqlalchemy psycopg2-binary python-dotenv
```

Set environment variables (or add to `.env`):

```bash
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=syda_large_dataset_demo
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python example_large_dataset_postgres.py
```

## Output

```
output/
├── customers.csv
├── orders.csv
├── order_items.csv
├── products.csv
├── reviews.csv
├── run_report_<timestamp>.html     ← full observability report
└── .syda_cache/
    ├── orders_<hash>.json          ← cached codegen artifact
    ├── order_items_<hash>.json
    └── reviews_<hash>.json
```

Run again with the same schema → cache HITs for all codegen tables, zero analysis LLM calls.

## Accessing the report programmatically

```python
results = generator.generate_for_schemas(schemas=schemas, output_dir="output/", ...)

# Always available after any run
generator.last_report.print_summary()
print(f"Estimated cost: ${generator.last_report.estimated_cost_usd:.4f}")
```

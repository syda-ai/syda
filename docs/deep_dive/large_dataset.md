---
title: Large Dataset Generation | Syda Deep Dive
description: Generate thousands to millions of synthetic rows efficiently using chunked direct mode and code-gen mode — with automatic retry, cost tracking, and per-column run reports.
keywords:
  - large dataset generation
  - chunked generation
  - code-gen mode
  - batch_size
  - generation_mode
  - force_llm
  - cost tracking
  - run report
---

# Large Dataset Generation

Syda supports generating large volumes of synthetic data through two complementary modes that dramatically reduce API calls and memory usage at scale.

---

## How It Works

### Direct Mode (chunked LLM calls)

Used automatically for tables with ≤ 500 rows (or when forced with `generation_mode="direct"`). The `sample_size` is split into chunks of `batch_size` rows; each chunk is a separate LLM call with exponential-backoff retry.

```
sample_size=300, batch_size=50 → 6 LLM calls, each requesting 50 rows
```

Short-count recovery: if a chunk returns fewer rows than requested, syda automatically makes top-up calls for the missing rows.

### Code-Gen Mode (LLM writes Python, runs locally)

Used automatically for tables with > 500 rows (or when forced with `generation_mode="codegen"`).

**Two-phase process:**

1. **Analysis call** (once per schema, cached) — LLM reads the schema and classifies each column:
    - `simple`: can be generated with Python (IDs, dates, enums, emails, phone numbers, codes, numbers)
    - `semantic`: must be LLM-generated (clinical notes, product descriptions, narratives, free-text)

    For each `simple` column, the LLM also writes a Python generator function.

2. **Semantic column calls** — one LLM call per semantic column (chunked if N is very large), generating all values for that column.

Simple columns run entirely locally with zero further LLM calls. For a table with 10,000 rows and 2 semantic columns out of 8 total, only 3 LLM calls are made regardless of row count (1 analysis + 2 semantic).

### Auto-Select

`generation_mode="auto"` (the default) selects the mode based on row count:

| Row count | Mode selected |
|---|---|
| ≤ 500 | `direct` |
| > 500 | `codegen` |

---

## Configuration

```python
from syda import SyntheticDataGenerator, ModelConfig

generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        generation_mode="auto",   # 'auto', 'direct', or 'codegen'
        batch_size=50,            # max rows per LLM call (direct mode)
        max_retries=3,            # exponential-backoff retries per chunk
        max_tokens=8192,
    )
)

results = generator.generate_for_schemas(
    schemas={"products": "schemas/product.yml"},
    sample_sizes={"products": 2000},
    output_dir="output",
)
```

### ModelConfig parameters

| Parameter | Description | Default |
|---|---|---|
| `generation_mode` | `'auto'` / `'direct'` / `'codegen'` | `'auto'` |
| `batch_size` | Max rows per LLM call in direct mode. Auto-selected when `None`. | `None` |
| `max_retries` | Retry attempts per chunk on transient errors | `3` |

---

## CLI

```bash
# Direct mode with explicit chunk size
syda generate --schema schemas/product.yml --rows 300 --batch-size 50 --output-dir ./data

# Code-gen triggered automatically (>500 rows)
syda generate --schema schemas/product.yml --rows 1000 --output-dir ./data

# Force code-gen for any row count
syda generate --schema schemas/product.yml --rows 50 --large-dataset --output-dir ./data

# Multi-table FK chain in code-gen mode
syda generate --schema schemas/ --rows 5000 --large-dataset --output-dir ./data
```

---

## Code-Gen Cache

Generated Python functions are saved as human-editable `.py` files under `output_dir/.syda_cache/`:

```
output_dir/
  products.csv
  orders.csv
  run_report_20260608_120000.html
  .syda_cache/
    products_21a14bdfa18fe8f5.py   ← edit to customise generators
    orders_8e10a7047151ce40.py
```

The cache key is a hash of the schema content + user prompt. On a **cache hit**, the analysis call is skipped entirely. This means:

- Second run on the same schema: instant (no LLM calls for simple columns)
- Schema change: new cache file written, old one deleted

To force regeneration for a specific column, delete its `generate_*` function from the `.py` file — syda treats missing functions as semantic and calls the LLM for that column.

---

## force_llm Column Flag

In code-gen mode, mark any column `force_llm: true` in the schema YAML to always generate it via LLM, even if a cached Python function exists:

```yaml
# schemas/product.yml
product_id:
  type: integer
  constraints:
    primary_key: true

tagline:
  type: text
  description: Short marketing tagline (one punchy sentence)
  force_llm: true    # always LLM-generated

description:
  type: text
  description: Detailed product description (2-3 sentences)
  force_llm: true    # always LLM-generated

category:
  type: text
  description: Product category — Electronics, Clothing, Home, Sports, Books
```

In this example, `product_id` and `category` run locally; `tagline` and `description` always call the LLM.

---

## Memory Management

When `output_dir` is set, each table is **flushed to disk immediately** after it completes. Only FK columns are kept in RAM for child-table generation. This keeps peak memory at `1 chunk × row_size` regardless of total row count.

!!! warning "Reloading from disk"
    When `output_dir` is set, the in-memory `results` dict returned by `generate_for_schemas()` contains only FK columns for tables > threshold (not the full DataFrame). Always reload from the saved CSV for post-generation analysis:

    ```python
    import pandas as pd

    generator.generate_for_schemas(schemas=schemas, output_dir="output", ...)

    # Reload from disk — the in-memory result may be FK-only
    results = {
        name: pd.read_csv(f"output/{name.lower()}.csv")
        for name in schemas
    }
    ```

---

## Observability & Cost Tracking

Every run produces a `RunReport` accessible at `generator.last_report`. An HTML version is auto-saved to `output_dir/run_report_<timestamp>.html`.

```python
generator.generate_for_schemas(...)

report = generator.last_report
report.print_summary()
```

Example output:

```
──────────────────────────────────────────────────────────────────────────
Table              Rows  Mode      Calls    In tok   Out tok  Cache
──────────────────────────────────────────────────────────────────────────
customers           500  direct       10     6,220    32,940  $0.51
products            200  direct        4     2,168    15,291  $0.24
orders             5000  codegen     100    28,300    25,201  $0.46
reviews            3000  codegen      60    16,980    34,856  $0.57
order_items       10000  codegen       0         0         0  $0.00
──────────────────────────────────────────────────────────────────────────
TOTAL             18700              174    53,668   108,288  $1.79
```

Cost is calculated via [genai-prices](https://github.com/pydantic/genai-prices), which supports 600+ models across all providers.

Accessing programmatically:

```python
for table_name, table_report in report.tables.items():
    print(f"{table_name}: {table_report.rows_generated} rows, ${table_report.cost_usd:.4f}")
    for col_name, col in table_report.columns.items():
        print(f"  {col_name}: {col.strategy}, {col.input_tokens} in / {col.output_tokens} out")
```

---

## End-to-End Example

```python
from syda import SyntheticDataGenerator, ModelConfig
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

generator = SyntheticDataGenerator(
    model_config=ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        generation_mode="auto",
        batch_size=50,
        max_retries=3,
        max_tokens=8192,
    )
)

schemas = {
    "products": "schemas/product.yml",
    "orders":   "schemas/order.yml",   # FK → products
}

sample_sizes = {
    "products": 200,    # direct mode  (≤500)
    "orders":  2000,    # codegen mode (>500)
}

output_dir = "output"

generator.generate_for_schemas(
    schemas=schemas,
    sample_sizes=sample_sizes,
    output_dir=output_dir,
    prompts={
        "products": "E-commerce products: electronics, clothing, home goods, sports. Prices $5–$500.",
        "orders":   "Customer orders over the past 2 years. Status: 40% delivered, 25% shipped, 20% processing, 10% pending, 5% cancelled.",
    },
)

# Reload from disk (in-memory result is FK-only after flush)
results = {
    name: pd.read_csv(os.path.join(output_dir, f"{name}.csv"))
    for name in schemas
}

print(f"Products: {len(results['products'])} rows")
print(f"Orders:   {len(results['orders'])} rows")

# Cost summary
generator.last_report.print_summary()
```

---

## See Also

- [Model Configuration](model_configuration.md) — `ModelConfig` parameters for large datasets
- [CLI Reference](cli.md) — `--batch-size` and `--large-dataset` flags
- [Output Options](output_options.md) — how `output_dir` affects memory and disk flushing
- [Large Dataset Example](https://github.com/syda-ai/syda/blob/main/examples/large_dataset/example_large_dataset_postgres.py) — 18,000-row PostgreSQL demo with Grok-3
- [CLI Large Dataset Demo](https://github.com/syda-ai/syda/blob/main/examples/cli/demo_large_dataset.sh) — all four modes in a single bash script

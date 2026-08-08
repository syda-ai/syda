#!/usr/bin/env python
"""
Benchmark: Grok-4.3 vs Claude Sonnet 4.6 on the same large e-commerce dataset.

Runs identical generation (same schema, same sample sizes, same prompts) with
two models back-to-back and prints a side-by-side comparison of:
  - Wall-clock time per table and total
  - LLM calls, token counts, estimated cost
  - Quality spot-check: sample rows from semantic columns (review_text, shipping_method)

Usage:
    python examples/large_dataset/benchmark_models.py

Requirements:
    GROK_API_KEY and ANTHROPIC_API_KEY in environment or .env file.
    PostgreSQL running (same defaults as example_large_dataset_postgres.py).
"""

import os
import sys
import time

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Smaller sizes for a fair, faster benchmark ────────────────────────────────
# Kept under the codegen threshold (500) for customers/products so direct mode
# is exercised; orders/reviews/order_items above it for codegen mode.
SAMPLE_SIZES = {
    "customers":   200,   # direct mode
    "products":    100,   # direct mode
    "orders":    1_000,   # codegen mode
    "reviews":     600,   # codegen mode
    "order_items": 2_000, # codegen mode
}

PROMPTS = {
    "customers": (
        "Generate realistic e-commerce customers from diverse countries. "
        "Loyalty tiers: ~40% Bronze, 30% Silver, 20% Gold, 10% Platinum."
    ),
    "products": (
        "Generate realistic e-commerce products spanning electronics, "
        "clothing, home goods, and sports equipment. Unit prices $5–$500."
    ),
    "orders": (
        "Generate realistic customer orders. Status distribution: "
        "40% Delivered, 25% Shipped, 20% Processing, 10% Pending, 5% Cancelled. "
        "Order dates spread across the past 2 years."
    ),
    "order_items": (
        "Generate order line items. Each item has quantity 1–5, "
        "a unit price matching the product, and an optional discount 0–20%."
    ),
    "reviews": (
        "Generate product reviews. Ratings skewed positive: "
        "~30% 5-star, 35% 4-star, 20% 3-star, 10% 2-star, 5% 1-star. "
        "review_text should be 1–3 sentences of realistic customer feedback."
    ),
}

MODELS = [
    {
        "label": "Grok-4.3",
        "config": ModelConfig(
            provider="grok",
            model_name="grok-4.3",
            temperature=0.8,
            max_tokens=16384,
            generation_mode="auto",
            batch_size=50,
            max_retries=3,
            extra_kwargs={"base_url": "https://api.x.ai/v1"},
        ),
        "kwargs": {"grok_api_key": os.getenv("GROK_API_KEY")},
        "output_subdir": "benchmark_grok43",
    },
    {
        "label": "Claude Sonnet 4.6",
        "config": ModelConfig(
            provider="anthropic",
            model_name="claude-sonnet-5",
            temperature=0.8,
            max_tokens=16384,
            generation_mode="auto",
            batch_size=50,
            max_retries=3,
        ),
        "kwargs": {},
        "output_subdir": "benchmark_claude_sonnet46",
    },
]


def build_connection_string() -> str:
    return (
        f"postgresql+psycopg2://{os.getenv('DB_USER','postgres')}:"
        f"{os.getenv('DB_PASSWORD','postgres')}@"
        f"{os.getenv('DB_HOST','localhost')}:"
        f"{os.getenv('DB_PORT','5432')}/"
        f"{os.getenv('DB_NAME','syda_large_dataset_demo')}"
    )


def create_demo_schema(engine) -> None:
    with engine.connect() as conn:
        for table in ["order_items", "reviews", "orders", "products", "customers"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        conn.commit()
        conn.execute(text("""
            CREATE TABLE customers (
                customer_id  SERIAL PRIMARY KEY,
                first_name   TEXT NOT NULL, last_name TEXT NOT NULL,
                email        TEXT NOT NULL UNIQUE, phone TEXT,
                city TEXT, country TEXT, signup_date DATE NOT NULL,
                loyalty_tier TEXT CHECK (loyalty_tier IN ('Bronze','Silver','Gold','Platinum'))
            )"""))
        conn.execute(text("""
            CREATE TABLE products (
                product_id   SERIAL PRIMARY KEY,
                product_name TEXT NOT NULL, category TEXT NOT NULL,
                brand TEXT, unit_price NUMERIC(10,2) NOT NULL,
                stock_qty INTEGER NOT NULL DEFAULT 0, description TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE orders (
                order_id        SERIAL PRIMARY KEY,
                customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
                order_date      DATE NOT NULL,
                status          TEXT CHECK (status IN ('pending','processing','shipped','delivered','cancelled')),
                shipping_method TEXT, total_amount NUMERIC(12,2) NOT NULL
            )"""))
        conn.execute(text("""
            CREATE TABLE order_items (
                item_id    SERIAL PRIMARY KEY,
                order_id   INTEGER NOT NULL REFERENCES orders(order_id),
                product_id INTEGER NOT NULL REFERENCES products(product_id),
                quantity   INTEGER NOT NULL CHECK (quantity > 0),
                unit_price NUMERIC(10,2) NOT NULL, discount NUMERIC(5,2) DEFAULT 0.00
            )"""))
        conn.execute(text("""
            CREATE TABLE reviews (
                review_id   SERIAL PRIMARY KEY,
                product_id  INTEGER NOT NULL REFERENCES products(product_id),
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                review_text TEXT, review_date DATE NOT NULL
            )"""))
        conn.commit()


def run_model(label, config, extra_kwargs, output_dir, schemas):
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")

    # Wipe codegen cache for a fair comparison (each model should do its own analysis)
    cache_dir = os.path.join(output_dir, ".syda_cache")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)

    generator = SyntheticDataGenerator(model_config=config, **extra_kwargs)

    t0 = time.time()
    generator.generate_for_schemas(
        schemas=schemas,
        sample_sizes=SAMPLE_SIZES,
        prompts=PROMPTS,
        output_dir=output_dir,
        output_format="csv",
        batch_size=50,
    )
    elapsed = time.time() - t0

    report = generator.last_report
    return elapsed, report


def quality_spot_check(output_dir, label):
    import pandas as pd
    print(f"\n── Quality spot-check: {label} ─────────────────────────────────")

    reviews_path = os.path.join(output_dir, "reviews.csv")
    orders_path  = os.path.join(output_dir, "orders.csv")

    if os.path.exists(reviews_path):
        df = pd.read_csv(reviews_path)
        print(f"  review_text samples (ratings vary):")
        for _, row in df.sample(min(3, len(df)), random_state=42).iterrows():
            print(f"    [{row.get('rating','?')}★] {str(row.get('review_text',''))[:100]}")

    if os.path.exists(orders_path):
        df = pd.read_csv(orders_path)
        if "shipping_method" in df.columns:
            print(f"  shipping_method distribution:")
            for val, cnt in df["shipping_method"].value_counts().head(5).items():
                print(f"    {val:<25} {cnt:>5} ({cnt/len(df)*100:.1f}%)")
        if "status" in df.columns:
            print(f"  order status distribution:")
            for val, cnt in df["status"].value_counts().items():
                print(f"    {val:<25} {cnt:>5} ({cnt/len(df)*100:.1f}%)")


def print_comparison(results):
    print(f"\n{'='*70}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*70}")

    labels   = [r["label"]   for r in results]
    elapsed  = [r["elapsed"] for r in results]
    reports  = [r["report"]  for r in results]

    # Header
    col = 22
    print(f"\n{'Metric':<30}", end="")
    for lbl in labels:
        print(f"{lbl:>{col}}", end="")
    print()
    print("-" * (30 + col * len(labels)))

    # Total time
    print(f"{'Total time (s)':<30}", end="")
    for e in elapsed:
        print(f"{e:>{col}.1f}", end="")
    print()

    # Per-table rows and time
    all_tables = list(SAMPLE_SIZES.keys())
    for table in all_tables:
        print(f"\n  {table}:")
        for metric, attr in [("  rows", "row_count"), ("  duration (s)", "duration_s"),
                              ("  LLM calls", "llm_calls"), ("  in tokens", "input_tokens"),
                              ("  out tokens", "output_tokens"), ("  cost ($)", "cost_usd")]:
            print(f"  {metric:<28}", end="")
            for rep in reports:
                tr = rep.tables.get(table)
                val = getattr(tr, attr, 0) if tr else 0
                if attr == "cost_usd":
                    print(f"  {val:>{col-2}.4f}", end="")
                elif isinstance(val, float):
                    print(f"  {val:>{col-2}.1f}", end="")
                else:
                    print(f"  {val:>{col-2},}", end="")
            print()

    # Totals
    print(f"\n{'─'*70}")
    for metric, fn in [
        ("Total LLM calls",   lambda r: sum(t.llm_calls    for t in r.tables.values())),
        ("Total in tokens",   lambda r: sum(t.input_tokens  for t in r.tables.values())),
        ("Total out tokens",  lambda r: sum(t.output_tokens for t in r.tables.values())),
        ("Total cost ($)",    lambda r: r.estimated_cost_usd),
    ]:
        print(f"{'  '+metric:<30}", end="")
        for rep in reports:
            val = fn(rep)
            if "cost" in metric:
                print(f"  {val:>{col-2}.4f}", end="")
            else:
                print(f"  {val:>{col-2},}", end="")
        print()

    # Speed winner
    fastest_idx = elapsed.index(min(elapsed))
    speedup = max(elapsed) / min(elapsed)
    print(f"\n  ⚡ Fastest: {labels[fastest_idx]} ({speedup:.1f}× faster)")
    cheapest_idx = [r["report"].total_cost_usd for r in results].index(
        min(r["report"].total_cost_usd for r in results))
    print(f"  💰 Cheapest: {labels[cheapest_idx]}")
    print(f"{'='*70}\n")


def main():
    conn_str = build_connection_string()
    print(f"Connecting to PostgreSQL: {conn_str}")
    engine = create_engine(conn_str)

    print("\nSetting up schema...")
    create_demo_schema(engine)

    loader  = DatabaseSchemaLoader(engine)
    schemas = loader.load_schemas()
    print(f"Inferred {len(schemas)} tables: {', '.join(schemas.keys())}")
    print(f"\nSample sizes: {SAMPLE_SIZES}")
    print(f"Total rows: {sum(SAMPLE_SIZES.values()):,}")

    results = []
    for m in MODELS:
        output_dir = os.path.join(EXAMPLE_DIR, "output", m["output_subdir"])
        os.makedirs(output_dir, exist_ok=True)
        elapsed, report = run_model(
            m["label"], m["config"], m["kwargs"], output_dir, schemas
        )
        results.append({"label": m["label"], "elapsed": elapsed,
                        "report": report, "output_dir": output_dir})
        report.print_summary()

    print_comparison(results)

    for r in results:
        quality_spot_check(r["output_dir"], r["label"])

    print(f"\nHTML reports saved to:")
    for r in results:
        print(f"  {r['output_dir']}/run_report_*.html")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Example: large dataset generation against PostgreSQL.

Demonstrates the chunked-generation and code-gen features introduced for large
datasets:
- auto mode: direct (≤500 rows) vs code-gen (>500 rows)
- batch_size: max rows per LLM call in direct mode
- code-gen mode: LLM classifies columns, writes Python generators for simple
  columns (IDs, dates, enums, phone numbers …), generates only semantic/
  free-text columns via LLM — dramatically fewer API calls at scale
- multi-table memory optimisation: each table is flushed to disk as soon as it
  is complete; only FK columns are kept in RAM for child-table generation

Schema — e-commerce (5 tables, FK chain):
    customers (200 rows)  →  orders (1 000 rows)  →  order_items (2 000 rows)
    products  (100 rows)  ↗                        ↗
    reviews   (600 rows)   ← products + customers

Default sample sizes fit comfortably within the 10 000 output tokens/minute
limit on free/tier-1 Anthropic accounts.  Scale up SCALE_FACTOR at the top of
main() for larger runs on higher-tier accounts.

With auto mode:
  customers / products → direct  (≤500 rows, chunked LLM calls, batch_size=50)
  orders / reviews     → code-gen (>500 rows, 1 analysis call + semantic cols)
  order_items          → code-gen (>500 rows, streaming to disk)

Requirements:
    pip install syda sqlalchemy psycopg2-binary python-dotenv

Environment variables (set in shell or .env file):
    DB_USER      (default: postgres)
    DB_PASSWORD  (default: postgres)
    DB_HOST      (default: localhost)
    DB_PORT      (default: 5432)
    DB_NAME      (default: syda_large_dataset_demo)
    ANTHROPIC_API_KEY  (or OPENAI_API_KEY — change model_config below)
"""

import os
import sys
import time

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def build_connection_string() -> str:
    user     = os.getenv("DB_USER",     "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "5432")
    db       = os.getenv("DB_NAME",     "syda_large_dataset_demo")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------

def create_demo_schema(engine) -> None:
    """Drop and recreate the e-commerce demo tables in FK-safe order."""
    with engine.connect() as conn:
        for table in ["order_items", "reviews", "orders", "products", "customers"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        conn.commit()

        conn.execute(text("""
            CREATE TABLE customers (
                customer_id   SERIAL PRIMARY KEY,
                first_name    TEXT NOT NULL,
                last_name     TEXT NOT NULL,
                email         TEXT NOT NULL UNIQUE,
                phone         TEXT,
                city          TEXT,
                country       TEXT,
                signup_date   DATE NOT NULL,
                loyalty_tier  TEXT CHECK (loyalty_tier IN ('Bronze','Silver','Gold','Platinum'))
            )"""))

        conn.execute(text("""
            CREATE TABLE products (
                product_id    SERIAL PRIMARY KEY,
                product_name  TEXT NOT NULL,
                category      TEXT NOT NULL,
                brand         TEXT,
                unit_price    NUMERIC(10,2) NOT NULL,
                stock_qty     INTEGER NOT NULL DEFAULT 0,
                description   TEXT
            )"""))

        conn.execute(text("""
            CREATE TABLE orders (
                order_id        SERIAL PRIMARY KEY,
                customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
                order_date      DATE NOT NULL,
                status          TEXT CHECK (status IN ('pending','processing','shipped','delivered','cancelled')),
                shipping_method TEXT,
                total_amount    NUMERIC(12,2) NOT NULL
            )"""))

        conn.execute(text("""
            CREATE TABLE order_items (
                item_id     SERIAL PRIMARY KEY,
                order_id    INTEGER NOT NULL REFERENCES orders(order_id),
                product_id  INTEGER NOT NULL REFERENCES products(product_id),
                quantity    INTEGER NOT NULL CHECK (quantity > 0),
                unit_price  NUMERIC(10,2) NOT NULL,
                discount    NUMERIC(5,2) DEFAULT 0.00
            )"""))

        conn.execute(text("""
            CREATE TABLE reviews (
                review_id    SERIAL PRIMARY KEY,
                product_id   INTEGER NOT NULL REFERENCES products(product_id),
                customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
                rating       INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                review_text  TEXT,
                review_date  DATE NOT NULL
            )"""))

        conn.commit()
    print("  E-commerce schema ready (5 tables).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    example_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(example_dir, "output")
    conn_str    = build_connection_string()

    # ------------------------------------------------------------------
    # Step 1: set up demo schema in PostgreSQL
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Connect and create schema")
    print("=" * 60)
    print(f"Connecting to: {conn_str}")
    engine = create_engine(conn_str)
    create_demo_schema(engine)

    # ------------------------------------------------------------------
    # Step 2: infer schemas from the live database
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Infer schemas from PostgreSQL")
    print("=" * 60)
    loader  = DatabaseSchemaLoader(engine)
    schemas = loader.load_schemas()
    print(f"  Inferred {len(schemas)} tables: {', '.join(schemas.keys())}")

    # ------------------------------------------------------------------
    # Step 3: configure the generator
    #
    # ModelConfig options for large datasets:
    #   generation_mode='auto'  → direct for ≤500 rows, code-gen for >500
    #   batch_size=50           → max rows per LLM call in direct mode
    #   max_retries=3           → exponential-backoff retries per chunk
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Configure generator (auto mode, batch_size=50)")
    print("=" * 60)

    generator = SyntheticDataGenerator(
        model_config=ModelConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
            temperature=0.8,
            max_tokens=8192,
            generation_mode="auto",   # direct ≤500, code-gen >500
            batch_size=50,            # max rows per LLM call (direct mode)
            max_retries=3,            # retries on transient API errors
        )
    )

    # Sample sizes chosen to exercise both modes in the same run.
    # Scale these up for your own Anthropic tier — these defaults target
    # the free/tier-1 limit of 10 000 output tokens/minute comfortably.
    #   customers (200)  → direct mode  (4 chunks of 50)
    #   products  (100)  → direct mode  (2 chunks of 50)
    #   orders    (1000) → code-gen mode (1 analysis call + semantic cols only)
    #   order_items(2000)→ code-gen mode + streaming to disk
    #   reviews   (600)  → code-gen mode
    sample_sizes = {
        "customers":    200,
        "products":     100,
        "orders":     1_000,
        "order_items": 2_000,
        "reviews":      600,
    }

    prompts = {
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

    # ------------------------------------------------------------------
    # Step 4: generate — chunks stream to disk, RAM stays bounded
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Generate synthetic data")
    print("  Tables ≤500 rows  → direct mode  (chunked LLM calls)")
    print("  Tables >500 rows  → code-gen mode (LLM writes Python functions,")
    print("                       only semantic columns use LLM at runtime)")
    print("  output_dir set   → each table flushed to disk as soon as done")
    print("=" * 60)
    print()

    t0 = time.time()
    results = generator.generate_for_schemas(
        schemas=schemas,
        sample_sizes=sample_sizes,
        prompts=prompts,
        output_dir=output_dir,
        output_format="csv",
        batch_size=50,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"Generation complete in {elapsed:.1f}s")
    print("=" * 60)
    import pandas as pd
    for table in ["customers", "products", "orders", "order_items", "reviews"]:
        csv = os.path.join(output_dir, f"{table}.csv")
        if os.path.exists(csv):
            row_count = sum(1 for _ in open(csv)) - 1  # fast line count minus header
            print(f"  {table:<12}: {row_count:,} rows  (CSV: {csv})")
        else:
            print(f"  {table:<12}: CSV not found")

    # ------------------------------------------------------------------
    # Step 5: load the CSVs back and write to PostgreSQL
    #
    # For very large tables the results dict may hold only FK columns
    # (the full data was streamed to disk). We reload from CSV so the
    # database gets every row.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Write data back to PostgreSQL")
    print("=" * 60)

    import pandas as pd
    from sqlalchemy import inspect as sa_inspect

    inspector  = sa_inspect(engine)
    table_order = ["customers", "products", "orders", "order_items", "reviews"]

    with engine.connect() as conn:
        for table in table_order:
            csv_path = os.path.join(output_dir, f"{table}.csv")
            if not os.path.exists(csv_path):
                print(f"  {table}: CSV not found, skipping")
                continue

            df = pd.read_csv(csv_path)

            # Keep only columns that actually exist in the target table
            db_cols = [c["name"] for c in inspector.get_columns(table)]
            df = df[[c for c in db_cols if c in df.columns]]

            # Fix duplicates in unique-constrained columns by renaming rather than
            # dropping so FK relationships remain intact (all PKs preserved).
            unique_cols = [
                uc["column_names"][0]
                for uc in inspector.get_unique_constraints(table)
                if len(uc.get("column_names", [])) == 1
                   and uc["column_names"][0] in df.columns
            ]
            for col in unique_cols:
                seen: set = set()
                for i, val in enumerate(df[col]):
                    candidate = str(val)
                    if candidate in seen:
                        # Append row-index suffix to make it unique
                        df.at[df.index[i], col] = f"{candidate}_{i}"
                    seen.add(str(df.at[df.index[i], col]))

            # Insert WITH explicit PK values so FK references in child tables
            # stay consistent (we keep the LLM-generated IDs as-is; tables were
            # just DROPped + recreated so sequences are reset to 1).
            # After insert, advance the sequence past the max inserted ID.
            pk_col = f"{table[:-1]}_id" if table != "order_items" else "item_id"

            df.to_sql(table, con=conn, if_exists="append", index=False,
                      method="multi", chunksize=1000)
            conn.commit()

            # Advance serial sequence so future inserts don't collide
            if pk_col in df.columns:
                max_id = int(df[pk_col].max())
                seq_name = f"{table}_{pk_col}_seq"
                conn.execute(text(f"SELECT setval('{seq_name}', {max_id}, true)"))
                conn.commit()

            print(f"  {table:<12}: {len(df):,} rows inserted")

    print("\nAll done. PostgreSQL contains a full synthetic e-commerce dataset.")
    print(f"CSVs saved to: {output_dir}/")


if __name__ == "__main__":
    main()

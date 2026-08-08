#!/usr/bin/env python
"""
Full-scale benchmark: Grok-4.3 vs Claude Sonnet 4.6 at 18,700 rows.

Runs Claude Sonnet 4.6 on the same schema and sample sizes used in the
Grok-4.3 production run (9,184s / $0.95), then prints a side-by-side
comparison using the saved Grok-4.3 run report.

Usage:
    python examples/large_dataset/benchmark_fullscale.py

Grok-4.3 results are read from the existing run report in output/.
Claude Sonnet 4.6 results are generated fresh.
"""

import os
import sys
import time
import re

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_SIZES = {
    "customers":     500,
    "products":      200,
    "orders":      5_000,
    "order_items": 10_000,
    "reviews":      3_000,
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
                first_name TEXT NOT NULL, last_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE, phone TEXT,
                city TEXT, country TEXT, signup_date DATE NOT NULL,
                loyalty_tier TEXT CHECK (loyalty_tier IN ('Bronze','Silver','Gold','Platinum'))
            )"""))
        conn.execute(text("""
            CREATE TABLE products (
                product_id SERIAL PRIMARY KEY,
                product_name TEXT NOT NULL, category TEXT NOT NULL,
                brand TEXT, unit_price NUMERIC(10,2) NOT NULL,
                stock_qty INTEGER NOT NULL DEFAULT 0, description TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE orders (
                order_id SERIAL PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                order_date DATE NOT NULL,
                status TEXT CHECK (status IN ('pending','processing','shipped','delivered','cancelled')),
                shipping_method TEXT, total_amount NUMERIC(12,2) NOT NULL
            )"""))
        conn.execute(text("""
            CREATE TABLE order_items (
                item_id SERIAL PRIMARY KEY,
                order_id INTEGER NOT NULL REFERENCES orders(order_id),
                product_id INTEGER NOT NULL REFERENCES products(product_id),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price NUMERIC(10,2) NOT NULL, discount NUMERIC(5,2) DEFAULT 0.00
            )"""))
        conn.execute(text("""
            CREATE TABLE reviews (
                review_id SERIAL PRIMARY KEY,
                product_id INTEGER NOT NULL REFERENCES products(product_id),
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                review_text TEXT, review_date DATE NOT NULL
            )"""))
        conn.commit()


def parse_html_report(path):
    """Extract per-table metrics and totals from a syda HTML run report."""
    html = open(path).read()
    pattern = (
        r'<td><a[^>]*>(\w+)</a></td>\s*'
        r'<td[^>]*>([\d,]+)</td>\s*'
        r'<td[^>]*><span[^>]*>(\w+)</span></td>\s*'
        r'<td[^>]*>([\d,]+)</td>\s*'
        r'<td[^>]*>([\d,]+)</td>\s*'
        r'<td[^>]*>([\d,]+)</td>\s*'
        r'<td[^>]*>([^<]+)</td>'
    )
    tables = {}
    for m in re.finditer(pattern, html, re.DOTALL):
        name, rows, mode, calls, in_tok, out_tok, cost = m.groups()
        cost_val = float(re.search(r'[\d.]+', cost).group()) if re.search(r'[\d.]+', cost) else 0.0
        tables[name] = {
            'rows': int(rows.replace(',', '')), 'mode': mode,
            'calls': int(calls.replace(',', '')),
            'in_tok': int(in_tok.replace(',', '')),
            'out_tok': int(out_tok.replace(',', '')),
            'cost': cost_val,
        }
    costs = re.findall(r'\$([\d.]+)', html)
    total_cost = float(costs[0]) if costs else sum(t['cost'] for t in tables.values())
    times = re.findall(r'([\d.]+)s<', html)
    total_time = float(times[0]) if times else 0.0
    return tables, total_time, total_cost


def print_comparison(g_tables, g_time, g_cost, c_tables, c_time, c_cost):
    tables = list(SAMPLE_SIZES.keys())
    W = 18
    print(f"\n{'='*68}")
    print(f"  FULL-SCALE BENCHMARK  (18,700 rows)")
    print(f"  Grok-4.3  vs  Claude Sonnet 4.6")
    print(f"  Same schema · same prompts · same sample sizes")
    print(f"{'='*68}")
    print(f"\n{'Metric':<32} {'Grok-4.3':>{W}} {'Sonnet 4.6':>{W}}")
    print('─' * 68)

    for tbl in tables:
        gd = g_tables.get(tbl, {}); cd = c_tables.get(tbl, {})
        print(f"\n  {tbl}  ({SAMPLE_SIZES[tbl]:,} rows · {gd.get('mode','?')} / {cd.get('mode','?')})")
        for label, gv, cv in [
            ('LLM calls',      gd.get('calls',0),   cd.get('calls',0)),
            ('Input tokens',   gd.get('in_tok',0),  cd.get('in_tok',0)),
            ('Output tokens',  gd.get('out_tok',0), cd.get('out_tok',0)),
        ]:
            print(f"    {label:<28} {gv:>{W},} {cv:>{W},}")
        print(f"    {'Cost':<28} ${gd.get('cost',0):>{W-1}.4f} ${cd.get('cost',0):>{W-1}.4f}")

    g_tc = sum(d['calls']   for d in g_tables.values())
    c_tc = sum(d['calls']   for d in c_tables.values())
    g_ti = sum(d['in_tok']  for d in g_tables.values())
    c_ti = sum(d['in_tok']  for d in c_tables.values())
    g_to = sum(d['out_tok'] for d in g_tables.values())
    c_to = sum(d['out_tok'] for d in c_tables.values())
    g_co = sum(d['cost']    for d in g_tables.values())
    c_co = sum(d['cost']    for d in c_tables.values())

    print(f"\n{'─'*68}")
    print(f"  {'Total LLM calls':<30} {g_tc:>{W},} {c_tc:>{W},}")
    print(f"  {'Total input tokens':<30} {g_ti:>{W},} {c_ti:>{W},}")
    print(f"  {'Total output tokens':<30} {g_to:>{W},} {c_to:>{W},}")
    print(f"  {'Total cost':<30} ${g_co:>{W-1}.4f} ${c_co:>{W-1}.4f}")
    print(f"  {'Total time':<30} {g_time:>{W}.0f}s {c_time:>{W}.0f}s")
    print(f"  {'Total time (hrs)':<30} {g_time/3600:>{W}.2f}h {c_time/3600:>{W}.2f}h")
    print(f"{'='*68}")

    faster  = 'Grok-4.3' if g_time < c_time else 'Sonnet 4.6'
    speedup = max(g_time, c_time) / min(g_time, c_time)
    cheaper = 'Grok-4.3' if g_co < c_co else 'Sonnet 4.6'
    savings = max(g_co, c_co) / min(g_co, c_co)
    print(f"\n  Fastest:  {faster}  ({speedup:.1f}× faster)")
    print(f"  Cheapest: {cheaper}  ({savings:.1f}× cheaper)")
    print(f"{'='*68}\n")


def quality_spot_check(g_dir, c_dir):
    import pandas as pd
    print(f"\n{'='*68}")
    print(f"  QUALITY SPOT-CHECK")
    print(f"{'='*68}")

    # review_text — most revealing semantic column
    print(f"\n  review_text samples (4 random rows each):")
    for label, path in [('Grok-4.3', os.path.join(g_dir, 'reviews.csv')),
                        ('Sonnet 4.6', os.path.join(c_dir, 'reviews.csv'))]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"\n  [{label}]")
        for _, row in df.sample(4, random_state=42).iterrows():
            txt = str(row.get('review_text', ''))[:115]
            print(f"    [{row.get('rating','?')}★] {txt}")

    # shipping_method distribution
    print(f"\n  shipping_method distribution (orders):")
    for label, path in [('Grok-4.3', os.path.join(g_dir, 'orders.csv')),
                        ('Sonnet 4.6', os.path.join(c_dir, 'orders.csv'))]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"\n  [{label}]")
        if 'shipping_method' in df.columns:
            for val, cnt in df['shipping_method'].value_counts().items():
                print(f"    {str(val):<30} {cnt:>5,} ({cnt/len(df)*100:.1f}%)")

    # loyalty_tier distribution
    print(f"\n  loyalty_tier distribution (customers, target: 40/30/20/10):")
    for label, path in [('Grok-4.3', os.path.join(g_dir, 'customers.csv')),
                        ('Sonnet 4.6', os.path.join(c_dir, 'customers.csv'))]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"\n  [{label}]")
        if 'loyalty_tier' in df.columns:
            for val, cnt in df['loyalty_tier'].value_counts().items():
                print(f"    {str(val):<30} {cnt:>5,} ({cnt/len(df)*100:.1f}%)")


def main():
    # ── Grok-4.3: load existing run report ───────────────────────────────────
    grok_report_dir = os.path.join(EXAMPLE_DIR, "output")
    grok_reports = sorted([f for f in os.listdir(grok_report_dir)
                           if f.startswith("run_report_") and f.endswith(".html")])
    if not grok_reports:
        print("[ERROR] No Grok-4.3 run report found in output/. Run example_large_dataset_postgres.py first.")
        sys.exit(1)
    grok_report_path = os.path.join(grok_report_dir, grok_reports[-1])
    print(f"Reading Grok-4.3 results from: {grok_report_path}")
    g_tables, g_time, g_cost = parse_html_report(grok_report_path)
    # Override with known timing from production run
    g_time = 9184.5

    # ── Claude Sonnet 4.6: fresh run ─────────────────────────────────────────
    claude_output_dir = os.path.join(EXAMPLE_DIR, "output", "benchmark_sonnet46_fullscale")
    os.makedirs(claude_output_dir, exist_ok=True)

    conn_str = build_connection_string()
    print(f"\nConnecting to PostgreSQL: {conn_str}")
    engine = create_engine(conn_str)
    print("Recreating demo schema...")
    create_demo_schema(engine)

    loader  = DatabaseSchemaLoader(engine)
    schemas = loader.load_schemas()
    print(f"Inferred {len(schemas)} tables: {', '.join(schemas.keys())}")

    print(f"\n{'='*60}")
    print("Running Claude Sonnet 4.6 — 18,700 rows")
    print(f"{'='*60}\n")

    generator = SyntheticDataGenerator(
        model_config=ModelConfig(
            provider="anthropic",
            model_name="claude-sonnet-5",
            temperature=0.8,
            max_tokens=16384,
            generation_mode="auto",
            batch_size=50,
            max_retries=3,
        )
    )

    t0 = time.time()
    generator.generate_for_schemas(
        schemas=schemas,
        sample_sizes=SAMPLE_SIZES,
        prompts=PROMPTS,
        output_dir=claude_output_dir,
        output_format="csv",
        batch_size=50,
    )
    c_time = time.time() - t0

    generator.last_report.print_summary()
    c_report_path = sorted([
        os.path.join(claude_output_dir, f)
        for f in os.listdir(claude_output_dir)
        if f.startswith("run_report_")
    ])[-1]
    c_tables, _, c_cost = parse_html_report(c_report_path)
    c_cost = sum(t['cost'] for t in c_tables.values())

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(g_tables, g_time, g_cost, c_tables, c_time, c_cost)
    quality_spot_check(grok_report_dir, claude_output_dir)

    print(f"\nHTML reports:")
    print(f"  Grok-4.3   : {grok_report_path}")
    print(f"  Sonnet 4.6 : {c_report_path}")


if __name__ == "__main__":
    main()

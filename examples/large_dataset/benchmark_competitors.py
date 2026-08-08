#!/usr/bin/env python
"""
Competitor benchmark: Syda vs Misata vs Faker vs Mimesis

All tools generate the same 5-table e-commerce schema:
  customers (500) → orders (1,000) → order_items (2,000)
  products  (200) ↗                ↗
  reviews   (600) ← products + customers

Metrics collected per tool:
  - Wall-clock time
  - LLM API cost (Syda only; others are free)
  - FK integrity: % child FKs referencing valid parents
  - Cardinality realism: unique values in categorical columns
  - Distribution accuracy: loyalty_tier vs declared 40/30/20/10% target

Usage:
    python examples/large_dataset/benchmark_competitors.py
"""

import os
import sys
import time
import random

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXAMPLE_DIR, "output", "benchmark_competitors")
os.makedirs(OUT, exist_ok=True)

SIZES = {
    "customers":   500,
    "products":    200,
    "orders":    1_000,
    "reviews":     600,
    "order_items": 2_000,
}
TOTAL = sum(SIZES.values())

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYDA — read results from saved run reports
# ─────────────────────────────────────────────────────────────────────────────

def load_syda_results():
    import re
    results = {}

    runs = {
        "Syda / Grok-4.3": {
            "report": os.path.join(EXAMPLE_DIR, "output", "run_report_20260630_080019.html"),
            "csv_dir": os.path.join(EXAMPLE_DIR, "output"),
            "time": 9184.5,
            "cost": 0.95,
        },
        "Syda / Sonnet 4.6": {
            "report": os.path.join(EXAMPLE_DIR, "output", "benchmark_sonnet46_fullscale",
                                   "run_report_20260702_142055.html"),
            "csv_dir": os.path.join(EXAMPLE_DIR, "output", "benchmark_sonnet46_fullscale"),
            "time": 2781.4,
            "cost": 2.38,
        },
    }

    for label, info in runs.items():
        dfs = {}
        for tbl in SIZES:
            p = os.path.join(info["csv_dir"], f"{tbl}.csv")
            if os.path.exists(p):
                dfs[tbl] = pd.read_csv(p)
        results[label] = {
            "time": info["time"],
            "cost": info["cost"],
            "dfs": dfs,
            "needs_real_data": False,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. MISATA
# ─────────────────────────────────────────────────────────────────────────────

def run_misata():
    import misata

    schema = misata.from_dict_schema({
        "customers": {
            "__rows__": SIZES["customers"],
            "customer_id": {"type": "integer", "primary_key": True},
            "first_name":  {"type": "text", "text_type": "first_name"},
            "last_name":   {"type": "text", "text_type": "last_name"},
            "email":       {"type": "text", "text_type": "email", "unique": True},
            "phone":       {"type": "text", "text_type": "phone"},
            "city":        {"type": "text", "text_type": "city"},
            "country":     {"type": "text", "text_type": "country"},
            "signup_date": {"type": "date"},
            "loyalty_tier": {
                "type": "categorical",
                "choices": ["Bronze", "Silver", "Gold", "Platinum"],
                "probabilities": [0.40, 0.30, 0.20, 0.10],
            },
        },
        "products": {
            "__rows__": SIZES["products"],
            "product_id":   {"type": "integer", "primary_key": True},
            "product_name": {"type": "text"},
            "category":     {"type": "categorical",
                             "choices": ["Electronics","Clothing","Home Goods","Sports"]},
            "brand":        {"type": "text"},
            "unit_price":   {"type": "float", "min": 5.0, "max": 500.0},
            "stock_qty":    {"type": "integer", "min": 0, "max": 500},
            "description":  {"type": "text"},
        },
        "orders": {
            "__rows__": SIZES["orders"],
            "order_id":      {"type": "integer", "primary_key": True},
            "customer_id":   {"type": "foreign_key", "references": "customers.customer_id"},
            "order_date":    {"type": "date"},
            "status": {
                "type": "categorical",
                "choices": ["pending","processing","shipped","delivered","cancelled"],
                "probabilities": [0.10, 0.20, 0.25, 0.40, 0.05],
            },
            "shipping_method": {"type": "categorical",
                                "choices": ["Standard","Express","Overnight","Two-Day","Economy"],
                                "probabilities": [0.30, 0.25, 0.15, 0.20, 0.10]},
            "total_amount":  {"type": "float", "min": 5.0, "max": 2000.0},
        },
        "reviews": {
            "__rows__": SIZES["reviews"],
            "review_id":   {"type": "integer", "primary_key": True},
            "product_id":  {"type": "foreign_key", "references": "products.product_id"},
            "customer_id": {"type": "foreign_key", "references": "customers.customer_id"},
            "rating":      {"type": "integer", "min": 1, "max": 5},
            "review_text": {"type": "text"},
            "review_date": {"type": "date"},
        },
        "order_items": {
            "__rows__": SIZES["order_items"],
            "item_id":    {"type": "integer", "primary_key": True},
            "order_id":   {"type": "foreign_key", "references": "orders.order_id"},
            "product_id": {"type": "foreign_key", "references": "products.product_id"},
            "quantity":   {"type": "integer", "min": 1, "max": 5},
            "unit_price": {"type": "float", "min": 5.0, "max": 500.0},
            "discount":   {"type": "float", "min": 0.0, "max": 0.20},
        },
    }, seed=42)

    t0 = time.time()
    tables = misata.generate_from_schema(schema)
    elapsed = time.time() - t0

    # Save CSVs
    misata_dir = os.path.join(OUT, "misata")
    os.makedirs(misata_dir, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(os.path.join(misata_dir, f"{name}.csv"), index=False)

    return {
        "time": elapsed,
        "cost": 0.0,
        "dfs": {k: v for k, v in tables.items()},
        "needs_real_data": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FAKER
# ─────────────────────────────────────────────────────────────────────────────

def run_faker():
    from faker import Faker
    fake = Faker()
    Faker.seed(42)
    random.seed(42)

    t0 = time.time()

    customers = pd.DataFrame([{
        "customer_id": i + 1,
        "first_name":  fake.first_name(),
        "last_name":   fake.last_name(),
        "email":       fake.unique.email(),
        "phone":       fake.phone_number(),
        "city":        fake.city(),
        "country":     fake.country(),
        "signup_date": fake.date_between(start_date="-5y"),
        "loyalty_tier": random.choices(
            ["Bronze","Silver","Gold","Platinum"], weights=[40,30,20,10])[0],
    } for i in range(SIZES["customers"])])

    products = pd.DataFrame([{
        "product_id":   i + 1,
        "product_name": fake.catch_phrase(),
        "category":     random.choice(["Electronics","Clothing","Home Goods","Sports"]),
        "brand":        fake.company(),
        "unit_price":   round(random.uniform(5, 500), 2),
        "stock_qty":    random.randint(0, 500),
        "description":  fake.sentence(nb_words=12),
    } for i in range(SIZES["products"])])

    cust_ids = customers["customer_id"].tolist()
    orders = pd.DataFrame([{
        "order_id":       i + 1,
        "customer_id":    random.choice(cust_ids),
        "order_date":     fake.date_between(start_date="-2y"),
        "status":         random.choices(
            ["pending","processing","shipped","delivered","cancelled"],
            weights=[10,20,25,40,5])[0],
        "shipping_method": random.choices(
            ["Standard","Express","Overnight","Two-Day","Economy"],
            weights=[30,25,15,20,10])[0],
        "total_amount":   round(random.uniform(5, 2000), 2),
    } for i in range(SIZES["orders"])])

    prod_ids = products["product_id"].tolist()
    reviews = pd.DataFrame([{
        "review_id":   i + 1,
        "product_id":  random.choice(prod_ids),
        "customer_id": random.choice(cust_ids),
        "rating":      random.choices([1,2,3,4,5], weights=[5,10,20,35,30])[0],
        "review_text": fake.paragraph(nb_sentences=2),
        "review_date": fake.date_between(start_date="-2y"),
    } for i in range(SIZES["reviews"])])

    order_ids = orders["order_id"].tolist()
    order_items = pd.DataFrame([{
        "item_id":    i + 1,
        "order_id":   random.choice(order_ids),
        "product_id": random.choice(prod_ids),
        "quantity":   random.randint(1, 5),
        "unit_price": round(random.uniform(5, 500), 2),
        "discount":   round(random.uniform(0, 0.20), 2),
    } for i in range(SIZES["order_items"])])

    elapsed = time.time() - t0

    faker_dir = os.path.join(OUT, "faker")
    os.makedirs(faker_dir, exist_ok=True)
    dfs = {"customers": customers, "products": products, "orders": orders,
           "reviews": reviews, "order_items": order_items}
    for name, df in dfs.items():
        df.to_csv(os.path.join(faker_dir, f"{name}.csv"), index=False)

    return {"time": elapsed, "cost": 0.0, "dfs": dfs, "needs_real_data": False}


# ─────────────────────────────────────────────────────────────────────────────
# 4. MIMESIS
# ─────────────────────────────────────────────────────────────────────────────

def run_mimesis():
    from mimesis import Person, Address, Finance, Text, Datetime
    from mimesis.locales import Locale

    person   = Person(Locale.EN)
    address  = Address(Locale.EN)
    finance  = Finance(Locale.EN)
    text     = Text(Locale.EN)
    dt       = Datetime(Locale.EN)
    random.seed(42)

    t0 = time.time()

    customers = pd.DataFrame([{
        "customer_id": i + 1,
        "first_name":  person.first_name(),
        "last_name":   person.last_name(),
        "email":       person.email(unique=True),
        "phone":       person.phone_number(),
        "city":        address.city(),
        "country":     address.country(),
        "signup_date": dt.date(start=2019, end=2025),
        "loyalty_tier": random.choices(
            ["Bronze","Silver","Gold","Platinum"], weights=[40,30,20,10])[0],
    } for i in range(SIZES["customers"])])

    products = pd.DataFrame([{
        "product_id":   i + 1,
        "product_name": text.title(),
        "category":     random.choice(["Electronics","Clothing","Home Goods","Sports"]),
        "brand":        finance.company(),
        "unit_price":   round(random.uniform(5, 500), 2),
        "stock_qty":    random.randint(0, 500),
        "description":  text.sentence(),
    } for i in range(SIZES["products"])])

    cust_ids = customers["customer_id"].tolist()
    orders = pd.DataFrame([{
        "order_id":        i + 1,
        "customer_id":     random.choice(cust_ids),
        "order_date":      dt.date(start=2023, end=2025),
        "status":          random.choices(
            ["pending","processing","shipped","delivered","cancelled"],
            weights=[10,20,25,40,5])[0],
        "shipping_method": random.choices(
            ["Standard","Express","Overnight","Two-Day","Economy"],
            weights=[30,25,15,20,10])[0],
        "total_amount":    round(random.uniform(5, 2000), 2),
    } for i in range(SIZES["orders"])])

    prod_ids = products["product_id"].tolist()
    reviews = pd.DataFrame([{
        "review_id":   i + 1,
        "product_id":  random.choice(prod_ids),
        "customer_id": random.choice(cust_ids),
        "rating":      random.choices([1,2,3,4,5], weights=[5,10,20,35,30])[0],
        "review_text": text.sentence(),
        "review_date": dt.date(start=2023, end=2025),
    } for i in range(SIZES["reviews"])])

    order_ids = orders["order_id"].tolist()
    order_items = pd.DataFrame([{
        "item_id":    i + 1,
        "order_id":   random.choice(order_ids),
        "product_id": random.choice(prod_ids),
        "quantity":   random.randint(1, 5),
        "unit_price": round(random.uniform(5, 500), 2),
        "discount":   round(random.uniform(0, 0.20), 2),
    } for i in range(SIZES["order_items"])])

    elapsed = time.time() - t0

    mimesis_dir = os.path.join(OUT, "mimesis")
    os.makedirs(mimesis_dir, exist_ok=True)
    dfs = {"customers": customers, "products": products, "orders": orders,
           "reviews": reviews, "order_items": order_items}
    for name, df in dfs.items():
        df.to_csv(os.path.join(mimesis_dir, f"{name}.csv"), index=False)

    return {"time": elapsed, "cost": 0.0, "dfs": dfs, "needs_real_data": False}


# ─────────────────────────────────────────────────────────────────────────────
# Quality evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(label, result):
    dfs = result["dfs"]
    scores = {}

    # 1. FK integrity
    fk_checks = [
        ("orders",      "customer_id", "customers", "customer_id"),
        ("reviews",     "product_id",  "products",  "product_id"),
        ("reviews",     "customer_id", "customers", "customer_id"),
        ("order_items", "order_id",    "orders",    "order_id"),
        ("order_items", "product_id",  "products",  "product_id"),
    ]
    valid, total = 0, 0
    for child, fk_col, parent, pk_col in fk_checks:
        if child in dfs and parent in dfs:
            if fk_col in dfs[child].columns and pk_col in dfs[parent].columns:
                parent_vals = set(dfs[parent][pk_col].unique())
                child_vals  = dfs[child][fk_col].dropna().unique()
                ok = sum(1 for v in child_vals if v in parent_vals)
                valid += ok; total += len(child_vals)
    scores["fk_integrity"] = (valid / total * 100) if total else 0

    # 2. Loyalty tier distribution accuracy (target: 40/30/20/10)
    if "customers" in dfs and "loyalty_tier" in dfs["customers"].columns:
        actual = dfs["customers"]["loyalty_tier"].value_counts(normalize=True)
        targets = {"Bronze": 0.40, "Silver": 0.30, "Gold": 0.20, "Platinum": 0.10}
        mae = sum(abs(actual.get(k, 0) - v) for k, v in targets.items()) / len(targets)
        scores["loyalty_tier_mae"] = mae * 100  # % points off
    else:
        scores["loyalty_tier_mae"] = None

    # 3. Shipping method cardinality (fewer unique values = more realistic)
    if "orders" in dfs and "shipping_method" in dfs["orders"].columns:
        scores["shipping_method_cardinality"] = dfs["orders"]["shipping_method"].nunique()
    else:
        scores["shipping_method_cardinality"] = None

    # 4. Review text uniqueness (% of unique review_text values)
    if "reviews" in dfs and "review_text" in dfs["reviews"].columns:
        n = len(dfs["reviews"])
        u = dfs["reviews"]["review_text"].nunique()
        scores["review_text_uniqueness"] = u / n * 100
    else:
        scores["review_text_uniqueness"] = None

    # 5. Review text avg length (characters)
    if "reviews" in dfs and "review_text" in dfs["reviews"].columns:
        scores["review_text_avg_len"] = dfs["reviews"]["review_text"].dropna().str.len().mean()
    else:
        scores["review_text_avg_len"] = None

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_report(all_results, all_scores):
    tools  = list(all_results.keys())
    W = 18

    print(f"\n{'='*80}")
    print(f"  COMPETITIVE BENCHMARK — {TOTAL:,} rows")
    print(f"  Same 5-table e-commerce schema · same sample sizes")
    print(f"{'='*80}")

    # Speed & cost
    print(f"\n{'── Speed & Cost ':─<80}")
    print(f"{'Tool':<28} {'Time (s)':>{W}} {'Time':>{W}} {'API Cost':>{W}}")
    print('─'*80)
    for tool in tools:
        r = all_results[tool]
        t = r['time']
        mins = f"{t/60:.1f}m" if t >= 60 else f"{t:.1f}s"
        cost = f"${r['cost']:.2f}" if r['cost'] > 0 else "free"
        print(f"  {tool:<26} {t:>{W}.1f} {mins:>{W}} {cost:>{W}}")

    # FK integrity
    print(f"\n{'── FK Integrity & Data Quality ':─<80}")
    print(f"{'Tool':<28} {'FK integrity':>{W}} {'loyalty MAE':>{W}} {'ship. unique':>{W}} {'review len':>{W}}")
    print('─'*80)
    for tool in tools:
        s = all_scores[tool]
        fk  = f"{s['fk_integrity']:.1f}%" if s['fk_integrity'] is not None else "N/A"
        mae = f"{s['loyalty_tier_mae']:.1f}pp" if s['loyalty_tier_mae'] is not None else "N/A"
        sc  = str(s['shipping_method_cardinality']) if s['shipping_method_cardinality'] is not None else "N/A"
        rl  = f"{s['review_text_avg_len']:.0f} chars" if s['review_text_avg_len'] is not None else "N/A"
        print(f"  {tool:<26} {fk:>{W}} {mae:>{W}} {sc:>{W}} {rl:>{W}}")

    # Review samples
    print(f"\n{'── Review Text Quality Samples ':─<80}")
    for tool in tools:
        dfs = all_results[tool]['dfs']
        if 'reviews' in dfs and 'review_text' in dfs['reviews'].columns:
            print(f"\n  [{tool}]")
            sample = dfs['reviews'].dropna(subset=['review_text']).sample(
                min(3, len(dfs['reviews'])), random_state=42)
            for _, row in sample.iterrows():
                print(f"    [{row.get('rating','?')}★] {str(row['review_text'])[:115]}")

    # Capability matrix
    print(f"\n{'── Capability Matrix ':─<80}")
    caps = [
        ("Cold-start (no real data)",  {"Syda / Grok-4.3": "✓", "Syda / Sonnet 4.6": "✓", "Misata": "✓", "Faker": "✓", "Mimesis": "✓"}),
        ("FK referential integrity",   {"Syda / Grok-4.3": "✓", "Syda / Sonnet 4.6": "✓", "Misata": "✓", "Faker": "manual", "Mimesis": "manual"}),
        ("Semantic / narrative text",  {"Syda / Grok-4.3": "LLM", "Syda / Sonnet 4.6": "LLM", "Misata": "grammar", "Faker": "lorem", "Mimesis": "lorem"}),
        ("Unstructured docs (PDF…)",   {"Syda / Grok-4.3": "✓", "Syda / Sonnet 4.6": "✓", "Misata": "✗", "Faker": "✗", "Mimesis": "✗"}),
        ("Outcome curves (KPIs)",      {"Syda / Grok-4.3": "✗", "Syda / Sonnet 4.6": "✗", "Misata": "✓", "Faker": "✗", "Mimesis": "✗"}),
        ("Exact incidence control",    {"Syda / Grok-4.3": "✗", "Syda / Sonnet 4.6": "✗", "Misata": "✓", "Faker": "approx", "Mimesis": "approx"}),
        ("Cross-table rollups",        {"Syda / Grok-4.3": "✗", "Syda / Sonnet 4.6": "✗", "Misata": "✓", "Faker": "✗", "Mimesis": "✗"}),
        ("License",                    {"Syda / Grok-4.3": "MIT", "Syda / Sonnet 4.6": "MIT", "Misata": "OSS", "Faker": "MIT", "Mimesis": "MIT"}),
    ]
    CW = 16
    header = f"{'Capability':<30}" + "".join(f"{t[:CW]:>{CW}}" for t in tools)
    print(f"\n  {header}")
    print("  " + "─" * (30 + CW * len(tools)))
    for cap, vals in caps:
        row = f"  {cap:<30}" + "".join(f"{vals.get(t,'?')[:CW]:>{CW}}" for t in tools)
        print(row)

    print(f"\n{'='*80}")
    fastest = min(tools, key=lambda t: all_results[t]['time'])
    cheapest = min((t for t in tools if all_results[t]['cost'] == 0), key=lambda t: all_results[t]['time'])
    best_fk = max(tools, key=lambda t: all_scores[t]['fk_integrity'])
    print(f"  Fastest overall : {fastest} ({all_results[fastest]['time']:.1f}s)")
    print(f"  Fastest (free)  : {cheapest} ({all_results[cheapest]['time']:.1f}s)")
    print(f"  Best FK integrity: {best_fk} ({all_scores[best_fk]['fk_integrity']:.1f}%)")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = {}
    all_scores  = {}

    # Syda — from saved reports
    print("Loading Syda results from saved run reports...")
    syda = load_syda_results()
    all_results.update(syda)

    # Misata
    print("Running Misata...")
    try:
        all_results["Misata"] = run_misata()
        print(f"  Misata done in {all_results['Misata']['time']:.1f}s")
    except Exception as e:
        print(f"  Misata failed: {e}")

    # Faker
    print("Running Faker...")
    all_results["Faker"] = run_faker()
    print(f"  Faker done in {all_results['Faker']['time']:.2f}s")

    # Mimesis
    print("Running Mimesis...")
    all_results["Mimesis"] = run_mimesis()
    print(f"  Mimesis done in {all_results['Mimesis']['time']:.2f}s")

    # Evaluate quality
    print("\nEvaluating quality metrics...")
    for tool, result in all_results.items():
        all_scores[tool] = evaluate(tool, result)

    print_report(all_results, all_scores)

    print("CSVs saved to:")
    print(f"  {OUT}/misata/, {OUT}/faker/, {OUT}/mimesis/")


if __name__ == "__main__":
    main()

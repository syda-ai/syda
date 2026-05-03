"""
syda_integration.py
===================
Syda Schema Inference Extension — CLI Pipeline

Connects to a database, infers its schema, and generates synthetic data
using Syda's generate_for_schemas(). Uses db_schema_inference.py to
attach create_schemas_from_database() to the Syda generator.

Requirements:
    pip install syda sqlalchemy psycopg2-binary pymysql python-dotenv

Environment variables (.env file):
    ANTHROPIC_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here

Usage:
    python syda_integration.py --db-url sqlite:///healthcare_demo.db --create-demo-sqlite --output-dir synthetic_output
    python syda_integration.py --db-url "postgresql+psycopg2://postgres:password@localhost/healthcare_demo" --output-dir synthetic_output
    python syda_integration.py --db-url "mysql+pymysql://root:password@localhost/healthcare_demo" --output-dir synthetic_output
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine

from schema_modules import (
    verify_connection,
    create_demo_sqlite,
    extract_schema,
    to_syda_dict,
    validate_syda_compatibility,
    save_versioned_output,
)
from db_schema_inference import patch_generator

load_dotenv()


def build_prompts(tables):
    """Returns domain-aware generation prompts for each healthcare table.

    Tables not in the domain map receive a generic fallback prompt.

    Args:
        tables (list[str]): Table names present in the schema.

    Returns:
        dict[str, str]: Mapping of table name to prompt string.
    """
    domain_prompts = {
        "patient": (
            "Generate realistic patient records for a healthcare claims system. "
            "Include diverse ages (18-85), genders (M/F/Other), and realistic names. "
            "Date of birth should be consistent with age."
        ),
        "provider": (
            "Generate realistic healthcare provider records. "
            "Include diverse specialties (Cardiology, General Practice, Oncology, Orthopedics, etc.). "
            "License numbers should follow a realistic format (e.g., LIC-XXXXX)."
        ),
        "diagnosis": (
            "Generate realistic diagnosis records linking patients to providers. "
            "Use realistic ICD-10 diagnosis codes (e.g., I10, E11.9, J45.50). "
            "Visit dates should be within the last 3 years."
        ),
        "claim": (
            "Generate realistic healthcare insurance claims. "
            "Use CPT procedure codes (e.g., 99213, 99214, 93000). "
            "Claim amounts should be realistic medical billing amounts ($50 - $5000). "
            "Submission dates should be within 90 days of the diagnosis visit date."
        ),
        "adjudication": (
            "Generate realistic claim adjudication records. "
            "Mix of Approved (~60%), Partial (~25%), and Denied (~15%) decisions. "
            "Denied claims should have realistic denial reasons. "
            "Approved amounts should be less than or equal to claim amounts."
        ),
        "payment": (
            "Generate realistic payment records for adjudicated claims. "
            "Status should be mostly Paid, some Pending, few NoPay. "
            "Payment amounts should match approved adjudication amounts. "
            "Payment dates should be 5-30 days after adjudication."
        ),
    }

    return {
        table: domain_prompts.get(
            table.lower(),
            f"Generate realistic synthetic data for the {table} table.",
        )
        for table in tables
    }


def build_sample_sizes(tables):
    """Returns row counts per table reflecting real-world healthcare data ratios.

    Args:
        tables (list[str]): Table names present in the schema.

    Returns:
        dict[str, int]: Mapping of table name to row count.
    """
    domain_sizes = {
        "patient":      10,
        "provider":      5,
        "diagnosis":    20,
        "claim":        20,
        "adjudication": 20,
        "payment":      20,
    }
    return {table: domain_sizes.get(table.lower(), 10) for table in tables}


def main():
    parser = argparse.ArgumentParser(
        description="Syda Extension — end-to-end schema inference and synthetic data generation."
    )
    parser.add_argument("--db-url", required=True, help="SQLAlchemy database URL")
    parser.add_argument("--create-demo-sqlite", action="store_true", help="Seed the 6-entity healthcare demo schema")
    parser.add_argument("--output-dir", default="synthetic_output", help="Output directory for generated data (default: synthetic_output)")
    parser.add_argument("--schema-output", default="inferred_schema.json", help="Path for schema JSON output")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"], help="AI provider (default: anthropic)")
    parser.add_argument("--rows", type=int, default=None, help="Override row count for all tables")
    args = parser.parse_args()

    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to a .env file.", file=sys.stderr)
        sys.exit(1)
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to a .env file.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Connect

    print("Step 1: Connecting to database...")
    engine = create_engine(args.db_url)
    verify_connection(engine)
    print("  Connection successful.")

    # Step 2: Seed demo database if requested

    if args.create_demo_sqlite:
        print("Step 2: Creating demo SQLite database...")
        create_demo_sqlite(engine)
        print("  Demo database created.")
    else:
        print("Step 2: Skipped (no --create-demo-sqlite flag).")

    # Step 3: Extract and validate schema

    print("Step 3: Extracting schema...")
    schema = extract_schema(engine)
    syda_schema = to_syda_dict(schema)
    print(f"  Extracted {len(syda_schema)} tables: {', '.join(syda_schema.keys())}")

    print("Step 4: Validating schema compatibility...")
    passed, validation_results = validate_syda_compatibility(syda_schema)
    for r in validation_results:
        icon = "v" if r["status"] == "PASS" else ("!" if r["status"] == "WARN" else "x")
        print(f"  [{icon}] {r['table']}: {r['detail']}")
    if not passed:
        print("WARNING: One or more tables failed validation.")

    # Step 4: Save schema artifacts

    print("Step 5: Saving schema artifacts...")
    syda_path, report_path = save_versioned_output(schema, syda_schema, args.schema_output, validation_results)
    print(f"  Schema     -> {args.schema_output}")
    print(f"  Syda dict  -> {syda_path}")
    print(f"  Validation -> {report_path}")

    # Step 5: Configure Syda generator

    print("Step 6: Configuring Syda generator...")
    try:
        from syda import SyntheticDataGenerator, ModelConfig
    except ImportError:
        print("ERROR: Syda is not installed. Run: pip install syda", file=sys.stderr)
        sys.exit(1)

    model_map = {
        "anthropic": ("anthropic", "claude-haiku-4-5-20251001"),
        "openai":    ("openai",    "gpt-4o-mini"),
    }
    provider_name, model_name = model_map[args.provider]

    generator = SyntheticDataGenerator(
        model_config=ModelConfig(
            provider=provider_name,
            model_name=model_name,
            temperature=0.7,
            max_tokens=8192,
        )
    )
    
    # Attach create_schemas_from_database() to the generator

    patch_generator(generator)
    print(f"  Provider: {provider_name} / {model_name}")

    # Step 6: Infer YAML schemas from database using new method

    print("Step 7: Inferring YAML schemas from database...")
    schema_files = generator.create_schemas_from_database(
        connection_string_or_engine=engine,
        output_dir="generated_schemas",
        format="yaml"
    )
    print(f"  {len(schema_files)} schema files created.")

    # Step 7: Build prompts and sample sizes

    tables = list(schema_files.keys())
    prompts = build_prompts(tables)
    sample_sizes = build_sample_sizes(tables)

    if args.rows:
        sample_sizes = {t: args.rows for t in tables}

    # Step 8: Generate synthetic data

    print(f"\nStep 8: Generating synthetic data -> {args.output_dir}")
    print("  This may take a minute depending on the AI provider...\n")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        generator.generate_for_schemas(
            schemas=schema_files,
            prompts=prompts,
            sample_sizes=sample_sizes,
            output_dir=args.output_dir,
        )
        print("\nDone.")
        print(f"  Synthetic data saved to: {args.output_dir}/")
        for table in tables:
            print(f"  {table}: {sample_sizes[table]} rows")

    except Exception as e:
        print(f"\nERROR during generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
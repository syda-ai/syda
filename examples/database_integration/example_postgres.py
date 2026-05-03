#!/usr/bin/env python
"""
Example: infer schemas from a PostgreSQL database and generate synthetic data,
then write the results back to the database.

This example demonstrates the full cycle against a real PostgreSQL instance:
1. Create the healthcare demo schema in PostgreSQL
2. Infer table schemas automatically via DatabaseSchemaLoader
3. Generate synthetic data and save to CSV
4. Write generated data back to PostgreSQL in FK-safe order

Requirements:
    pip install syda sqlalchemy psycopg2-binary python-dotenv

Environment variables (set in shell or .env):
    DB_USER      (default: postgres)
    DB_PASSWORD  (default: postgres)
    DB_HOST      (default: localhost)
    DB_PORT      (default: 5432)
    DB_NAME      (default: syda_healthcare_demo)
    ANTHROPIC_API_KEY
"""

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig


def build_connection_string():
    user     = os.getenv("DB_USER",     "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "5432")
    db       = os.getenv("DB_NAME",     "syda_healthcare_demo")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def create_demo_schema(engine):
    with engine.connect() as conn:
        # Drop in reverse FK order (children first) so we can re-run cleanly
        for table in ["payment", "adjudication", "claim", "diagnosis", "provider", "patient"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        conn.commit()
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patient (
                patient_id   SERIAL PRIMARY KEY,
                patient_name TEXT NOT NULL,
                age          INTEGER,
                gender       TEXT,
                date_of_birth DATE
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS provider (
                provider_id   SERIAL PRIMARY KEY,
                provider_name TEXT NOT NULL,
                specialty     TEXT,
                license_number TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS diagnosis (
                diagnosis_id   SERIAL PRIMARY KEY,
                patient_id     INTEGER NOT NULL REFERENCES patient(patient_id),
                provider_id    INTEGER NOT NULL REFERENCES provider(provider_id),
                diagnosis_code TEXT,
                visit_date     DATE
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS claim (
                claim_id       SERIAL PRIMARY KEY,
                patient_id     INTEGER NOT NULL REFERENCES patient(patient_id),
                provider_id    INTEGER NOT NULL REFERENCES provider(provider_id),
                diagnosis_id   INTEGER NOT NULL REFERENCES diagnosis(diagnosis_id),
                procedure_code TEXT,
                claim_amount   NUMERIC(10,2),
                submission_date DATE
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS adjudication (
                adjudication_id SERIAL PRIMARY KEY,
                claim_id        INTEGER NOT NULL REFERENCES claim(claim_id),
                decision        TEXT,
                denial_reason   TEXT,
                approved_amount NUMERIC(10,2)
            )"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS payment (
                payment_id     SERIAL PRIMARY KEY,
                claim_id       INTEGER NOT NULL REFERENCES claim(claim_id),
                payment_date   DATE,
                payment_amount NUMERIC(10,2),
                status         TEXT
            )"""))
        conn.commit()


def main():
    example_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir  = os.path.join(example_dir, "output", "postgres")
    conn_str    = build_connection_string()

    # Step 1: Set up demo schema in PostgreSQL
    print(f"Connecting to PostgreSQL: {conn_str}")
    engine = create_engine(conn_str)
    create_demo_schema(engine)
    print("  Healthcare schema ready.")

    # Step 2: Infer schemas
    print("\nInferring schemas from database...")
    loader  = DatabaseSchemaLoader(engine)
    schemas = loader.load_schemas()
    print(f"  Inferred {len(schemas)} tables: {', '.join(schemas.keys())}")

    # Step 3: Generate synthetic data
    generator = SyntheticDataGenerator(
        model_config=ModelConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
            temperature=0.7,
            max_tokens=8192,
        )
    )

    results = generator.generate_for_schemas(
        schemas=schemas,
        sample_sizes={
            "patient": 10, "provider": 5,
            "diagnosis": 20, "claim": 20,
            "adjudication": 20, "payment": 20,
        },
        prompts={
            "patient":      "Generate realistic patient records with diverse ages (18-85) and genders.",
            "provider":     "Generate realistic healthcare providers with diverse specialties.",
            "diagnosis":    "Generate realistic diagnoses using ICD-10 codes (e.g. I10, E11.9).",
            "claim":        "Generate realistic healthcare claims using CPT codes, amounts $50-$5000.",
            "adjudication": "Generate adjudication records: ~60% Approved, ~25% Partial, ~15% Denied.",
            "payment":      "Generate payment records, mostly Paid status.",
        },
        output_dir=output_dir,
    )

    print(f"\nDone. CSV files saved to: {output_dir}/")
    for table, df in results.items():
        print(f"  {table}: {len(df)} rows")

    # Step 4: Write generated data back to PostgreSQL
    print("\nWriting synthetic data back to PostgreSQL...")
    loader.write_to_database(results)
    print("  PostgreSQL updated.")


if __name__ == "__main__":
    main()

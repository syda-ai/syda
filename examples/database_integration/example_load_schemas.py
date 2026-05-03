#!/usr/bin/env python
"""
Example: infer schemas from a database and generate synthetic data directly
(Option A — schema dicts passed in-memory, no intermediate files).

This example demonstrates:
1. Connecting to an existing relational database via DatabaseSchemaLoader
2. Inferring table schemas (columns, types, primary keys, foreign keys) automatically
3. Passing the inferred schemas as dicts directly to generate_for_schemas()
4. Referential integrity preserved automatically across all related tables
"""

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig


def create_demo_db(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("""CREATE TABLE IF NOT EXISTS patient (
            patient_id INTEGER PRIMARY KEY, patient_name TEXT NOT NULL,
            age INTEGER, gender TEXT, date_of_birth DATE)"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS provider (
            provider_id INTEGER PRIMARY KEY, provider_name TEXT NOT NULL,
            specialty TEXT, license_number TEXT)"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS diagnosis (
            diagnosis_id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL,
            provider_id INTEGER NOT NULL, diagnosis_code TEXT, visit_date DATE,
            FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
            FOREIGN KEY (provider_id) REFERENCES provider(provider_id))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS claim (
            claim_id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL,
            provider_id INTEGER NOT NULL, diagnosis_id INTEGER NOT NULL,
            procedure_code TEXT, claim_amount DECIMAL(10,2), submission_date DATE,
            FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
            FOREIGN KEY (provider_id) REFERENCES provider(provider_id),
            FOREIGN KEY (diagnosis_id) REFERENCES diagnosis(diagnosis_id))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS adjudication (
            adjudication_id INTEGER PRIMARY KEY, claim_id INTEGER NOT NULL,
            decision TEXT, denial_reason TEXT, approved_amount DECIMAL(10,2),
            FOREIGN KEY (claim_id) REFERENCES claim(claim_id))"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS payment (
            payment_id INTEGER PRIMARY KEY, claim_id INTEGER NOT NULL,
            payment_date DATE, payment_amount DECIMAL(10,2), status TEXT,
            FOREIGN KEY (claim_id) REFERENCES claim(claim_id))"""))
        conn.commit()
    return engine


def main():
    example_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(example_dir, "healthcare_demo.db")
    output_dir = os.path.join(example_dir, "output", "load_schemas")

    # Step 1: Set up demo database
    print("Setting up healthcare demo database...")
    engine = create_demo_db(db_path)
    print(f"  Database ready: {db_path}")

    # Step 2: Infer schemas directly as dicts (no files written)
    print("\nInferring schemas from database...")
    loader = DatabaseSchemaLoader(engine)
    schemas = loader.load_schemas()
    print(f"  Inferred {len(schemas)} tables: {', '.join(schemas.keys())}")

    # Step 3: Generate synthetic data — schemas passed as dicts
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

    print(f"\nDone. Output saved to: {output_dir}/")
    for table, df in results.items():
        print(f"  {table}: {len(df)} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Bio-IT World 2026 — Live Demo
==============================
Beyond Real Data: How AI-Generated Synthetic Data Enables Safe, Scalable GenAI Deployment

Demonstrates the full Syda pipeline:
  1. Connect to a database and infer schemas automatically
  2. Generate synthetic structured data (patient, provider, diagnosis, claim, payment)
  3. Generate linked unstructured documents (PDFs) for each entity
  4. Write all structured data back to the database in FK-safe order

Run:
    source ../../.venv/bin/activate
    python demo.py

Environment variables (.env):
    DB_USER=postgres
    DB_PASSWORD=postgres
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=syda_bioit_demo
    ANTHROPIC_API_KEY=your_key
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
DEMO_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(DEMO_DIR, "templates")
OUTPUT_DIR   = os.path.join(DEMO_DIR, "output")


def build_connection_string():
    user     = os.getenv("DB_USER",     "postgres")
    password = os.getenv("DB_PASSWORD", "postgres")
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "5432")
    db       = os.getenv("DB_NAME",     "syda_bioit_demo")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


# ── Step 0: Set up demo database ───────────────────────────────────────────────

def setup_database():
    conn_str = build_connection_string()
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS payment"))
        conn.execute(text("DROP TABLE IF EXISTS claim"))
        conn.execute(text("DROP TABLE IF EXISTS diagnosis"))
        conn.execute(text("DROP TABLE IF EXISTS provider"))
        conn.execute(text("DROP TABLE IF EXISTS patient"))

        conn.execute(text("""
            CREATE TABLE patient (
                patient_id   INTEGER PRIMARY KEY,
                patient_name TEXT NOT NULL,
                age          INTEGER,
                gender       TEXT,
                date_of_birth DATE
            )"""))
        conn.execute(text("""
            CREATE TABLE provider (
                provider_id    INTEGER PRIMARY KEY,
                provider_name  TEXT NOT NULL,
                specialty      TEXT,
                license_number TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE diagnosis (
                diagnosis_id   INTEGER PRIMARY KEY,
                patient_id     INTEGER NOT NULL REFERENCES patient(patient_id),
                provider_id    INTEGER NOT NULL REFERENCES provider(provider_id),
                diagnosis_code TEXT,
                visit_date     DATE
            )"""))
        conn.execute(text("""
            CREATE TABLE claim (
                claim_id       INTEGER PRIMARY KEY,
                patient_id     INTEGER NOT NULL REFERENCES patient(patient_id),
                provider_id    INTEGER NOT NULL REFERENCES provider(provider_id),
                diagnosis_id   INTEGER NOT NULL REFERENCES diagnosis(diagnosis_id),
                procedure_code TEXT,
                claim_amount   REAL,
                submission_date DATE
            )"""))
        conn.execute(text("""
            CREATE TABLE payment (
                payment_id     INTEGER PRIMARY KEY,
                claim_id       INTEGER NOT NULL REFERENCES claim(claim_id),
                payment_date   DATE,
                payment_amount REAL,
                status         TEXT
            )"""))
        conn.commit()
    return engine


# ── Document template schemas ──────────────────────────────────────────────────

def document_schemas():
    """
    Template schemas for AI-generated PDFs linked to the synthetic structured records.
    Each document references fields from the structured tables via foreign_keys.
    """
    return {
        "EnrollmentLetter": {
            "__template__": True,
            "__description__": "Member health plan enrollment confirmation letter",
            "__name__": "EnrollmentLetter",
            "__depends_on__": ["patient"],
            "__foreign_keys__": {
                "patient_name": ["patient", "patient_name"],
                "date_of_birth": ["patient", "date_of_birth"],
            },
            "__template_source__": os.path.join(TEMPLATE_DIR, "enrollment_letter.html"),
            "__input_file_type__": "html",
            "__output_file_type__": "pdf",
            "patient_name":    {"type": "string", "description": "Member full name"},
            "date_of_birth":   {"type": "date",   "description": "Member date of birth"},
            "member_id":       {"type": "string", "description": "Unique member ID, format MBR-XXXXXX"},
            "plan_name":       {"type": "string", "description": "Health plan name (e.g. HealthFirst Gold PPO)"},
            "effective_date":  {"type": "date",   "description": "Plan effective date"},
            "primary_provider":{"type": "string", "description": "Assigned primary care provider name"},
            "enrollment_date": {"type": "date",   "description": "Date of enrollment letter"},
        },

        "ClinicalNote": {
            "__template__": True,
            "__description__": "Clinical visit note linked to a diagnosis record",
            "__name__": "ClinicalNote",
            "__depends_on__": ["diagnosis", "patient", "provider"],
            "__foreign_keys__": {
                "patient_name":     ["patient",   "patient_name"],
                "date_of_birth":    ["patient",   "date_of_birth"],
                "provider_name":    ["provider",  "provider_name"],
                "specialty":        ["provider",  "specialty"],
                "diagnosis_code":   ["diagnosis", "diagnosis_code"],
                "visit_date":       ["diagnosis", "visit_date"],
            },
            "__template_source__": os.path.join(TEMPLATE_DIR, "clinical_note.html"),
            "__input_file_type__": "html",
            "__output_file_type__": "pdf",
            "patient_name":          {"type": "string", "description": "Patient full name"},
            "date_of_birth":         {"type": "date",   "description": "Patient date of birth"},
            "visit_date":            {"type": "date",   "description": "Date of clinical visit"},
            "provider_name":         {"type": "string", "description": "Treating provider name"},
            "specialty":             {"type": "string", "description": "Provider specialty"},
            "diagnosis_code":        {"type": "string", "description": "ICD-10 diagnosis code"},
            "diagnosis_description": {"type": "string", "description": "Human-readable diagnosis description"},
            "chief_complaint":       {"type": "string", "description": "Patient's chief complaint in 1-2 sentences"},
            "clinical_notes":        {"type": "string", "description": "Clinical observations and findings, 3-4 sentences"},
            "treatment_plan":        {"type": "string", "description": "Treatment plan prescribed, 2-3 sentences"},
            "follow_up_instructions":{"type": "string", "description": "Follow-up instructions for the patient"},
        },

        "ExplanationOfBenefits": {
            "__template__": True,
            "__description__": "Explanation of Benefits document linked to a claim",
            "__name__": "ExplanationOfBenefits",
            "__depends_on__": ["claim", "patient", "provider", "diagnosis"],
            "__foreign_keys__": {
                "patient_name":   ["patient",   "patient_name"],
                "provider_name":  ["provider",  "provider_name"],
                "procedure_code": ["claim",     "procedure_code"],
                "visit_date":     ["diagnosis", "visit_date"],
            },
            "__template_source__": os.path.join(TEMPLATE_DIR, "eob.html"),
            "__input_file_type__": "html",
            "__output_file_type__": "pdf",
            "patient_name":        {"type": "string", "description": "Member full name"},
            "provider_name":       {"type": "string", "description": "Provider name"},
            "claim_id":            {"type": "string", "description": "Claim reference number, format CLM-XXXXXXXX"},
            "procedure_code":      {"type": "string", "description": "CPT procedure code"},
            "procedure_description":{"type": "string","description": "Plain-language description of procedure"},
            "visit_date":          {"type": "date",   "description": "Date of service"},
            "processed_date":      {"type": "date",   "description": "Date claim was processed"},
            "billed_amount":       {"type": "float",  "description": "Total amount billed by provider, $100-$3000"},
            "allowed_amount":      {"type": "float",  "description": "Plan allowed amount (less than billed)"},
            "discount_amount":     {"type": "float",  "description": "Plan network discount"},
            "plan_paid":           {"type": "float",  "description": "Amount paid by plan"},
            "member_responsibility":{"type": "float", "description": "Amount member owes (copay/deductible)"},
        },

        "PaymentConfirmation": {
            "__template__": True,
            "__description__": "Payment confirmation letter linked to a payment record",
            "__name__": "PaymentConfirmation",
            "__depends_on__": ["payment", "claim", "patient", "provider", "diagnosis"],
            "__foreign_keys__": {
                "patient_name":   ["patient",   "patient_name"],
                "provider_name":  ["provider",  "provider_name"],
                "payment_amount": ["payment",   "payment_amount"],
                "payment_date":   ["payment",   "payment_date"],
                "visit_date":     ["diagnosis", "visit_date"],
            },
            "__template_source__": os.path.join(TEMPLATE_DIR, "payment_letter.html"),
            "__input_file_type__": "html",
            "__output_file_type__": "pdf",
            "patient_name":   {"type": "string", "description": "Member full name"},
            "payment_id":     {"type": "string", "description": "Payment reference number, format PAY-XXXXXXXX"},
            "claim_id":       {"type": "string", "description": "Related claim number, format CLM-XXXXXXXX"},
            "provider_name":  {"type": "string", "description": "Provider who received payment"},
            "visit_date":     {"type": "date",   "description": "Original date of service"},
            "payment_amount": {"type": "float",  "description": "Payment amount in USD"},
            "payment_date":   {"type": "date",   "description": "Date payment was issued"},
            "payment_status": {"type": "string", "description": "Payment status: Paid, Pending, or NoPay"},
        },
    }


# ── Main demo ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Bio-IT World 2026 — Syda Live Demo")
    print("  AI-Generated Synthetic Healthcare Data")
    print("=" * 60)

    # Step 1: Database setup
    conn_str = build_connection_string()
    print(f"\nStep 1: Setting up healthcare database...")
    print(f"  Connecting to: {conn_str.split('@')[-1]}")  # hide credentials
    engine = setup_database()
    print(f"  ✓ Database ready")

    # Step 2: Infer structured schemas from database
    print("\nStep 2: Inferring schemas from database...")
    loader  = DatabaseSchemaLoader(engine)
    schemas = loader.load_schemas()
    print(f"  ✓ Inferred {len(schemas)} tables: {', '.join(schemas.keys())}")

    # Step 3: Add document template schemas
    schemas.update(document_schemas())
    print(f"  ✓ Added 4 PDF document templates")

    # Step 4: Generate everything
    print("\nStep 3: Generating synthetic data + linked PDFs...")
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
            "patient":               5,
            "provider":              3,
            "diagnosis":            10,
            "claim":                10,
            "payment":              10,
            "EnrollmentLetter":      3,
            "ClinicalNote":          3,
            "ExplanationOfBenefits": 3,
            "PaymentConfirmation":   3,
        },
        prompts={
            "patient":   "Generate realistic patient records with diverse ages (25-80) and genders.",
            "provider":  "Generate realistic healthcare providers with specialties like Cardiology, Internal Medicine, Family Practice.",
            "diagnosis": "Generate realistic diagnoses using common ICD-10 codes (e.g. I10, E11.9, J45.50, M54.5). Visit dates within last 2 years.",
            "claim":     "Generate realistic healthcare claims with CPT procedure codes (e.g. 99213, 99214, 93000). Amounts $100-$2500.",
            "payment":   "Generate payment records. Status mostly Paid, some Pending. Amounts slightly less than claim amounts.",
        },
        output_dir=OUTPUT_DIR,
    )

    # Step 5: Write structured data back to database
    structured = {k: v for k, v in results.items()
                  if k in ("patient", "provider", "diagnosis", "claim", "payment")}

    print("\nStep 4: Writing structured data back to database...")
    loader.write_to_database(structured)

    # Summary
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print("\nStructured data (CSV + database):")
    for table in ("patient", "provider", "diagnosis", "claim", "payment"):
        if table in results:
            print(f"  ✓ {table}: {len(results[table])} rows")

    print("\nLinked PDF documents:")
    for doc in ("EnrollmentLetter", "ClinicalNote", "ExplanationOfBenefits", "PaymentConfirmation"):
        if doc in results:
            print(f"  ✓ {doc}: {len(results[doc])} PDFs")

    print(f"\nAll output saved to: {OUTPUT_DIR}/")
    print(f"Database updated:    {build_connection_string().split('@')[-1]}\n")


if __name__ == "__main__":
    main()

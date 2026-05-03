"""
db_schema_inference.py
======================
Database schema inference extension for the Syda framework.

This module adds schema inference capability to Syda's SyntheticDataGenerator,
allowing users to automatically extract schemas from existing relational databases
instead of defining them manually.

Supported databases: SQLite, MySQL, PostgreSQL (via SQLAlchemy)

Usage:
    from syda import SyntheticDataGenerator, ModelConfig
    from dotenv import load_dotenv
    from db_schema_inference import patch_generator

    load_dotenv()

    generator = SyntheticDataGenerator(
        model_config=ModelConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001"
        )
    )

    # Patch the generator with the new method
    patch_generator(generator)

    # Step 1: Infer schema from database
    schema_files = generator.create_schemas_from_database(
        connection_string_or_engine="sqlite:///healthcare_demo.db",
        output_dir="generated_schemas",
        table_names=None,  # None = all tables
        format="yaml"
    )

    # Step 2: Generate synthetic data
    results = generator.generate_for_schemas(
        schemas=schema_files,
        sample_sizes={"patient": 10, "claim": 20},
        prompts={"patient": "Generate realistic patient records."},
        output_dir="synthetic_output"
    )
"""

import os
import json
import types
from typing import Dict, List, Optional, Union


def create_schemas_from_database(
    self,
    connection_string_or_engine: Union[str, object],
    output_dir: str,
    table_names: Optional[List[str]] = None,
    format: str = "yaml"
) -> Dict[str, str]:
    """
    Infer schemas from an existing relational database and save them as
    YAML or JSON files compatible with generate_for_schemas().

    Connects to a database, extracts table structure (columns, types,
    primary keys, foreign keys, nullability), and writes one schema file
    per table to the specified output directory.

    Args:
        connection_string_or_engine: SQLAlchemy connection URL string or
            an existing SQLAlchemy engine object.
            Examples:
                "sqlite:///mydb.db"
                "postgresql+psycopg2://user:pass@localhost/mydb"
                "mysql+pymysql://user:pass@localhost/mydb"
        output_dir (str): Directory to save schema files. Created
            automatically if it does not exist.
        table_names (list, optional): Specific tables to infer. If None,
            all tables in the database are processed.
        format (str): Output format — "yaml" (default) or "json".

    Returns:
        dict: Table name mapped to the absolute path of its schema file.
              Pass this directly to generate_for_schemas(schemas=...).

    Raises:
        ImportError: If SQLAlchemy is not installed.
        ValueError: If an unsupported format is specified.
        sqlalchemy.exc.OperationalError: If the database connection fails.
    """
    try:
        from sqlalchemy import create_engine, inspect as sa_inspect
    except ImportError:
        raise ImportError(
            "SQLAlchemy is required. Install it with: pip install sqlalchemy"
        )

    if format not in ("yaml", "json"):
        raise ValueError(f"Unsupported format '{format}'. Use 'yaml' or 'json'.")

    # Accept either a connection string or an existing engine

    if isinstance(connection_string_or_engine, str):
        engine = create_engine(connection_string_or_engine)
    else:
        engine = connection_string_or_engine

    inspector = sa_inspect(engine)

    all_tables = inspector.get_table_names()
    tables_to_process = table_names if table_names else all_tables

    os.makedirs(output_dir, exist_ok=True)

    schema_files = {}

    for table_name in tables_to_process:
        if table_name not in all_tables:
            print(f"  Warning: Table '{table_name}' not found. Skipping.")
            continue

        columns = inspector.get_columns(table_name)
        pk_cols = set(
            inspector.get_pk_constraint(table_name).get("constrained_columns", [])
        )

        # Build FK lookup: column name → referenced table and column

        fk_map = {}
        for fk in inspector.get_foreign_keys(table_name):
            referred_cols = fk["referred_columns"]
            for i, col in enumerate(fk["constrained_columns"]):
                fk_map[col] = {
                    "referred_table":  fk["referred_table"],
                    "referred_column": referred_cols[i] if i < len(referred_cols) else referred_cols[0]
                }

        # Build Syda-compatible schema dict for this table

        schema = {}
        for col in columns:
            col_name = col["name"]
            is_pk = col_name in pk_cols

            col_def = {"type": self._infer_syda_type(col["type"])}

            if is_pk:
                col_def["primary_key"] = True
                col_def["not_null"] = True
            elif not col.get("nullable", True):
                col_def["not_null"] = True

            if col_name in fk_map:
                col_def["type"] = "foreign_key"
                col_def["references"] = {
                    "table": fk_map[col_name]["referred_table"],
                    "field": fk_map[col_name]["referred_column"]
                }

            schema[col_name] = col_def

        # Save schema file

        ext = "yaml" if format == "yaml" else "json"
        file_path = os.path.abspath(os.path.join(output_dir, f"{table_name}.{ext}"))

        if format == "yaml":
            self._schema_to_yaml(schema, file_path)
        else:
            with open(file_path, "w") as f:
                json.dump(schema, f, indent=2)

        schema_files[table_name] = file_path
        print(f"  [OK] {table_name} -> {file_path}")

    return schema_files


def _infer_syda_type(self, sql_type) -> str:
    """
    Map a SQLAlchemy column type to a Syda-compatible type string.

    Uses substring matching to handle vendor-specific type variants
    (e.g. BIGINT, TINYINT, NVARCHAR). Unknown types default to 'string'.

    Args:
        sql_type: SQLAlchemy column type object.

    Returns:
        str: One of 'integer', 'string', 'date', 'float', 'boolean'.
    """
    t = str(sql_type).lower()
    if "int" in t:
        return "integer"
    elif "char" in t or "text" in t or "varchar" in t or "clob" in t:
        return "string"
    elif "date" in t or "time" in t:
        return "date"
    elif "decimal" in t or "numeric" in t or "float" in t or "real" in t or "double" in t:
        return "float"
    elif "bool" in t:
        return "boolean"
    else:
        return "string"


def _schema_to_yaml(self, schema: dict, file_path: str) -> None:
    """
    Write a schema dictionary to a YAML file.

    Uses PyYAML if available (it is a Syda dependency), otherwise falls
    back to a simple hand-written serializer.

    Args:
        schema (dict): Column-keyed schema dictionary.
        file_path (str): Absolute path for the output YAML file.
    """
    try:
        import yaml
        with open(file_path, "w") as f:
            yaml.dump(schema, f, default_flow_style=False, allow_unicode=True)
    except ImportError:

        # Fallback serializer — this should rarely be needed

        lines = []
        for col_name, col_def in schema.items():
            lines.append(f"{col_name}:")
            for key, value in col_def.items():
                if isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        lines.append(f"    {k}: {v}")
                elif isinstance(value, bool):
                    lines.append(f"  {key}: {'true' if value else 'false'}")
                else:
                    lines.append(f"  {key}: {value}")
        with open(file_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def patch_generator(generator) -> None:
    """
    Attach the schema inference methods to an existing SyntheticDataGenerator
    instance.

    Call this once after creating the generator to enable
    create_schemas_from_database().

    Args:
        generator: An initialized SyntheticDataGenerator instance.
    """
    generator.create_schemas_from_database = types.MethodType(
        create_schemas_from_database, generator
    )
    generator._infer_syda_type = types.MethodType(_infer_syda_type, generator)
    generator._schema_to_yaml = types.MethodType(_schema_to_yaml, generator)


def run_demo():
    """
    Runs the full end-to-end demo using the healthcare domain schema.
    Demonstrates the complete workflow from database connection to
    synthetic data generation.
    """
    from syda import SyntheticDataGenerator, ModelConfig
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text

    load_dotenv()

    # Set up the healthcare demo database

    print("Setting up healthcare demo database...")
    engine = create_engine("sqlite:///healthcare_demo.db")
    with engine.connect() as conn:
        conn.execute(text("""CREATE TABLE IF NOT EXISTS patient (
            patient_id INTEGER PRIMARY KEY, patient_name TEXT NOT NULL,
            age INTEGER, gender TEXT, date_of_birth DATE)"""))
        conn.execute(text("""CREATE TABLE IF NOT EXISTS provider (
            provider_id INTEGER PRIMARY KEY, provider_name TEXT NOT NULL,
            specialty TEXT, license_number TEXT, facility_id TEXT)"""))
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
    print("  Demo database ready.\n")

    # Initialize generator and patch it with inference methods

    generator = SyntheticDataGenerator(
        model_config=ModelConfig(
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001"
        )
    )
    patch_generator(generator)

    # Step 1: Infer schema from database

    print("Inferring schema from database...")
    schema_files = generator.create_schemas_from_database(
        connection_string_or_engine="sqlite:///healthcare_demo.db",
        output_dir="generated_schemas",
        table_names=None,
        format="yaml"
    )
    print(f"  Schema files: {list(schema_files.keys())}\n")

    # Step 2: Generate synthetic data

    print("Generating synthetic data...")
    results = generator.generate_for_schemas(
        schemas=schema_files,
        sample_sizes={
            "patient":       10,
            "provider":       5,
            "diagnosis":     20,
            "claim":         20,
            "adjudication":  20,
            "payment":       20,
        },
        prompts={
            "patient":      "Generate realistic patient records with diverse ages (18-85) and genders.",
            "provider":     "Generate realistic healthcare providers with diverse specialties.",
            "diagnosis":    "Generate realistic diagnoses using ICD-10 codes (e.g. I10, E11.9).",
            "claim":        "Generate realistic healthcare claims using CPT codes, amounts $50-$5000.",
            "adjudication": "Generate adjudication records: ~60% Approved, ~25% Partial, ~15% Denied.",
            "payment":      "Generate payment records, mostly Paid status.",
        },
        output_dir="synthetic_output"
    )

    print("\nDone.")
    for table, df in results.items():
        print(f"  {table}: {len(df)} rows -> synthetic_output/{table}.csv")


if __name__ == "__main__":
    run_demo()
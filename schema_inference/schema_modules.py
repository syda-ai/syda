# Supported databases: SQLite, MySQL, PostgreSQL (all used via an SQLAlchemy URL)

# Modules:
#    - Type Mapping Engine
#    - Database Connector
#    - Metadata Extractor
#    - Schema Builder
#    - Compatibility Validator
#    - Versioning Module

# Dependencies 
#
# Required (install before running):
#   pip install sqlalchemy            — core dependency, needed for all three databases
#
# Optional (install based on database used that are used):
#   pip install sqlite3               — SQLite support (since it isbuilt-in with Python, no separate install needed)
#   pip install psycopg2-binary       — PostgreSQL support
#   pip install pymysql               — MySQL support
#
# For running tests:
#   pip install pytest
#
# Usage:
#   python schema_modules.py --db-url "sqlite:///mydb.db" --output "schema.json"
#   python schema_modules.py --db-url "sqlite:///mydb.db" --create-demo-sqlite
#   python schema_modules.py --db-url "postgresql://user:pass@localhost/dbname" --output "schema.json"
#   python schema_modules.py --db-url "mysql+pymysql://user:pass@localhost/dbname" --output "schema.json"
#   python schema_modules.py --db-url "sqlite:///mydb.db" --output "schema.json" --skip-validation
#   python schema_modules.py --db-url "sqlite:///mydb.db" --output "schema.json" > extraction.log 2>&1
#   python schema_modules.py --db-url "sqlite:///mydb.db" --output "schema.json" && cat schema_validation_report.json
#   python schema_modules.py --db-url "sqlite:///mydb.db" --output "schema.json" && cat schema_syda_dict.json
#   python schema_modules.py --db-url "sqlite:///mydb.db" --output "schema.json" && cat schema.json
#   python schema_modules.py --db-url "sqlite:///mydb.db" --output "schema.json" && cat schema_syda_dict.json && cat schema_validation_report.json 


import argparse
import json
import os
import sys
import datetime
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError

# Type Mapping Engine

def map_sql_type(sql_type):
    
    t = str(sql_type).lower()
    if "int" in t:
        return "integer"
    elif "char" in t or "text" in t or "varchar" in t:
        return "string"
    elif "date" in t or "time" in t:
        return "date"
    elif "decimal" in t or "numeric" in t:
        return "decimal"
    elif "float" in t or "real" in t or "double" in t:
        return "float"
    elif "bool" in t:
        return "boolean"
    else:
        return "string"

# Demo Database Builder

def create_demo_sqlite(engine):
    
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS patient (
            patient_id INTEGER PRIMARY KEY,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            date_of_birth DATE
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS provider (
            provider_id INTEGER PRIMARY KEY,
            provider_name TEXT NOT NULL,
            specialty TEXT,
            license_number TEXT,
            facility_id TEXT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS diagnosis (
            diagnosis_id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            provider_id INTEGER NOT NULL,
            diagnosis_code TEXT,
            visit_date DATE,
            FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
            FOREIGN KEY (provider_id) REFERENCES provider(provider_id)
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS claim (
            claim_id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            provider_id INTEGER NOT NULL,
            diagnosis_id INTEGER NOT NULL,
            procedure_code TEXT,
            claim_amount DECIMAL(10,2),
            submission_date DATE,
            FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
            FOREIGN KEY (provider_id) REFERENCES provider(provider_id),
            FOREIGN KEY (diagnosis_id) REFERENCES diagnosis(diagnosis_id)
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS adjudication (
            adjudication_id INTEGER PRIMARY KEY,
            claim_id INTEGER NOT NULL,
            decision TEXT CHECK(decision IN ('Approved', 'Denied', 'Partial')),
            denial_reason TEXT,
            approved_amount DECIMAL(10,2),
            FOREIGN KEY (claim_id) REFERENCES claim(claim_id)
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS payment (
            payment_id INTEGER PRIMARY KEY,
            claim_id INTEGER NOT NULL,
            payment_date DATE,
            payment_amount DECIMAL(10,2),
            status TEXT CHECK(status IN ('Paid', 'Pending', 'NoPay')),
            FOREIGN KEY (claim_id) REFERENCES claim(claim_id)
        );
        """))
        conn.commit()

# Metadata Extractor

def extract_schema(engine):
    
    inspector = inspect(engine)
    schema = {}

    for table in inspector.get_table_names():
        schema[table] = _extract_table_schema(inspector, table, is_view=False)

    for view in inspector.get_view_names():
        schema[view] = _extract_table_schema(inspector, view, is_view=True)

    return schema


def _extract_table_schema(inspector, table_name, is_view=False):
    
    columns = inspector.get_columns(table_name)
    pk_cols = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))
    fks = inspector.get_foreign_keys(table_name)

    fk_map = {}
    for fk in fks:
        referred_cols = fk["referred_columns"]
        for i, col in enumerate(fk["constrained_columns"]):
            fk_map[col] = {
                "references_table": fk["referred_table"],
                "references_column": referred_cols[i] if i < len(referred_cols) else referred_cols[0]
            }

    table_schema = {}
    if is_view:
        table_schema["_meta"] = {"is_view": True}

    for col in columns:
        col_name = col["name"]
        is_pk = col_name in pk_cols

        col_def = {
            "type": map_sql_type(col["type"]),
            "nullable": False if is_pk else bool(col.get("nullable", True))
        }

        if is_pk:
            col_def["primary_key"] = True

        if col_name in fk_map:
            col_def["foreign_key"] = fk_map[col_name]

        table_schema[col_name] = col_def

    return table_schema

# Schema Builder

def to_syda_dict(schema):
    
    syda_schema = {}
    for table, columns in schema.items():
        syda_schema[table] = {
            col: attrs
            for col, attrs in columns.items()
            if not col.startswith("_")
        }
    return syda_schema

# Compatibility Validator

def validate_syda_compatibility(syda_schema):
    
    results = []

    try:
        from syda import generate_for_schemas  # noqa

        for table, columns in syda_schema.items():
            try:
                generate_for_schemas({table: columns}, n=1)
                results.append({"table": table, "status": "PASS", "detail": "generate_for_schemas() succeeded"})
            except Exception as e:
                results.append({"table": table, "status": "FAIL", "detail": str(e)})

        return all(r["status"] == "PASS" for r in results), results

    except ImportError:
        VALID_TYPES = {"integer", "string", "date", "decimal", "float", "boolean"}
        for table, columns in syda_schema.items():
            issues = []
            has_pk = False
            for col, attrs in columns.items():
                if attrs.get("type") not in VALID_TYPES:
                    issues.append(f"'{col}': unknown type '{attrs.get('type')}'")
                if attrs.get("primary_key"):
                    has_pk = True
            if not has_pk:
                issues.append("no primary key defined")
            if issues:
                results.append({"table": table, "status": "WARN", "detail": "; ".join(issues)})
            else:
                results.append({"table": table, "status": "PASS", "detail": "structural check passed (Syda not installed)"})

        passed = all(r["status"] in ("PASS", "WARN") for r in results)
        return passed, results

# Versioning Module

def save_versioned_output(schema, syda_schema, output_path, validation_results):
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    versioned = {
        "_version": {
            "generated_at": timestamp,
            "format": "schema_extractor"
        },
        "schema": schema
    }
    with open(output_path, "w") as f:
        json.dump(versioned, f, indent=4)

    base, ext = os.path.splitext(output_path)
    syda_path = f"{base}_syda_dict{ext}"
    syda_versioned = {
        "_version": {
            "generated_at": timestamp,
            "format": "syda_dict"
        },
        "schema": syda_schema
    }
    with open(syda_path, "w") as f:
        json.dump(syda_versioned, f, indent=4)

    report_path = f"{base}_validation_report.json"
    report = {
        "_version": {"generated_at": timestamp},
        "results": validation_results,
        "summary": {
            "total": len(validation_results),
            "passed": sum(1 for r in validation_results if r["status"] == "PASS"),
            "warned": sum(1 for r in validation_results if r["status"] == "WARN"),
            "failed": sum(1 for r in validation_results if r["status"] == "FAIL"),
        }
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return syda_path, report_path

# Database Connector Helpers

def verify_connection(engine):
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as e:
        print(f"ERROR: Could not connect to database.\n  {e}", file=sys.stderr)
        sys.exit(1)


def validate_output_path(output_path):
    
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(output_dir):
        print(f"ERROR: Output directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

# Main CLI Entry Point

def main():
    
    parser = argparse.ArgumentParser(description="Syda Extension")
    parser.add_argument("--db-url", required=True, help="SQLAlchemy database URL (e.g. sqlite:///mydb.db)")
    parser.add_argument("--create-demo-sqlite", action="store_true", help="Populate the full demo schema")
    parser.add_argument("--output", default="inferred_schema.json", help="Output JSON file path (default: inferred_schema.json)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip Syda compatibility validation step when it is not installed")
    args = parser.parse_args()

    validate_output_path(args.output)

    engine = create_engine(args.db_url)

    print("Testing database connection")
    verify_connection(engine)

    if args.create_demo_sqlite:
        print("Creating full demo SQLite database schema...")
        create_demo_sqlite(engine)

    print("Extracting schema...")
    schema = extract_schema(engine)

    print("Converting to Syda dictionary format...")
    syda_schema = to_syda_dict(schema)

    validation_results = []
    if not args.skip_validation:
        print("Running Syda compatibility validation...")
        passed, validation_results = validate_syda_compatibility(syda_schema)
        for r in validation_results:
            icon = "v" if r["status"] == "PASS" else ("!" if r["status"] == "WARN" else "x")
            print(f"  [{icon}] {r['table']}: {r['detail']}")
        if not passed:
            print("WARNING: One or more tables failed validation.")

    print("Saving versioned output files...")
    syda_path, report_path = save_versioned_output(schema, syda_schema, args.output, validation_results)

    table_count = sum(1 for v in schema.values() if not (isinstance(v.get("_meta"), dict) and v["_meta"].get("is_view")))
    view_count = len(schema) - table_count

    print(f"\nDone.")
    print(f"  Full schema    -> {args.output}")
    print(f"  Syda dict      -> {syda_path}")
    print(f"  Validation     -> {report_path}")
    print(f"  Entities: {table_count} tables, {view_count} views")


if __name__ == "__main__":
    main()
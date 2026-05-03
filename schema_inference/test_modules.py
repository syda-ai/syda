# Test coverage:
#    - Type Mapping Engine
#    - Database Connector
#    - Metadata Extractor
#    - Constraint Extractor
#    - Schema Builder
#    - Compatibility Validator
#    - Versioning Module
#    - Edge Cases
#
# Run with:
#    python -m pytest test_modules.py -v
#    -- or --
#    python test_modules.py

import json
import os
import tempfile
import unittest

from sqlalchemy import create_engine, text

from schema_modules import (
    map_sql_type,
    extract_schema,
    to_syda_dict,
    validate_syda_compatibility,
    save_versioned_output,
    verify_connection,
    create_demo_sqlite,
    _extract_table_schema,
)

# Helpers

def make_engine(sql_statements=None):
    """
    Creates an in-memory SQLite engine and optionally seeds it with DDL.

    Args:
        sql_statements (list of str, optional): DDL statements to execute
            after the engine is created.

    Returns:
        sqlalchemy.engine.Engine: A connected in-memory SQLite engine.
    """
    engine = create_engine("sqlite:///:memory:")
    if sql_statements:
        with engine.connect() as conn:
            for stmt in sql_statements:
                conn.execute(text(stmt))
            conn.commit()
    return engine

# 1. Type Mapping Engine

class TestMapSqlType(unittest.TestCase):
    """Tests for map_sql_type() — the SQL-to-Syda type mapping engine."""

    def test_integer_variants(self):
        """INT, INTEGER, BIGINT, SMALLINT, TINYINT all map to 'integer'."""
        for t in ["INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"]:
            with self.subTest(t=t):
                self.assertEqual(map_sql_type(t), "integer")

    def test_string_variants(self):
        """TEXT, VARCHAR, CHAR, NVARCHAR all map to 'string'."""
        for t in ["TEXT", "VARCHAR(255)", "CHAR(10)", "NVARCHAR(100)"]:
            with self.subTest(t=t):
                self.assertEqual(map_sql_type(t), "string")

    def test_date_variants(self):
        """DATE, DATETIME, TIMESTAMP, TIME all map to 'date'."""
        for t in ["DATE", "DATETIME", "TIMESTAMP", "TIME"]:
            with self.subTest(t=t):
                self.assertEqual(map_sql_type(t), "date")

    def test_decimal_variants(self):
        """DECIMAL and NUMERIC map to 'decimal'."""
        for t in ["DECIMAL(10,2)", "NUMERIC(8,4)"]:
            with self.subTest(t=t):
                self.assertEqual(map_sql_type(t), "decimal")

    def test_float_variants(self):
        """FLOAT, REAL, DOUBLE map to 'float'."""
        for t in ["FLOAT", "REAL", "DOUBLE"]:
            with self.subTest(t=t):
                self.assertEqual(map_sql_type(t), "float")

    def test_boolean(self):
        """BOOLEAN maps to 'boolean'."""
        self.assertEqual(map_sql_type("BOOLEAN"), "boolean")

    def test_unknown_type_defaults_to_string(self):
        """Unsupported types (e.g. GEOMETRY, BLOB, JSON) default to 'string'."""
        for t in ["GEOMETRY", "BLOB", "JSON", "UUID", "XML"]:
            with self.subTest(t=t):
                self.assertEqual(map_sql_type(t), "string")

    def test_case_insensitive(self):
        """Type matching is case-insensitive."""
        self.assertEqual(map_sql_type("integer"), "integer")
        self.assertEqual(map_sql_type("Integer"), "integer")
        self.assertEqual(map_sql_type("VARCHAR"), "string")

    def test_returns_integer_not_int(self):
        """Return value must be 'integer' (Syda type), not 'int'."""
        result = map_sql_type("INTEGER")
        self.assertEqual(result, "integer")
        self.assertNotEqual(result, "int")

# 2. Database Connector

class TestDatabaseConnector(unittest.TestCase):
    """Tests for database connectivity and verify_connection()."""

    def test_valid_connection_does_not_raise(self):
        """verify_connection() should succeed silently on a valid engine."""
        engine = make_engine()
        try:
            verify_connection(engine)
        except SystemExit:
            self.fail("verify_connection() raised SystemExit on a valid connection.")

    def test_invalid_connection_exits(self):
        """verify_connection() should call sys.exit(1) on an unreachable database."""
        bad_engine = create_engine("sqlite:////nonexistent/path/db.sqlite")
        with self.assertRaises(SystemExit) as ctx:
            verify_connection(bad_engine)
        self.assertEqual(ctx.exception.code, 1)

# 3. Metadata Extractor

class TestMetadataExtractor(unittest.TestCase):
    """Tests for extract_schema() and _extract_table_schema()."""

    def setUp(self):
        self.engine = make_engine([
            """CREATE TABLE patient (
                patient_id INTEGER PRIMARY KEY,
                patient_name TEXT NOT NULL,
                age INTEGER,
                date_of_birth DATE
            )"""
        ])

    def test_table_appears_in_schema(self):
        """The extracted schema should contain the 'patient' table."""
        schema = extract_schema(self.engine)
        self.assertIn("patient", schema)

    def test_all_columns_extracted(self):
        """All four columns should be present in the patient schema."""
        schema = extract_schema(self.engine)
        cols = schema["patient"]
        for col in ["patient_id", "patient_name", "age", "date_of_birth"]:
            with self.subTest(col=col):
                self.assertIn(col, cols)

    def test_column_types_mapped(self):
        """Column types should be Syda-compatible strings, not raw SQL types."""
        schema = extract_schema(self.engine)
        cols = schema["patient"]
        self.assertEqual(cols["patient_id"]["type"], "integer")
        self.assertEqual(cols["patient_name"]["type"], "string")
        self.assertEqual(cols["date_of_birth"]["type"], "date")

    def test_pk_column_not_nullable(self):
        """Primary key columns must be marked nullable=False."""
        schema = extract_schema(self.engine)
        self.assertFalse(schema["patient"]["patient_id"]["nullable"])

    def test_nullable_column(self):
        """Non-PK, non-NOT NULL columns should be nullable=True."""
        schema = extract_schema(self.engine)
        self.assertTrue(schema["patient"]["age"]["nullable"])

    def test_empty_table_extracted(self):
        """An empty table (no rows) should still produce a valid schema."""
        engine = make_engine([
            "CREATE TABLE empty_table (id INTEGER PRIMARY KEY, val TEXT)"
        ])
        schema = extract_schema(engine)
        self.assertIn("empty_table", schema)
        self.assertIn("id", schema["empty_table"])

    def test_multiple_tables_extracted(self):
        """All tables in the database should appear in the schema."""
        engine = make_engine([
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY)",
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY)",
            "CREATE TABLE t3 (id INTEGER PRIMARY KEY)",
        ])
        schema = extract_schema(engine)
        for t in ["t1", "t2", "t3"]:
            self.assertIn(t, schema)

# 4. Constraint Extractor — Primary Keys

class TestPrimaryKeyExtraction(unittest.TestCase):
    """Tests for primary key detection via _extract_table_schema()."""

    def test_single_column_pk(self):
        """A single-column PK should be marked with primary_key=True."""
        engine = make_engine([
            "CREATE TABLE t (pk_col INTEGER PRIMARY KEY, data TEXT)"
        ])
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(engine)
        schema = _extract_table_schema(inspector, "t")
        self.assertTrue(schema["pk_col"].get("primary_key"))
        self.assertFalse(schema["data"].get("primary_key", False))

    def test_composite_pk_both_columns_marked(self):
        """Both columns of a composite PK should be marked primary_key=True."""
        engine = make_engine([
            """CREATE TABLE composite (
                col_a INTEGER NOT NULL,
                col_b INTEGER NOT NULL,
                val TEXT,
                PRIMARY KEY (col_a, col_b)
            )"""
        ])
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(engine)
        schema = _extract_table_schema(inspector, "composite")
        self.assertTrue(schema["col_a"].get("primary_key"),
                        "col_a should be marked as primary_key in composite PK")
        self.assertTrue(schema["col_b"].get("primary_key"),
                        "col_b should be marked as primary_key in composite PK")
        self.assertFalse(schema["val"].get("primary_key", False),
                         "val should NOT be marked as primary_key")

    def test_no_pk_column_has_no_primary_key_key(self):
        """Columns that are not PKs should not have the 'primary_key' key at all."""
        engine = make_engine([
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"
        ])
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(engine)
        schema = _extract_table_schema(inspector, "t")
        self.assertNotIn("primary_key", schema["name"])

# 5. Constraint Extractor — Foreign Keys

class TestForeignKeyExtraction(unittest.TestCase):
    """Tests for foreign key detection via _extract_table_schema()."""

    def setUp(self):
        self.engine = make_engine([
            "CREATE TABLE parent (parent_id INTEGER PRIMARY KEY, name TEXT)",
            """CREATE TABLE child (
                child_id INTEGER PRIMARY KEY,
                parent_id INTEGER NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES parent(parent_id)
            )"""
        ])

    def test_fk_column_has_foreign_key_entry(self):
        """The FK column should include a 'foreign_key' dict."""
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(self.engine)
        schema = _extract_table_schema(inspector, "child")
        self.assertIn("foreign_key", schema["parent_id"])

    def test_fk_references_correct_table(self):
        """The FK's references_table should be 'parent'."""
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(self.engine)
        schema = _extract_table_schema(inspector, "child")
        fk = schema["parent_id"]["foreign_key"]
        self.assertEqual(fk["references_table"], "parent")

    def test_fk_references_correct_column(self):
        """The FK's references_column should be 'parent_id'."""
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(self.engine)
        schema = _extract_table_schema(inspector, "child")
        fk = schema["parent_id"]["foreign_key"]
        self.assertEqual(fk["references_column"], "parent_id")

    def test_non_fk_column_has_no_foreign_key(self):
        """Non-FK columns should not have a 'foreign_key' key."""
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(self.engine)
        schema = _extract_table_schema(inspector, "child")
        self.assertNotIn("foreign_key", schema["child_id"])

    def test_parent_table_has_no_fk(self):
        """The parent table itself should have no foreign key entries."""
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(self.engine)
        schema = _extract_table_schema(inspector, "parent")
        for col_def in schema.values():
            self.assertNotIn("foreign_key", col_def)

# 6. Schema Builder

class TestSchemaBuilder(unittest.TestCase):
    """Tests for to_syda_dict() — the schema format converter."""

    def test_meta_key_stripped(self):
        """The '_meta' key should be removed from view schemas."""
        raw = {
            "my_view": {
                "_meta": {"is_view": True},
                "col1": {"type": "string", "nullable": True}
            }
        }
        result = to_syda_dict(raw)
        self.assertNotIn("_meta", result["my_view"])

    def test_columns_preserved(self):
        """Real column entries should be preserved unchanged."""
        raw = {
            "patient": {
                "patient_id": {"type": "integer", "nullable": False, "primary_key": True},
                "name": {"type": "string", "nullable": False}
            }
        }
        result = to_syda_dict(raw)
        self.assertIn("patient_id", result["patient"])
        self.assertIn("name", result["patient"])

    def test_flat_dict_structure(self):
        """Output should be a flat dict keyed by table name (no 'tables' wrapper)."""
        raw = {"t1": {"id": {"type": "integer", "nullable": False, "primary_key": True}}}
        result = to_syda_dict(raw)
        self.assertIn("t1", result)
        self.assertNotIn("tables", result)

    def test_table_without_meta_unchanged(self):
        """Tables without _meta should pass through untouched."""
        raw = {
            "claim": {
                "claim_id": {"type": "integer", "nullable": False, "primary_key": True}
            }
        }
        result = to_syda_dict(raw)
        self.assertEqual(result["claim"], raw["claim"])

    def test_all_private_keys_stripped(self):
        """Any key starting with '_' should be stripped."""
        raw = {
            "t": {
                "_meta": {"is_view": True},
                "_internal": {"debug": True},
                "real_col": {"type": "string", "nullable": True}
            }
        }
        result = to_syda_dict(raw)
        for key in result["t"]:
            self.assertFalse(key.startswith("_"),
                             f"Private key '{key}' should have been stripped")

# 7. Compatibility Validator

class TestCompatibilityValidator(unittest.TestCase):
    """Tests for validate_syda_compatibility() — structural validation mode."""

    def _make_valid_schema(self):
        return {
            "patient": {
                "patient_id": {"type": "integer", "nullable": False, "primary_key": True},
                "name": {"type": "string", "nullable": False}
            }
        }

    def test_valid_schema_passes(self):
        """A well-formed schema should return passed=True and all PASS statuses."""
        passed, results = validate_syda_compatibility(self._make_valid_schema())
        self.assertTrue(passed)
        self.assertTrue(all(r["status"] in ("PASS", "WARN") for r in results))

    def test_missing_pk_produces_warn(self):
        """A table with no primary key should produce a WARN result."""
        schema = {
            "no_pk_table": {
                "col1": {"type": "string", "nullable": True}
            }
        }
        passed, results = validate_syda_compatibility(schema)
        statuses = {r["table"]: r["status"] for r in results}
        self.assertEqual(statuses["no_pk_table"], "WARN")

    def test_unknown_type_produces_warn(self):
        """A column with an unmapped type should produce a WARN result."""
        schema = {
            "t": {
                "id": {"type": "integer", "nullable": False, "primary_key": True},
                "geo": {"type": "geometry", "nullable": True}
            }
        }
        passed, results = validate_syda_compatibility(schema)
        statuses = {r["table"]: r["status"] for r in results}
        self.assertEqual(statuses["t"], "WARN")

    def test_results_list_has_entry_per_table(self):
        """There should be exactly one result entry per table."""
        schema = {
            "t1": {"id": {"type": "integer", "nullable": False, "primary_key": True}},
            "t2": {"id": {"type": "integer", "nullable": False, "primary_key": True}},
        }
        _, results = validate_syda_compatibility(schema)
        self.assertEqual(len(results), 2)

    def test_empty_schema_passes(self):
        """An empty schema (no tables) should return passed=True with no results."""
        passed, results = validate_syda_compatibility({})
        self.assertTrue(passed)
        self.assertEqual(results, [])

    def test_valid_types_accepted(self):
        """All Syda-supported types should produce PASS, not WARN."""
        for syda_type in ["integer", "string", "date", "decimal", "float", "boolean"]:
            with self.subTest(type=syda_type):
                schema = {
                    "t": {
                        "id": {"type": "integer", "nullable": False, "primary_key": True},
                        "col": {"type": syda_type, "nullable": True}
                    }
                }
                passed, results = validate_syda_compatibility(schema)
                self.assertEqual(results[0]["status"], "PASS")

# 8. Versioning Module

class TestVersioningModule(unittest.TestCase):
    """Tests for save_versioned_output() — file generation and version envelopes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.tmpdir, "schema.json")
        self.schema = {
            "patient": {
                "patient_id": {"type": "integer", "nullable": False, "primary_key": True}
            }
        }
        self.syda_schema = self.schema.copy()
        self.validation_results = [
            {"table": "patient", "status": "PASS", "detail": "structural check passed (Syda not installed)"}
        ]

    def test_three_files_created(self):
        """save_versioned_output() should create exactly three files."""
        syda_path, report_path = save_versioned_output(
            self.schema, self.syda_schema, self.output_path, self.validation_results
        )
        self.assertTrue(os.path.exists(self.output_path), "Main schema file missing")
        self.assertTrue(os.path.exists(syda_path), "Syda dict file missing")
        self.assertTrue(os.path.exists(report_path), "Validation report file missing")

    def test_main_file_has_version_envelope(self):
        """The main output file should contain a '_version' key."""
        save_versioned_output(
            self.schema, self.syda_schema, self.output_path, self.validation_results
        )
        with open(self.output_path) as f:
            data = json.load(f)
        self.assertIn("_version", data)
        self.assertIn("generated_at", data["_version"])

    def test_syda_dict_file_has_version_envelope(self):
        """The Syda dict file should also contain a '_version' key."""
        syda_path, _ = save_versioned_output(
            self.schema, self.syda_schema, self.output_path, self.validation_results
        )
        with open(syda_path) as f:
            data = json.load(f)
        self.assertIn("_version", data)

    def test_validation_report_summary_counts(self):
        """The validation report summary should correctly count PASS/WARN/FAIL."""
        results = [
            {"table": "t1", "status": "PASS",  "detail": "ok"},
            {"table": "t2", "status": "WARN",  "detail": "no pk"},
            {"table": "t3", "status": "FAIL",  "detail": "bad type"},
        ]
        _, report_path = save_versioned_output(
            self.schema, self.syda_schema, self.output_path, results
        )
        with open(report_path) as f:
            report = json.load(f)
        summary = report["summary"]
        self.assertEqual(summary["total"],  3)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["warned"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_empty_validation_results_saved(self):
        """save_versioned_output() should handle an empty validation list gracefully."""
        _, report_path = save_versioned_output(
            self.schema, self.syda_schema, self.output_path, []
        )
        with open(report_path) as f:
            report = json.load(f)
        self.assertEqual(report["summary"]["total"], 0)

    def test_schema_content_preserved(self):
        """The schema content in the output file should match the input."""
        save_versioned_output(
            self.schema, self.syda_schema, self.output_path, self.validation_results
        )
        with open(self.output_path) as f:
            data = json.load(f)
        self.assertEqual(data["schema"], self.schema)

# 9. Demo Database Integration

class TestDemoDatabaseIntegration(unittest.TestCase):
    """Integration tests using create_demo_sqlite() to build the full schema."""

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        create_demo_sqlite(self.engine)
        self.schema = extract_schema(self.engine)

    def test_all_six_entities_present(self):
        """All six healthcare entities should be present after demo setup."""
        expected = {"patient", "provider", "diagnosis", "claim", "adjudication", "payment"}
        self.assertEqual(set(self.schema.keys()), expected)

    def test_patient_primary_key(self):
        """patient_id should be the primary key of the patient table."""
        self.assertTrue(self.schema["patient"]["patient_id"].get("primary_key"))

    def test_diagnosis_foreign_keys(self):
        """diagnosis should have FKs to both patient and provider."""
        diagnosis = self.schema["diagnosis"]
        self.assertIn("foreign_key", diagnosis["patient_id"])
        self.assertIn("foreign_key", diagnosis["provider_id"])
        self.assertEqual(diagnosis["patient_id"]["foreign_key"]["references_table"], "patient")
        self.assertEqual(diagnosis["provider_id"]["foreign_key"]["references_table"], "provider")

    def test_claim_foreign_keys(self):
        """claim should have FKs to patient, provider, and diagnosis."""
        claim = self.schema["claim"]
        for col, expected_table in [
            ("patient_id", "patient"),
            ("provider_id", "provider"),
            ("diagnosis_id", "diagnosis"),
        ]:
            with self.subTest(col=col):
                self.assertIn("foreign_key", claim[col])
                self.assertEqual(claim[col]["foreign_key"]["references_table"], expected_table)

    def test_claim_amount_is_decimal(self):
        """claim_amount (DECIMAL) should map to 'decimal' in the inferred schema."""
        self.assertEqual(self.schema["claim"]["claim_amount"]["type"], "decimal")

    def test_full_pipeline_produces_valid_syda_schema(self):
        """End-to-end: extract → convert → validate should pass for all tables."""
        syda_schema = to_syda_dict(self.schema)
        passed, results = validate_syda_compatibility(syda_schema)
        self.assertTrue(passed, f"Validation failed: {results}")
        self.assertEqual(len(results), 6)
        for r in results:
            with self.subTest(table=r["table"]):
                self.assertIn(r["status"], ("PASS", "WARN"))

# 10. Edge Cases

class TestEdgeCases(unittest.TestCase):
    """Edge case tests: empty tables, null-only columns, no tables in DB."""

    def test_empty_table_produces_schema_entry(self):
        """A table with no rows should still yield a valid schema dictionary."""
        engine = make_engine([
            "CREATE TABLE empty (id INTEGER PRIMARY KEY, note TEXT)"
        ])
        schema = extract_schema(engine)
        self.assertIn("empty", schema)
        self.assertIn("id", schema["empty"])

    def test_null_only_column_has_correct_type(self):
        """A nullable column (even if all values are NULL) should still map a type."""
        engine = make_engine([
            "CREATE TABLE t (id INTEGER PRIMARY KEY, nullable_col TEXT)"
        ])
        schema = extract_schema(engine)
        self.assertEqual(schema["t"]["nullable_col"]["type"], "string")
        self.assertTrue(schema["t"]["nullable_col"]["nullable"])

    def test_empty_database_produces_empty_schema(self):
        """A database with no tables should produce an empty schema dict."""
        engine = make_engine()
        schema = extract_schema(engine)
        self.assertEqual(schema, {})

    def test_syda_dict_on_empty_schema(self):
        """to_syda_dict() on an empty schema should return an empty dict."""
        result = to_syda_dict({})
        self.assertEqual(result, {})

    def test_validate_skips_gracefully_on_empty(self):
        """validate_syda_compatibility on an empty schema should return (True, [])."""
        passed, results = validate_syda_compatibility({})
        self.assertTrue(passed)
        self.assertEqual(results, [])

# Entry point

if __name__ == "__main__":
    unittest.main(verbosity=2)

# ---------------------------------------------------------------------------
# 11. DB Schema Inference (db_schema_inference.py)
# ---------------------------------------------------------------------------

class TestDbSchemaInference(unittest.TestCase):
    """Tests for db_schema_inference.py — create_schemas_from_database()."""

    def setUp(self):
        """Create a shared in-memory engine with the healthcare schema."""
        self.engine = make_engine([
            "CREATE TABLE patient (patient_id INTEGER PRIMARY KEY, patient_name TEXT NOT NULL, age INTEGER, date_of_birth DATE)",
            "CREATE TABLE provider (provider_id INTEGER PRIMARY KEY, provider_name TEXT NOT NULL, specialty TEXT)",
            """CREATE TABLE diagnosis (
                diagnosis_id INTEGER PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                provider_id INTEGER NOT NULL,
                diagnosis_code TEXT,
                FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
                FOREIGN KEY (provider_id) REFERENCES provider(provider_id))"""
        ])
        import types as _types
        from db_schema_inference import (
            create_schemas_from_database,
            _infer_syda_type,
            _schema_to_yaml,
            patch_generator,
        )

        # Create a minimal mock generator to test the methods

        class MockGenerator:
            pass
        self.gen = MockGenerator()
        self.gen.create_schemas_from_database = _types.MethodType(create_schemas_from_database, self.gen)
        self.gen._infer_syda_type = _types.MethodType(_infer_syda_type, self.gen)
        self.gen._schema_to_yaml = _types.MethodType(_schema_to_yaml, self.gen)
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def test_yaml_files_created_for_all_tables(self):
        """One YAML file should be created per table."""
        schema_files = self.gen.create_schemas_from_database(
            connection_string_or_engine=self.engine,
            output_dir=self.tmpdir,
            format="yaml"
        )
        import os
        self.assertEqual(len(schema_files), 3)
        for path in schema_files.values():
            self.assertTrue(os.path.exists(path))

    def test_json_files_created_for_all_tables(self):
        """One JSON file should be created per table when format='json'."""
        schema_files = self.gen.create_schemas_from_database(
            connection_string_or_engine=self.engine,
            output_dir=self.tmpdir,
            format="json"
        )
        import os
        for path in schema_files.values():
            self.assertTrue(path.endswith(".json"))
            self.assertTrue(os.path.exists(path))

    def test_returns_dict_of_table_name_to_file_path(self):
        """Return value should be a dict keyed by table name."""
        schema_files = self.gen.create_schemas_from_database(
            connection_string_or_engine=self.engine,
            output_dir=self.tmpdir,
            format="yaml"
        )
        self.assertIn("patient", schema_files)
        self.assertIn("provider", schema_files)
        self.assertIn("diagnosis", schema_files)

    def test_specific_table_names_filter(self):
        """Only the specified tables should be inferred when table_names is set."""
        schema_files = self.gen.create_schemas_from_database(
            connection_string_or_engine=self.engine,
            output_dir=self.tmpdir,
            table_names=["patient"],
            format="yaml"
        )
        self.assertEqual(list(schema_files.keys()), ["patient"])

    def test_invalid_format_raises_value_error(self):
        """Passing an unsupported format should raise ValueError."""
        with self.assertRaises(ValueError):
            self.gen.create_schemas_from_database(
                connection_string_or_engine=self.engine,
                output_dir=self.tmpdir,
                format="xml"
            )

    def test_infer_syda_type_integer(self):
        """INTEGER SQL type should map to 'integer'."""
        self.assertEqual(self.gen._infer_syda_type("INTEGER"), "integer")

    def test_infer_syda_type_string(self):
        """TEXT SQL type should map to 'string'."""
        self.assertEqual(self.gen._infer_syda_type("TEXT"), "string")

    def test_infer_syda_type_date(self):
        """DATE SQL type should map to 'date'."""
        self.assertEqual(self.gen._infer_syda_type("DATE"), "date")

    def test_infer_syda_type_float(self):
        """DECIMAL SQL type should map to 'float'."""
        self.assertEqual(self.gen._infer_syda_type("DECIMAL(10,2)"), "float")

    def test_infer_syda_type_unknown_defaults_to_string(self):
        """Unknown SQL types should safely default to 'string'."""
        self.assertEqual(self.gen._infer_syda_type("GEOMETRY"), "string")

    def test_fk_column_type_is_foreign_key_in_yaml(self):
        """FK columns should have type 'foreign_key' in the output schema file."""
        import yaml
        schema_files = self.gen.create_schemas_from_database(
            connection_string_or_engine=self.engine,
            output_dir=self.tmpdir,
            format="yaml"
        )
        with open(schema_files["diagnosis"]) as f:
            content = yaml.safe_load(f)
        self.assertEqual(content["patient_id"]["type"], "foreign_key")
        self.assertEqual(content["patient_id"]["references"]["table"], "patient")
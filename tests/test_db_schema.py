import json
import os
import tempfile
import unittest

from sqlalchemy import create_engine, text

from syda.db_schema_loader import DatabaseSchemaLoader, _map_sql_type


def make_engine(sql_statements=None):
    engine = create_engine("sqlite:///:memory:")
    if sql_statements:
        with engine.connect() as conn:
            for stmt in sql_statements:
                conn.execute(text(stmt))
            conn.commit()
    return engine


def make_healthcare_engine():
    return make_engine([
        "CREATE TABLE patient (patient_id INTEGER PRIMARY KEY, patient_name TEXT NOT NULL, age INTEGER, date_of_birth DATE)",
        "CREATE TABLE provider (provider_id INTEGER PRIMARY KEY, provider_name TEXT NOT NULL, specialty TEXT, license_number TEXT, facility_id TEXT)",
        """CREATE TABLE diagnosis (
            diagnosis_id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL,
            provider_id INTEGER NOT NULL, diagnosis_code TEXT, visit_date DATE,
            FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
            FOREIGN KEY (provider_id) REFERENCES provider(provider_id))""",
        """CREATE TABLE claim (
            claim_id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL,
            provider_id INTEGER NOT NULL, diagnosis_id INTEGER NOT NULL,
            procedure_code TEXT, claim_amount DECIMAL(10,2), submission_date DATE,
            FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
            FOREIGN KEY (provider_id) REFERENCES provider(provider_id),
            FOREIGN KEY (diagnosis_id) REFERENCES diagnosis(diagnosis_id))""",
        """CREATE TABLE adjudication (
            adjudication_id INTEGER PRIMARY KEY, claim_id INTEGER NOT NULL,
            decision TEXT, denial_reason TEXT, approved_amount DECIMAL(10,2),
            FOREIGN KEY (claim_id) REFERENCES claim(claim_id))""",
        """CREATE TABLE payment (
            payment_id INTEGER PRIMARY KEY, claim_id INTEGER NOT NULL,
            payment_date DATE, payment_amount DECIMAL(10,2), status TEXT,
            FOREIGN KEY (claim_id) REFERENCES claim(claim_id))""",
    ])


# 1. Type mapping

class TestMapSqlType(unittest.TestCase):

    def test_integer_variants(self):
        for t in ["INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"]:
            with self.subTest(t=t):
                self.assertEqual(_map_sql_type(t), "integer")

    def test_string_variants(self):
        for t in ["TEXT", "VARCHAR(255)", "CHAR(10)", "NVARCHAR(100)", "CLOB"]:
            with self.subTest(t=t):
                self.assertEqual(_map_sql_type(t), "string")

    def test_date_variants(self):
        for t in ["DATE", "DATETIME", "TIMESTAMP", "TIME"]:
            with self.subTest(t=t):
                self.assertEqual(_map_sql_type(t), "date")

    def test_float_variants(self):
        for t in ["DECIMAL(10,2)", "NUMERIC(8,4)", "FLOAT", "REAL", "DOUBLE"]:
            with self.subTest(t=t):
                self.assertEqual(_map_sql_type(t), "float")

    def test_boolean(self):
        self.assertEqual(_map_sql_type("BOOLEAN"), "boolean")

    def test_unknown_defaults_to_string(self):
        for t in ["GEOMETRY", "BLOB", "JSON", "UUID", "XML"]:
            with self.subTest(t=t):
                self.assertEqual(_map_sql_type(t), "string")

    def test_case_insensitive(self):
        self.assertEqual(_map_sql_type("integer"), "integer")
        self.assertEqual(_map_sql_type("Integer"), "integer")
        self.assertEqual(_map_sql_type("VARCHAR"), "string")

    def test_returns_integer_not_int(self):
        result = _map_sql_type("INTEGER")
        self.assertEqual(result, "integer")
        self.assertNotEqual(result, "int")


# 2. DatabaseSchemaLoader — connection

class TestDatabaseConnection(unittest.TestCase):

    def test_valid_connection_string(self):
        loader = DatabaseSchemaLoader("sqlite:///:memory:")
        self.assertIsNotNone(loader._inspector)

    def test_valid_engine_object(self):
        engine = make_engine()
        loader = DatabaseSchemaLoader(engine)
        self.assertIsNotNone(loader._inspector)

    def test_missing_sqlalchemy_import(self):
        import unittest.mock as mock
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sqlalchemy":
                raise ImportError
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with self.assertRaises(ImportError):
                DatabaseSchemaLoader("sqlite:///:memory:")


# 3. Metadata extraction

class TestMetadataExtraction(unittest.TestCase):

    def setUp(self):
        self.engine = make_engine([
            """CREATE TABLE patient (
                patient_id INTEGER PRIMARY KEY,
                patient_name TEXT NOT NULL,
                age INTEGER,
                date_of_birth DATE
            )"""
        ])
        self.loader = DatabaseSchemaLoader(self.engine)

    def test_table_appears_in_schemas(self):
        schemas = self.loader.load_schemas()
        self.assertIn("patient", schemas)

    def test_all_columns_extracted(self):
        schema = self.loader.load_schemas()["patient"]
        for col in ["patient_id", "patient_name", "age", "date_of_birth"]:
            with self.subTest(col=col):
                self.assertIn(col, schema)

    def test_column_types_mapped(self):
        schema = self.loader.load_schemas()["patient"]
        self.assertEqual(schema["patient_id"]["type"], "integer")
        self.assertEqual(schema["patient_name"]["type"], "string")
        self.assertEqual(schema["date_of_birth"]["type"], "date")

    def test_empty_table_extracted(self):
        engine = make_engine(["CREATE TABLE empty_table (id INTEGER PRIMARY KEY, val TEXT)"])
        schema = DatabaseSchemaLoader(engine).load_schemas()
        self.assertIn("empty_table", schema)
        self.assertIn("id", schema["empty_table"])

    def test_multiple_tables_extracted(self):
        engine = make_engine([
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY)",
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY)",
            "CREATE TABLE t3 (id INTEGER PRIMARY KEY)",
        ])
        schemas = DatabaseSchemaLoader(engine).load_schemas()
        for t in ["t1", "t2", "t3"]:
            self.assertIn(t, schemas)

    def test_empty_database_returns_empty_dict(self):
        schemas = DatabaseSchemaLoader(make_engine()).load_schemas()
        self.assertEqual(schemas, {})


# 4. Primary key extraction

class TestPrimaryKeyExtraction(unittest.TestCase):

    def test_single_column_pk(self):
        engine = make_engine(["CREATE TABLE t (pk_col INTEGER PRIMARY KEY, data TEXT)"])
        schema = DatabaseSchemaLoader(engine).load_schemas()["t"]
        self.assertTrue(schema["pk_col"].get("primary_key"))
        self.assertFalse(schema["data"].get("primary_key", False))

    def test_composite_pk_both_columns_marked(self):
        engine = make_engine([
            """CREATE TABLE composite (
                col_a INTEGER NOT NULL, col_b INTEGER NOT NULL, val TEXT,
                PRIMARY KEY (col_a, col_b))"""
        ])
        schema = DatabaseSchemaLoader(engine).load_schemas()["composite"]
        self.assertTrue(schema["col_a"].get("primary_key"))
        self.assertTrue(schema["col_b"].get("primary_key"))
        self.assertFalse(schema["val"].get("primary_key", False))

    def test_non_pk_column_has_no_primary_key_key(self):
        engine = make_engine(["CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)"])
        schema = DatabaseSchemaLoader(engine).load_schemas()["t"]
        self.assertNotIn("primary_key", schema["name"])


# 5. Foreign key extraction

class TestForeignKeyExtraction(unittest.TestCase):

    def setUp(self):
        self.engine = make_engine([
            "CREATE TABLE parent (parent_id INTEGER PRIMARY KEY, name TEXT)",
            """CREATE TABLE child (
                child_id INTEGER PRIMARY KEY, parent_id INTEGER NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES parent(parent_id))"""
        ])
        self.schemas = DatabaseSchemaLoader(self.engine).load_schemas()

    def test_fk_column_type_is_foreign_key(self):
        self.assertEqual(self.schemas["child"]["parent_id"]["type"], "foreign_key")

    def test_fk_references_correct_table(self):
        refs = self.schemas["child"]["parent_id"]["references"]
        self.assertEqual(refs["schema"], "parent")

    def test_fk_references_correct_column(self):
        refs = self.schemas["child"]["parent_id"]["references"]
        self.assertEqual(refs["field"], "parent_id")

    def test_non_fk_column_has_no_references(self):
        self.assertNotIn("references", self.schemas["child"]["child_id"])

    def test_parent_table_has_no_fk(self):
        for col_def in self.schemas["parent"].values():
            self.assertNotEqual(col_def.get("type"), "foreign_key")

    def test_fk_uses_schema_key_not_table(self):
        """SchemaLoader._load_dict_schema() expects 'schema', not 'table'."""
        refs = self.schemas["child"]["parent_id"]["references"]
        self.assertIn("schema", refs)
        self.assertNotIn("table", refs)


# 6. table_names filter

class TestTableNamesFilter(unittest.TestCase):

    def setUp(self):
        self.engine = make_engine([
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY)",
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY)",
            "CREATE TABLE t3 (id INTEGER PRIMARY KEY)",
        ])
        self.loader = DatabaseSchemaLoader(self.engine)

    def test_specific_tables_only(self):
        schemas = self.loader.load_schemas(table_names=["t1", "t2"])
        self.assertIn("t1", schemas)
        self.assertIn("t2", schemas)
        self.assertNotIn("t3", schemas)

    def test_missing_table_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.loader.load_schemas(table_names=["nonexistent"])

    def test_none_returns_all_tables(self):
        schemas = self.loader.load_schemas(table_names=None)
        self.assertEqual(set(schemas.keys()), {"t1", "t2", "t3"})


# 7. save_schemas file output

class TestSaveSchemas(unittest.TestCase):

    def setUp(self):
        self.engine = make_engine([
            "CREATE TABLE patient (patient_id INTEGER PRIMARY KEY, patient_name TEXT NOT NULL)",
            "CREATE TABLE provider (provider_id INTEGER PRIMARY KEY, provider_name TEXT NOT NULL)",
        ])
        self.loader = DatabaseSchemaLoader(self.engine)
        self.tmpdir = tempfile.mkdtemp()

    def test_yaml_files_created(self):
        result = self.loader.save_schemas(self.tmpdir, format="yaml")
        for path in result.values():
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".yaml"))

    def test_json_files_created(self):
        result = self.loader.save_schemas(self.tmpdir, format="json")
        for path in result.values():
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".json"))

    def test_returns_dict_of_table_to_path(self):
        result = self.loader.save_schemas(self.tmpdir, format="yaml")
        self.assertIn("patient", result)
        self.assertIn("provider", result)

    def test_invalid_format_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.loader.save_schemas(self.tmpdir, format="xml")

    def test_output_dir_created_if_missing(self):
        new_dir = os.path.join(self.tmpdir, "subdir")
        self.loader.save_schemas(new_dir, format="json")
        self.assertTrue(os.path.isdir(new_dir))

    def test_json_content_is_valid(self):
        result = self.loader.save_schemas(self.tmpdir, format="json")
        with open(result["patient"]) as f:
            data = json.load(f)
        self.assertIn("patient_id", data)

    def test_table_names_filter_applies_to_save(self):
        result = self.loader.save_schemas(self.tmpdir, table_names=["patient"], format="json")
        self.assertIn("patient", result)
        self.assertNotIn("provider", result)


# 8. Full healthcare integration

class TestHealthcareIntegration(unittest.TestCase):

    def setUp(self):
        self.loader = DatabaseSchemaLoader(make_healthcare_engine())
        self.schemas = self.loader.load_schemas()

    def test_all_six_tables_present(self):
        expected = {"patient", "provider", "diagnosis", "claim", "adjudication", "payment"}
        self.assertEqual(set(self.schemas.keys()), expected)

    def test_patient_primary_key(self):
        self.assertTrue(self.schemas["patient"]["patient_id"].get("primary_key"))

    def test_diagnosis_fk_to_patient(self):
        refs = self.schemas["diagnosis"]["patient_id"]["references"]
        self.assertEqual(refs["schema"], "patient")

    def test_diagnosis_fk_to_provider(self):
        refs = self.schemas["diagnosis"]["provider_id"]["references"]
        self.assertEqual(refs["schema"], "provider")

    def test_claim_fks(self):
        claim = self.schemas["claim"]
        for col, expected_table in [
            ("patient_id", "patient"),
            ("provider_id", "provider"),
            ("diagnosis_id", "diagnosis"),
        ]:
            with self.subTest(col=col):
                self.assertEqual(claim[col]["references"]["schema"], expected_table)

    def test_claim_amount_is_float(self):
        self.assertEqual(self.schemas["claim"]["claim_amount"]["type"], "float")

    def test_fk_columns_do_not_have_duplicate_type(self):
        for table, schema in self.schemas.items():
            for col, col_def in schema.items():
                if col_def.get("type") == "foreign_key":
                    self.assertIn("references", col_def, f"{table}.{col} missing references")

    def test_not_null_propagated_for_pk(self):
        self.assertTrue(self.schemas["patient"]["patient_id"].get("not_null"))


# 9. Edge cases

class TestEdgeCases(unittest.TestCase):

    def test_empty_table_produces_schema_entry(self):
        engine = make_engine(["CREATE TABLE empty (id INTEGER PRIMARY KEY, note TEXT)"])
        schemas = DatabaseSchemaLoader(engine).load_schemas()
        self.assertIn("empty", schemas)
        self.assertIn("id", schemas["empty"])

    def test_nullable_column_no_not_null(self):
        engine = make_engine(["CREATE TABLE t (id INTEGER PRIMARY KEY, nullable_col TEXT)"])
        schema = DatabaseSchemaLoader(engine).load_schemas()["t"]
        self.assertNotIn("not_null", schema["nullable_col"])

    def test_non_null_non_pk_column(self):
        engine = make_engine(["CREATE TABLE t (id INTEGER PRIMARY KEY, required TEXT NOT NULL)"])
        schema = DatabaseSchemaLoader(engine).load_schemas()["t"]
        self.assertTrue(schema["required"].get("not_null"))

    def test_empty_database(self):
        schemas = DatabaseSchemaLoader(make_engine()).load_schemas()
        self.assertEqual(schemas, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)

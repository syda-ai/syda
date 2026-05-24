"""Tests for the syda CLI (syda.cli)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from syda.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema_file(tmp_path: Path, name: str = "patients", ext: str = ".yaml") -> Path:
    schema = {
        "id": {"type": "integer", "description": "Patient ID"},
        "name": {"type": "text", "description": "Full name"},
        "age": {"type": "integer", "description": "Age in years"},
    }
    p = tmp_path / f"{name}{ext}"
    if ext in (".yaml", ".yml"):
        p.write_text(yaml.dump(schema))
    else:
        p.write_text(json.dumps(schema))
    return p


def _fake_generate_for_schemas(schemas, **kwargs):
    """Return a minimal DataFrame for every schema, matching the real API."""
    result = {}
    for name in schemas:
        result[name] = pd.DataFrame(
            [{"id": 1, "name": "Alice", "age": 30}]
        )
    return result


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

class TestVersionCommand:
    def test_prints_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert "syda" in result.output


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class TestValidateCommand:
    def test_valid_yaml_schema(self, tmp_path):
        schema_file = _make_schema_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--schema", str(schema_file)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_valid_json_schema(self, tmp_path):
        schema_file = _make_schema_file(tmp_path, ext=".json")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--schema", str(schema_file)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_valid_directory_of_schemas(self, tmp_path):
        _make_schema_file(tmp_path, name="patients")
        _make_schema_file(tmp_path, name="claims")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--schema", str(tmp_path)])
        assert result.exit_code == 0
        assert "2 schema(s) valid" in result.output

    def test_missing_path_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--schema", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_wrong_extension_errors(self, tmp_path):
        bad = tmp_path / "schema.txt"
        bad.write_text("id: integer")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--schema", str(bad)])
        assert result.exit_code != 0

    def test_empty_directory_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--schema", str(tmp_path)])
        assert result.exit_code != 0

    def test_invalid_schema_reports_error(self, tmp_path):
        bad_schema = tmp_path / "bad.yaml"
        bad_schema.write_text("id:\n  type: not_a_real_type\n")
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "--schema", str(bad_schema)])
        assert result.exit_code != 0
        assert "FAIL" in result.output


# ---------------------------------------------------------------------------
# generate — option validation (no actual LLM calls)
# ---------------------------------------------------------------------------

class TestGenerateOptionValidation:
    def test_output_and_multi_schema_errors(self, tmp_path):
        _make_schema_file(tmp_path, name="a")
        _make_schema_file(tmp_path, name="b")
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--schema", str(tmp_path),
                "--output", str(tmp_path / "out.csv"),
                "--provider", "anthropic",
            ],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code != 0
        assert "--output" in result.output

    def test_output_and_output_dir_errors(self, tmp_path):
        schema_file = _make_schema_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--schema", str(schema_file),
                "--output", str(tmp_path / "out.csv"),
                "--output-dir", str(tmp_path),
                "--provider", "anthropic",
            ],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code != 0

    def test_no_provider_no_env_errors(self, tmp_path):
        schema_file = _make_schema_file(tmp_path)
        runner = CliRunner()
        env = {k: "" for k in [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
            "GEMINI_API_KEY", "GROK_API_KEY",
        ]}
        result = runner.invoke(
            main,
            ["generate", "--schema", str(schema_file)],
            env=env,
        )
        assert result.exit_code != 0
        assert "provider" in result.output.lower()

    def test_missing_schema_errors(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--schema", str(tmp_path / "nope.yaml"), "--provider", "anthropic"],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# generate — CSV/JSON output (mocked generator)
# ---------------------------------------------------------------------------

class TestGenerateOutput:
    @patch("syda.generate.SyntheticDataGenerator")
    def test_single_schema_csv_output(self, MockGen, tmp_path):
        instance = MockGen.return_value
        instance.generate_for_schemas.side_effect = _fake_generate_for_schemas

        schema_file = _make_schema_file(tmp_path)
        out_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--schema", str(schema_file),
                "--output", str(out_file),
                "--rows", "1",
                "--provider", "anthropic",
            ],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()
        df = pd.read_csv(out_file)
        assert len(df) == 1
        assert "name" in df.columns

    @patch("syda.generate.SyntheticDataGenerator")
    def test_single_schema_json_output(self, MockGen, tmp_path):
        instance = MockGen.return_value
        instance.generate_for_schemas.side_effect = _fake_generate_for_schemas

        schema_file = _make_schema_file(tmp_path)
        out_file = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--schema", str(schema_file),
                "--output", str(out_file),
                "--rows", "1",
                "--provider", "openai",
            ],
            env={"OPENAI_API_KEY": "fake"},
        )
        assert result.exit_code == 0, result.output
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert isinstance(data, list)
        assert data[0]["name"] == "Alice"

    @patch("syda.generate.SyntheticDataGenerator")
    def test_format_flag_overrides_extension(self, MockGen, tmp_path):
        instance = MockGen.return_value
        instance.generate_for_schemas.side_effect = _fake_generate_for_schemas

        schema_file = _make_schema_file(tmp_path)
        out_file = tmp_path / "out.csv"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--schema", str(schema_file),
                "--output", str(out_file),
                "--format", "json",
                "--rows", "1",
                "--provider", "anthropic",
            ],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code == 0, result.output
        # --format json overrides the .csv extension
        data = json.loads(out_file.read_text())
        assert isinstance(data, list)

    @patch("syda.generate.SyntheticDataGenerator")
    def test_output_dir_multi_schema(self, MockGen, tmp_path):
        instance = MockGen.return_value
        instance.generate_for_schemas.side_effect = _fake_generate_for_schemas

        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        _make_schema_file(schema_dir, name="patients")
        _make_schema_file(schema_dir, name="claims")

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--schema", str(schema_dir),
                "--output-dir", str(out_dir),
                "--rows", "1",
                "--provider", "anthropic",
            ],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code == 0, result.output

    @patch("syda.generate.SyntheticDataGenerator")
    def test_provider_auto_detected_from_env(self, MockGen, tmp_path):
        instance = MockGen.return_value
        instance.generate_for_schemas.side_effect = _fake_generate_for_schemas

        schema_file = _make_schema_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--schema", str(schema_file), "--rows", "1"],
            env={"ANTHROPIC_API_KEY": "fake_key"},
        )
        assert result.exit_code == 0, result.output
        assert "anthropic" in result.output

    @patch("syda.generate.SyntheticDataGenerator")
    def test_generation_failure_exits_nonzero(self, MockGen, tmp_path):
        instance = MockGen.return_value
        instance.generate_for_schemas.side_effect = RuntimeError("LLM timeout")

        schema_file = _make_schema_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--schema", str(schema_file), "--rows", "1", "--provider", "openai"],
            env={"OPENAI_API_KEY": "fake"},
        )
        assert result.exit_code != 0
        assert "Generation failed" in result.output


# ---------------------------------------------------------------------------
# db infer
# ---------------------------------------------------------------------------

class TestDbInferCommand:
    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_infer_writes_schema_files(self, MockLoader, tmp_path):
        out_dir = tmp_path / "schemas"
        out_dir.mkdir()
        instance = MockLoader.return_value
        instance.save_schemas.return_value = {
            "patients": str(out_dir / "patients.yaml"),
            "claims": str(out_dir / "claims.yaml"),
        }

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "db", "infer",
                "--db-url", "sqlite:///fake.db",
                "--output-dir", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "patients" in result.output
        assert "claims" in result.output
        instance.save_schemas.assert_called_once_with(
            output_dir=str(out_dir),
            table_names=None,
            format="yaml",
        )

    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_infer_specific_tables(self, MockLoader, tmp_path):
        out_dir = tmp_path / "schemas"
        out_dir.mkdir()
        instance = MockLoader.return_value
        instance.save_schemas.return_value = {"patients": str(out_dir / "patients.yaml")}

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "db", "infer",
                "--db-url", "sqlite:///fake.db",
                "--output-dir", str(out_dir),
                "--tables", "patients",
                "--format", "json",
            ],
        )
        assert result.exit_code == 0, result.output
        instance.save_schemas.assert_called_once_with(
            output_dir=str(out_dir),
            table_names=["patients"],
            format="json",
        )

    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_infer_connection_failure(self, MockLoader, tmp_path):
        MockLoader.side_effect = Exception("connection refused")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["db", "infer", "--db-url", "sqlite:///bad.db", "--output-dir", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert "connect" in result.output.lower()


# ---------------------------------------------------------------------------
# db generate
# ---------------------------------------------------------------------------

class TestDbGenerateCommand:
    def _fake_results(self):
        return {
            "patients": pd.DataFrame([{"id": 1, "name": "Alice"}]),
            "claims": pd.DataFrame([{"id": 1, "patient_id": 1}]),
        }

    @patch("syda.generate.SyntheticDataGenerator")
    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_generate_from_db_csv(self, MockLoader, MockGen, tmp_path):
        loader_instance = MockLoader.return_value
        loader_instance.load_schemas.return_value = {
            "patients": {"id": {"type": "integer"}, "name": {"type": "text"}},
        }
        gen_instance = MockGen.return_value
        gen_instance.generate_for_schemas.return_value = self._fake_results()

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "db", "generate",
                "--db-url", "sqlite:///fake.db",
                "--rows", "5",
                "--output-dir", str(out_dir),
                "--provider", "anthropic",
            ],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code == 0, result.output
        loader_instance.load_schemas.assert_called_once_with(table_names=None)
        assert "patients" in result.output or "claims" in result.output

    @patch("syda.generate.SyntheticDataGenerator")
    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_generate_with_write_back(self, MockLoader, MockGen, tmp_path):
        loader_instance = MockLoader.return_value
        loader_instance.load_schemas.return_value = {
            "patients": {"id": {"type": "integer"}},
        }
        gen_instance = MockGen.return_value
        gen_instance.generate_for_schemas.return_value = self._fake_results()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "db", "generate",
                "--db-url", "sqlite:///fake.db",
                "--rows", "5",
                "--write-back",
                "--if-exists", "replace",
                "--provider", "openai",
            ],
            env={"OPENAI_API_KEY": "fake"},
        )
        assert result.exit_code == 0, result.output
        loader_instance.write_to_database.assert_called_once()
        call_kwargs = loader_instance.write_to_database.call_args
        assert call_kwargs.kwargs.get("if_exists") == "replace" or call_kwargs.args[1] == "replace"

    @patch("syda.generate.SyntheticDataGenerator")
    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_generate_specific_tables(self, MockLoader, MockGen, tmp_path):
        loader_instance = MockLoader.return_value
        loader_instance.load_schemas.return_value = {
            "patients": {"id": {"type": "integer"}},
        }
        gen_instance = MockGen.return_value
        gen_instance.generate_for_schemas.return_value = {
            "patients": pd.DataFrame([{"id": 1}]),
        }

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "db", "generate",
                "--db-url", "sqlite:///fake.db",
                "--tables", "patients",
                "--rows", "3",
                "--provider", "anthropic",
            ],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code == 0, result.output
        loader_instance.load_schemas.assert_called_once_with(table_names=["patients"])

    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_db_connection_failure(self, MockLoader, tmp_path):
        MockLoader.side_effect = Exception("bad credentials")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["db", "generate", "--db-url", "mysql://bad", "--provider", "anthropic"],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code != 0
        assert "connect" in result.output.lower()

    @patch("syda.generate.SyntheticDataGenerator")
    @patch("syda.db_schema_loader.DatabaseSchemaLoader")
    def test_generation_failure_exits_nonzero(self, MockLoader, MockGen, tmp_path):
        loader_instance = MockLoader.return_value
        loader_instance.load_schemas.return_value = {"t": {"id": {"type": "integer"}}}
        gen_instance = MockGen.return_value
        gen_instance.generate_for_schemas.side_effect = RuntimeError("timeout")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["db", "generate", "--db-url", "sqlite:///x.db", "--provider", "anthropic"],
            env={"ANTHROPIC_API_KEY": "fake"},
        )
        assert result.exit_code != 0
        assert "Generation failed" in result.output

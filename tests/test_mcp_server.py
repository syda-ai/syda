"""Tests for the Syda MCP server tools."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_report(tables):
    """Build a minimal RunReport-like mock."""
    report = MagicMock()
    report.estimated_cost_usd = sum(t["cost"] for t in tables.values())
    report.total_duration_s = 1.5
    table_mocks = {}
    for name, meta in tables.items():
        t = MagicMock()
        t.row_count = meta["rows"]
        t.mode = meta.get("mode", "direct")
        t.llm_calls = meta.get("calls", 1)
        t.input_tokens = meta.get("in_tok", 100)
        t.output_tokens = meta.get("out_tok", 200)
        t.cost_usd = meta.get("cost", 0.01)
        t.duration_s = meta.get("dur", 1.0)
        table_mocks[name] = t
    report.tables = table_mocks
    return report


# ── validate_schema ───────────────────────────────────────────────────────────

class TestValidateSchema:

    def _call(self, schema):
        from syda.mcp_server import validate_schema
        return validate_schema(schema)

    def test_valid_simple_schema(self):
        schema = {
            "users": {
                "id":    {"type": "integer", "primary_key": True},
                "email": {"type": "email", "unique": True},
            }
        }
        result = self._call(schema)
        assert result["ok"] is True
        assert "users" in result["tables"]
        assert result["errors"] == []

    def test_valid_fk_schema(self):
        schema = {
            "customers": {
                "customer_id": {"type": "integer", "primary_key": True},
                "name":        {"type": "string"},
            },
            "orders": {
                "order_id":   {"type": "integer", "primary_key": True},
                "customer_id": {
                    "type": "integer",
                    "foreign_key": {"table": "customers", "column": "customer_id"},
                },
            },
        }
        result = self._call(schema)
        assert result["ok"] is True
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["child"] == "orders.customer_id"
        assert result["relationships"][0]["parent"] == "customers.customer_id"

    def test_invalid_fk_missing_table(self):
        schema = {
            "orders": {
                "order_id":   {"type": "integer", "primary_key": True},
                "customer_id": {
                    "type": "integer",
                    "foreign_key": {"table": "nonexistent", "column": "id"},
                },
            },
        }
        result = self._call(schema)
        assert result["ok"] is False
        assert any("nonexistent" in e for e in result["errors"])

    def test_invalid_fk_missing_table_and_column_keys(self):
        schema = {
            "orders": {
                "customer_id": {
                    "type": "foreign_key",
                    "foreign_key": {"table": "customers"},  # missing "column"
                },
            },
        }
        result = self._call(schema)
        assert result["ok"] is False

    def test_no_primary_key_warns(self):
        schema = {
            "logs": {
                "message": {"type": "string"},
            }
        }
        result = self._call(schema)
        assert result["ok"] is True   # warning not error
        assert any("primary_key" in w for w in result["warnings"])

    def test_unknown_type_warns(self):
        schema = {
            "things": {
                "id":   {"type": "integer", "primary_key": True},
                "data": {"type": "jsonb"},   # unknown type
            }
        }
        result = self._call(schema)
        assert result["ok"] is True
        assert any("jsonb" in w for w in result["warnings"])

    def test_enum_column_detected(self):
        schema = {
            "users": {
                "id":   {"type": "integer", "primary_key": True},
                "plan": {"type": "string", "enum": ["free", "pro"]},
            }
        }
        result = self._call(schema)
        assert result["ok"] is True
        plan_col = next(
            c for c in result["tables"]["users"]["column_details"]
            if c["name"] == "plan"
        )
        assert plan_col["has_enum"] is True

    def test_empty_schema(self):
        result = self._call({})
        assert result["ok"] is True
        assert result["tables"] == {}

    def test_message_on_success(self):
        schema = {"t": {"id": {"type": "integer", "primary_key": True}}}
        result = self._call(schema)
        assert "valid" in result["message"].lower()

    def test_message_on_error(self):
        schema = {
            "orders": {
                "cid": {"type": "integer", "foreign_key": {"table": "missing", "column": "id"}},
            }
        }
        result = self._call(schema)
        assert "error" in result["message"].lower()


# ── get_providers ─────────────────────────────────────────────────────────────

class TestGetProviders:

    def _call(self):
        from syda.mcp_server import get_providers
        return get_providers()

    def test_returns_ok(self):
        result = self._call()
        assert result["ok"] is True

    def test_all_expected_providers_present(self):
        result = self._call()
        names = [p["provider"] for p in result["providers"]]
        assert "anthropic" in names
        assert "openai" in names
        assert "gemini" in names
        assert "grok" in names
        assert "openai_compatible" in names

    def test_configured_reflects_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            from syda.mcp_server import get_providers
            result = get_providers()
        assert "anthropic" in result["configured"]

    def test_not_configured_without_env(self):
        import os
        env_backup = {k: os.environ.pop(k) for k in [
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROK_API_KEY"
        ] if k in os.environ}
        try:
            from syda.mcp_server import get_providers
            result = get_providers()
            assert result["configured"] == [] or all(
                p not in result["configured"]
                for p in ["anthropic", "openai", "gemini", "grok"]
            )
        finally:
            os.environ.update(env_backup)

    def test_each_provider_has_required_fields(self):
        result = self._call()
        for p in result["providers"]:
            assert "provider" in p
            assert "configured" in p
            assert "env_var" in p
            assert "recommended_model" in p


# ── get_run_report ────────────────────────────────────────────────────────────

class TestGetRunReport:

    def test_returns_hint(self):
        from syda.mcp_server import get_run_report
        result = get_run_report()
        assert result["ok"] is False
        assert "report" in result["hint"].lower() or "generate_from_schema" in result["hint"]


# ── generate_from_schema ──────────────────────────────────────────────────────

class TestGenerateFromSchema:

    SIMPLE_SCHEMA = {
        "customers": {
            "customer_id": {"type": "integer", "primary_key": True},
            "name":        {"type": "string"},
            "email":       {"type": "email", "unique": True},
        }
    }

    def _make_df(self):
        return pd.DataFrame({
            "customer_id": [1, 2, 3],
            "name":        ["Alice", "Bob", "Charlie"],
            "email":       ["a@x.com", "b@x.com", "c@x.com"],
        })

    def _mock_generator(self, df):
        gen = MagicMock()
        gen.model_config.provider = "anthropic"
        gen.model_config.model_name = "claude-haiku-4-5-20251001"
        gen.last_report = _make_report({
            "customers": {"rows": 3, "calls": 1, "in_tok": 100, "out_tok": 200, "cost": 0.001}
        })
        gen.generate_for_schemas.return_value = {"customers": df}
        return gen

    def test_success_returns_ok(self):
        df = self._make_df()
        with patch("syda.mcp_server._build_generator", return_value=self._mock_generator(df)), \
             patch("syda.schema_loader.SchemaLoader") as MockLoader:
            MockLoader.return_value.load_schema.return_value = self.SIMPLE_SCHEMA["customers"]
            from syda.mcp_server import generate_from_schema
            result = generate_from_schema(
                schema=self.SIMPLE_SCHEMA,
                default_sample_size=3,
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
            )
        assert result["ok"] is True
        assert "customers" in result["tables"]
        assert result["row_counts"]["customers"] == 3

    def test_preview_rows_respected(self):
        df = self._make_df()
        with patch("syda.mcp_server._build_generator", return_value=self._mock_generator(df)), \
             patch("syda.schema_loader.SchemaLoader") as MockLoader:
            MockLoader.return_value.load_schema.return_value = self.SIMPLE_SCHEMA["customers"]
            from syda.mcp_server import generate_from_schema
            result = generate_from_schema(
                schema=self.SIMPLE_SCHEMA,
                default_sample_size=3,
                preview_rows=2,
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
            )
        assert len(result["tables"]["customers"]) <= 2

    def test_report_included(self):
        df = self._make_df()
        with patch("syda.mcp_server._build_generator", return_value=self._mock_generator(df)), \
             patch("syda.schema_loader.SchemaLoader") as MockLoader:
            MockLoader.return_value.load_schema.return_value = self.SIMPLE_SCHEMA["customers"]
            from syda.mcp_server import generate_from_schema
            result = generate_from_schema(
                schema=self.SIMPLE_SCHEMA,
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
            )
        assert "report" in result
        assert "estimated_cost_usd" in result["report"]
        assert "total_llm_calls" in result["report"]
        assert "per_table" in result["report"]

    def test_error_on_bad_provider(self):
        with patch("syda.mcp_server._build_generator", side_effect=ValueError("No API key")):
            from syda.mcp_server import generate_from_schema
            result = generate_from_schema(schema=self.SIMPLE_SCHEMA)
        assert result["ok"] is False
        assert "error" in result
        assert "suggestion" in result

    def test_message_in_response(self):
        df = self._make_df()
        with patch("syda.mcp_server._build_generator", return_value=self._mock_generator(df)), \
             patch("syda.schema_loader.SchemaLoader") as MockLoader:
            MockLoader.return_value.load_schema.return_value = self.SIMPLE_SCHEMA["customers"]
            from syda.mcp_server import generate_from_schema
            result = generate_from_schema(
                schema=self.SIMPLE_SCHEMA,
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
            )
        assert "message" in result
        assert "3" in result["message"]   # row count mentioned


# ── infer_schema_from_db ──────────────────────────────────────────────────────

class TestInferSchemaFromDb:

    def test_success(self):
        mock_schemas = {
            "customers": {
                "customer_id": {"type": "integer", "primary_key": True},
                "email": {"type": "string"},
            }
        }
        with patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = mock_schemas
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db("sqlite:///test.db")

        assert result["ok"] is True
        assert "customers" in result["schema"]
        assert result["tables"] == ["customers"]

    def test_error_bad_url(self):
        with patch("syda.DatabaseSchemaLoader",
                   side_effect=Exception("could not connect")):
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db("bad://url")
        assert result["ok"] is False
        assert "suggestion" in result

    def test_table_filter_passed_through(self):
        with patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = {}
            from syda.mcp_server import infer_schema_from_db
            infer_schema_from_db("sqlite:///test.db", tables=["customers"])
            MockLoader.return_value.load_schemas.assert_called_once_with(
                table_names=["customers"]
            )

    def test_fk_relationships_extracted(self):
        mock_schemas = {
            "customers": {"id": {"type": "integer", "primary_key": True}},
            "orders": {
                "id": {"type": "integer", "primary_key": True},
                "customer_id": {
                    "type": "integer",
                    "foreign_key": {"table": "customers", "column": "id"},
                },
            },
        }
        with patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = mock_schemas
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db("sqlite:///test.db")

        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["child"] == "orders.customer_id"

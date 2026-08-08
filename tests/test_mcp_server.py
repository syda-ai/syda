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


# ── credential redaction ─────────────────────────────────────────────────────

class TestRedactUrlCredentials:

    def _redact(self, text):
        from syda.mcp_server import _redact_url_credentials
        return _redact_url_credentials(text)

    def test_redacts_password_in_url(self):
        text = "connection failed: postgresql://john:hunter2@db.internal:5432/app"
        redacted = self._redact(text)
        assert "hunter2" not in redacted
        assert "john" not in redacted
        assert "postgresql://***:***@db.internal:5432/app" in redacted

    def test_leaves_url_without_credentials_untouched(self):
        text = "connection failed: postgresql://db.internal:5432/app"
        assert self._redact(text) == text

    def test_leaves_plain_text_untouched(self):
        text = "table 'orders' does not exist"
        assert self._redact(text) == text

    def test_tool_error_uses_redaction(self):
        from syda.mcp_server import _tool_error
        exc = Exception("dsn: mysql://root:s3cr3t@localhost/db")
        result = _tool_error(exc, "check your config")
        assert "s3cr3t" not in result["message"]


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
                    "type": "foreign_key",
                    "references": {"schema": "customers", "field": "customer_id"},
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
                    "type": "foreign_key",
                    "references": {"schema": "nonexistent", "field": "id"},
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
                    "references": {"schema": "customers"},  # missing "field"
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
                "cid": {"type": "foreign_key", "references": {"schema": "missing", "field": "id"}},
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


# ── FK schema format consistency ────────────────────────────────────────────
#
# generate_from_schema documents an FK column shape and passes it straight
# through to SyntheticDataGenerator.generate_for_schemas() with no
# translation. These tests exercise the real SchemaLoader (no mocking) to
# make sure that documented shape is actually understood by the core engine
# — i.e. that a schema which passes validate_schema will really produce FK
# columns sampled from the parent table, not silently fall through to plain
# unrelated integers.

class TestFkSchemaFormatConsistency:

    def test_documented_fk_format_is_parsed_by_schema_loader(self):
        from syda.schema_loader import SchemaLoader

        orders_schema = {
            "order_id":    {"type": "integer", "primary_key": True},
            "customer_id": {"type": "foreign_key", "references": {"schema": "customers", "field": "customer_id"}},
            "amount":      {"type": "float", "min": 5.0, "max": 500.0},
        }
        _, _, _, foreign_keys, _, _ = SchemaLoader().load_schema(orders_schema)
        assert foreign_keys == {"customer_id": ("customers", "customer_id")}

    def test_documented_fk_format_string_shorthand_is_parsed(self):
        from syda.schema_loader import SchemaLoader

        orders_schema = {
            "order_id":    {"type": "integer", "primary_key": True},
            "customer_id": {"type": "foreign_key", "references": "customers.customer_id"},
        }
        _, _, _, foreign_keys, _, _ = SchemaLoader().load_schema(orders_schema)
        assert foreign_keys == {"customer_id": ("customers", "customer_id")}

    def test_validate_schema_relationship_matches_schema_loader(self):
        """A schema validate_schema calls "valid" with N relationships must
        yield the exact same foreign_keys mapping when run through the real
        SchemaLoader that generate_from_schema actually uses."""
        from syda.mcp_server import validate_schema
        from syda.schema_loader import SchemaLoader

        schema = {
            "customers": {
                "customer_id": {"type": "integer", "primary_key": True},
            },
            "orders": {
                "order_id":    {"type": "integer", "primary_key": True},
                "customer_id": {"type": "foreign_key", "references": {"schema": "customers", "field": "customer_id"}},
            },
        }
        result = validate_schema(schema)
        assert result["ok"] is True
        assert result["relationships"] == [{"child": "orders.customer_id", "parent": "customers.customer_id"}]

        _, _, _, foreign_keys, _, _ = SchemaLoader().load_schema(schema["orders"])
        assert foreign_keys == {"customer_id": ("customers", "customer_id")}


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

    def test_extra_kwargs_passed_through(self):
        """extra_kwargs reaches _build_generator."""
        df = self._make_df()
        captured = {}

        def capture_extra(provider, model, api_key, temperature, max_tokens,
                          generation_mode, batch_size, max_workers, extra_kwargs=None):
            captured["extra_kwargs"] = extra_kwargs
            return self._mock_generator(df)

        with patch("syda.mcp_server._build_generator", side_effect=capture_extra), \
             patch("syda.schema_loader.SchemaLoader") as MockLoader:
            MockLoader.return_value.load_schema.return_value = self.SIMPLE_SCHEMA["customers"]
            from syda.mcp_server import generate_from_schema
            generate_from_schema(
                schema=self.SIMPLE_SCHEMA,
                provider="openai_compatible",
                extra_kwargs={"base_url": "http://localhost:11434/v1", "api_key": "ollama"},
            )
        assert captured["extra_kwargs"] == {
            "base_url": "http://localhost:11434/v1", "api_key": "ollama"
        }


class TestBuildGenerator:

    def test_openai_compatible_requires_base_url(self):
        from syda.mcp_server import _build_generator
        with pytest.raises(ValueError, match="base_url"):
            _build_generator(
                provider="openai_compatible", model="llama3", api_key=None,
                temperature=0.8, max_tokens=4096, generation_mode="auto",
                batch_size=None, max_workers=1,
                extra_kwargs=None,   # missing base_url
            )

    def test_azure_requires_endpoint(self):
        from syda.mcp_server import _build_generator
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "key"}):
            with pytest.raises(ValueError, match="azure_endpoint"):
                _build_generator(
                    provider="azureopenai", model="gpt-4o", api_key=None,
                    temperature=0.8, max_tokens=4096, generation_mode="auto",
                    batch_size=None, max_workers=1,
                    extra_kwargs=None,   # missing azure_endpoint
                )

    def test_grok_gets_default_base_url(self):
        from syda.mcp_server import _build_generator
        with patch("syda.SyntheticDataGenerator") as MockGen, \
             patch("syda.ModelConfig") as MockMC, \
             patch.dict("os.environ", {"GROK_API_KEY": "xai-key"}):
            MockMC.return_value = MagicMock()
            _build_generator(
                provider="grok", model="grok-4.3", api_key=None,
                temperature=0.8, max_tokens=4096, generation_mode="auto",
                batch_size=None, max_workers=1,
                extra_kwargs=None,
            )
        call_kwargs = MockMC.call_args[1]
        assert call_kwargs["extra_kwargs"]["base_url"] == "https://api.x.ai/v1"

    def test_user_extra_kwargs_override_defaults(self):
        from syda.mcp_server import _build_generator
        with patch("syda.SyntheticDataGenerator") as MockGen, \
             patch("syda.ModelConfig") as MockMC, \
             patch.dict("os.environ", {"GROK_API_KEY": "xai-key"}):
            MockMC.return_value = MagicMock()
            _build_generator(
                provider="grok", model="grok-4.3", api_key=None,
                temperature=0.8, max_tokens=4096, generation_mode="auto",
                batch_size=None, max_workers=1,
                extra_kwargs={"base_url": "https://custom.endpoint/v1"},
            )
        call_kwargs = MockMC.call_args[1]
        assert call_kwargs["extra_kwargs"]["base_url"] == "https://custom.endpoint/v1"

    def test_explicit_api_key_forwarded_for_gemini(self):
        """An explicit api_key override must actually reach the generator,
        not be silently dropped (gemini has no 'if provider == ...' branch
        unless one is added)."""
        from syda.mcp_server import _build_generator
        with patch("syda.SyntheticDataGenerator") as MockGen, \
             patch("syda.ModelConfig") as MockMC:
            MockMC.return_value = MagicMock()
            _build_generator(
                provider="gemini", model="gemini-1.5-flash", api_key="AIza-explicit",
                temperature=0.8, max_tokens=4096, generation_mode="auto",
                batch_size=None, max_workers=1,
                extra_kwargs=None,
            )
        assert MockGen.call_args[1]["gemini_api_key"] == "AIza-explicit"

    def test_explicit_api_key_forwarded_for_azureopenai(self):
        """azureopenai has no api_key constructor kwarg on SyntheticDataGenerator —
        the override must be routed into extra_kwargs['api_key'] instead, since
        that's the only place pydantic-ai's AzureProvider reads it from."""
        from syda.mcp_server import _build_generator
        with patch("syda.SyntheticDataGenerator") as MockGen, \
             patch("syda.ModelConfig") as MockMC:
            MockMC.return_value = MagicMock()
            _build_generator(
                provider="azureopenai", model="gpt-4o", api_key="azure-explicit-key",
                temperature=0.8, max_tokens=4096, generation_mode="auto",
                batch_size=None, max_workers=1,
                extra_kwargs={"azure_endpoint": "https://my-resource.openai.azure.com/"},
            )
        call_kwargs = MockMC.call_args[1]
        assert call_kwargs["extra_kwargs"]["api_key"] == "azure-explicit-key"

    def test_azureopenai_extra_kwargs_api_key_not_overwritten(self):
        """If the caller already set extra_kwargs['api_key'] explicitly, the
        top-level api_key param must not clobber it."""
        from syda.mcp_server import _build_generator
        with patch("syda.SyntheticDataGenerator") as MockGen, \
             patch("syda.ModelConfig") as MockMC:
            MockMC.return_value = MagicMock()
            _build_generator(
                provider="azureopenai", model="gpt-4o", api_key="should-not-be-used",
                temperature=0.8, max_tokens=4096, generation_mode="auto",
                batch_size=None, max_workers=1,
                extra_kwargs={
                    "azure_endpoint": "https://my-resource.openai.azure.com/",
                    "api_key": "explicit-extra-kwargs-key",
                },
            )
        call_kwargs = MockMC.call_args[1]
        assert call_kwargs["extra_kwargs"]["api_key"] == "explicit-extra-kwargs-key"


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
            result = infer_schema_from_db(db_url="bad://url")
        assert result["ok"] is False
        assert "suggestion" in result

    def test_error_no_url_and_no_env(self):
        import os
        backup = {k: os.environ.pop(k) for k in ["SYDA_DB_URL", "DATABASE_URL"] if k in os.environ}
        try:
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db()
            assert result["ok"] is False
            assert "SYDA_DB_URL" in result["suggestion"]
        finally:
            os.environ.update(backup)

    def test_env_var_fallback(self):
        mock_schemas = {"users": {"id": {"type": "integer", "primary_key": True}}}
        with patch.dict("os.environ", {"SYDA_DB_URL": "sqlite:///test.db"}), \
             patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = mock_schemas
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db()   # no db_url passed
        assert result["ok"] is True
        assert "users" in result["schema"]

    def test_component_env_vars_fallback(self):
        """DB_HOST + DB_NAME + DB_USER + DB_PASSWORD builds a URL (syda convention)."""
        mock_schemas = {"orders": {"id": {"type": "integer", "primary_key": True}}}
        env = {
            "DB_HOST": "prod-db.internal",
            "DB_NAME": "myapp",
            "DB_USER": "john",
            "DB_PASSWORD": "secret",
            "DB_PORT": "5432",
        }
        with patch.dict("os.environ", env), \
             patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = mock_schemas
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db()
        assert result["ok"] is True
        # Confirm the constructed URL contains host and db name
        call_args = MockLoader.call_args[0][0]
        assert "prod-db.internal" in call_args
        assert "myapp" in call_args
        assert "john" in call_args

    def test_component_env_vars_url_encode_special_chars(self):
        """A password containing URL-special characters (@, :, /, %) must be
        percent-encoded, otherwise it breaks — or silently corrupts — the
        connection URL's host/port parsing."""
        mock_schemas = {"orders": {"id": {"type": "integer", "primary_key": True}}}
        env = {
            "DB_HOST": "prod-db.internal",
            "DB_NAME": "myapp",
            "DB_USER": "john",
            "DB_PASSWORD": "p@ss:w/rd%25",
            "DB_PORT": "5432",
        }
        with patch.dict("os.environ", env), \
             patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = mock_schemas
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db()
        assert result["ok"] is True
        call_args = MockLoader.call_args[0][0]
        # The raw password must not appear unescaped in the URL...
        assert "p@ss:w/rd%25" not in call_args
        # ...and the URL must still parse back to the original credentials.
        from urllib.parse import urlsplit, unquote
        parsed = urlsplit(call_args)
        assert unquote(parsed.password) == "p@ss:w/rd%25"
        assert parsed.hostname == "prod-db.internal"
        assert parsed.port == 5432

    def test_connection_error_redacts_password(self):
        """A DB connection failure must not leak the cleartext password back
        through the tool's error message, even if the underlying driver
        embeds the full DSN in its exception text."""
        from syda.mcp_server import infer_schema_from_db
        with patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.side_effect = Exception(
                "could not connect to server: "
                "postgresql://john:hunter2@prod-db.internal:5432/myapp"
            )
            result = infer_schema_from_db("postgresql://john:hunter2@prod-db.internal:5432/myapp")
        assert result["ok"] is False
        assert "hunter2" not in result["message"]
        assert "***:***@" in result["message"]

    def test_table_filter_passed_through(self):
        with patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = {}
            from syda.mcp_server import infer_schema_from_db
            infer_schema_from_db("sqlite:///test.db", tables=["customers"])
            MockLoader.return_value.load_schemas.assert_called_once_with(
                table_names=["customers"]
            )

    def test_fk_relationships_extracted(self):
        # Shape matches what DatabaseSchemaLoader (db_schema_loader.py) actually
        # produces: {"type": "foreign_key", "references": {"schema", "field"}}.
        mock_schemas = {
            "customers": {"id": {"type": "integer", "primary_key": True}},
            "orders": {
                "id": {"type": "integer", "primary_key": True},
                "customer_id": {
                    "type": "foreign_key",
                    "references": {"schema": "customers", "field": "id"},
                },
            },
        }
        with patch("syda.DatabaseSchemaLoader") as MockLoader:
            MockLoader.return_value.load_schemas.return_value = mock_schemas
            from syda.mcp_server import infer_schema_from_db
            result = infer_schema_from_db("sqlite:///test.db")

        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["child"] == "orders.customer_id"

"""
Tests for the templates module.
"""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from syda.templates import TemplateProcessor


@pytest.fixture
def processor():
    with patch("syda.templates.TemplateProcessor.__init__", lambda self, fp=None: None):
        p = TemplateProcessor.__new__(TemplateProcessor)
    import re
    p.placeholder_pattern = re.compile(r'{{\s*([a-zA-Z0-9_]+)\s*}}')
    p.file_processor = MagicMock()
    return p


class TestTemplateProcessor:

    # ── __init__ ──────────────────────────────────────────────────────────────

    def test_initialization_default(self):
        with patch("syda.unstructured.UnstructuredDataProcessor") as mock_udp:
            tp = TemplateProcessor()
        assert tp.file_processor is not None
        assert tp.placeholder_pattern is not None

    def test_initialization_custom_file_processor(self):
        mock_fp = MagicMock()
        tp = TemplateProcessor(file_processor=mock_fp)
        assert tp.file_processor is mock_fp

    # ── extract_placeholders ──────────────────────────────────────────────────

    def test_extract_placeholders_single(self, processor):
        result = processor.extract_placeholders("Hello {{ name }}, welcome!")
        assert result == {"name"}

    def test_extract_placeholders_multiple(self, processor):
        result = processor.extract_placeholders("Dear {{ customer_name }}, your ID is {{ customer_id }}.")
        assert result == {"customer_name", "customer_id"}

    def test_extract_placeholders_no_placeholders(self, processor):
        result = processor.extract_placeholders("No placeholders here.")
        assert result == set()

    def test_extract_placeholders_deduplicates(self, processor):
        result = processor.extract_placeholders("{{ name }} and {{ name }} again")
        assert result == {"name"}

    def test_extract_placeholders_no_spaces(self, processor):
        result = processor.extract_placeholders("{{name}} {{email}}")
        assert result == {"name", "email"}

    def test_extract_placeholders_extra_spaces(self, processor):
        result = processor.extract_placeholders("{{  field_one  }}")
        assert result == {"field_one"}

    # ── replace_placeholders ─────────────────────────────────────────────────

    def test_replace_placeholders_basic(self, processor):
        result = processor.replace_placeholders(
            "Hello {{ name }}!", {"name": "Alice"}
        )
        assert result == "Hello Alice!"

    def test_replace_placeholders_multiple_fields(self, processor):
        result = processor.replace_placeholders(
            "{{ greeting }}, {{ name }}!", {"greeting": "Hi", "name": "Bob"}
        )
        assert result == "Hi, Bob!"

    def test_replace_placeholders_numeric_value(self, processor):
        result = processor.replace_placeholders("Total: {{ amount }}", {"amount": 99.5})
        assert result == "Total: 99.5"

    def test_replace_placeholders_missing_key_leaves_placeholder(self, processor):
        result = processor.replace_placeholders("Hello {{ name }}!", {})
        assert "{{ name }}" in result

    def test_replace_placeholders_extra_keys_ignored(self, processor):
        result = processor.replace_placeholders("Hi {{ name }}", {"name": "X", "unused": "Y"})
        assert result == "Hi X"

    # ── get_template_content ─────────────────────────────────────────────────

    def test_get_template_content_file_not_found(self, processor):
        with pytest.raises(ValueError, match="not found"):
            processor.get_template_content("/nonexistent/path/template.txt")

    def test_get_template_content_success(self, processor):
        processor.file_processor.process_file.return_value = {"text": "Hello {{ name }}", "type": "txt"}
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Hello {{ name }}")
            tmp = f.name
        try:
            result = processor.get_template_content(tmp)
            assert result == "Hello {{ name }}"
        finally:
            os.unlink(tmp)

    def test_get_template_content_processor_error(self, processor):
        processor.file_processor.process_file.return_value = {"error": "read failed", "type": "txt"}
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="Error processing"):
                processor.get_template_content(tmp)
        finally:
            os.unlink(tmp)

    def test_get_template_content_no_text_key(self, processor):
        processor.file_processor.process_file.return_value = {"type": "pdf"}
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="Unable to extract text"):
                processor.get_template_content(tmp)
        finally:
            os.unlink(tmp)

    # ── create_schema_from_placeholders ──────────────────────────────────────

    def test_schema_name_field(self, processor):
        schema = processor.create_schema_from_placeholders({"customer_name"})
        assert schema["customer_name"] == "text"

    def test_schema_email_field(self, processor):
        schema = processor.create_schema_from_placeholders({"email"})
        assert schema["email"] == "email"

    def test_schema_phone_field(self, processor):
        schema = processor.create_schema_from_placeholders({"mobile_phone"})
        assert schema["mobile_phone"] == "phone"

    def test_schema_address_field(self, processor):
        schema = processor.create_schema_from_placeholders({"street_address"})
        assert schema["street_address"] == "address"

    def test_schema_date_field(self, processor):
        schema = processor.create_schema_from_placeholders({"invoice_date"})
        assert schema["invoice_date"] == "date"

    def test_schema_amount_field(self, processor):
        schema = processor.create_schema_from_placeholders({"total_price"})
        assert schema["total_price"] == "number"

    def test_schema_unknown_defaults_to_text(self, processor):
        schema = processor.create_schema_from_placeholders({"foobar"})
        assert schema["foobar"] == "text"

    def test_schema_empty_placeholders(self, processor):
        schema = processor.create_schema_from_placeholders(set())
        assert schema == {}

    def test_schema_multiple_fields(self, processor):
        schema = processor.create_schema_from_placeholders(
            {"customer_name", "email", "total_amount"}
        )
        assert schema["customer_name"] == "text"
        assert schema["email"] == "email"
        assert schema["total_amount"] == "number"

    # ── process_template_with_data ────────────────────────────────────────────

    def test_process_txt_template_no_output_path(self, processor):
        processor.file_processor.process_file.return_value = {
            "text": "Hello {{ name }}", "type": "txt"
        }
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello {{ name }}")
            tmp = f.name
        try:
            result = processor.process_template_with_data(tmp, {"name": "Alice"})
            assert result == "Hello Alice"
        finally:
            os.unlink(tmp)

    def test_process_txt_template_saves_to_output_path(self, processor):
        processor.file_processor.process_file.return_value = {
            "text": "Dear {{ name }}", "type": "txt"
        }
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Dear {{ name }}")
            src = f.name
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
            dst = out.name
        try:
            returned = processor.process_template_with_data(src, {"name": "Bob"}, output_path=dst)
            assert returned == dst
            assert open(dst).read() == "Dear Bob"
        finally:
            os.unlink(src)
            os.unlink(dst)

    def test_process_template_falls_back_to_direct_read(self, processor):
        # file_processor raises, but direct read should succeed
        processor.file_processor.process_file.return_value = {"error": "fail", "type": "txt"}
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hi {{ name }}")
            tmp = f.name
        try:
            result = processor.process_template_with_data(tmp, {"name": "Carol"})
            assert result == "Hi Carol"
        finally:
            os.unlink(tmp)

    # ── process_template_dataframes ───────────────────────────────────────────

    def test_process_template_dataframes_requires_output_dir(self, processor):
        with pytest.raises(ValueError, match="Output directory"):
            processor.process_template_dataframes({}, output_dir=None)

    def test_process_template_dataframes_generates_documents(self, processor):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real template file
            tpl_path = os.path.join(tmpdir, "letter.txt")
            with open(tpl_path, "w") as f:
                f.write("Hello {{ name }}")

            processor.file_processor.process_file.return_value = {
                "text": "Hello {{ name }}", "type": "txt"
            }

            df = pd.DataFrame({"name": ["Alice", "Bob"]})
            schema = {
                "__template_source__": tpl_path,
                "__input_file_type__": "txt",
                "__output_file_type__": "txt",
            }

            out_dir = os.path.join(tmpdir, "out")
            results = processor.process_template_dataframes(
                {"letters": (df, schema)}, output_dir=out_dir
            )

        assert "letters" in results
        assert len(results["letters"]) == 2

    def test_process_template_dataframes_skips_missing_template(self, processor, capsys):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"name": ["Alice"]})
            schema = {
                "__template_source__": "/nonexistent/template.txt",
                "__input_file_type__": "txt",
                "__output_file_type__": "txt",
            }
            # Should not raise — just warn and skip rows
            processor.process_template_dataframes(
                {"letters": (df, schema)}, output_dir=tmpdir
            )

        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Error" in captured.out
